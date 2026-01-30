#!/bin/bash
# Full experiment pipeline - runs experiments in parallel and post-processing
# Usage: nohup ./scripts/run_full_pipeline.sh > pipeline.log 2>&1 &
#        ./scripts/run_full_pipeline.sh -j 4   # Run with 4 parallel jobs

set -e
cd "$(dirname "$0")/.."

VENV=".venv/bin/python"
LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"
PARALLEL_JOBS=4  # Default number of parallel jobs

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Starting Full Pipeline (parallel=$PARALLEL_JOBS) ==="

# Seeds for experiments
SEEDS=(42 123 456 789 1011 2024 2025 2026 3141 2718 1618 1414 1732 9999 8888 7777 6666 5555 4444 3333)

# All experiment configs to run
CONFIGS=(
    "configs/experiments/baseline_sweep.json:baseline_001"
    "configs/experiments/high_censoring_sweep.json:high_censoring_001"
    "configs/experiments/regularized_sweep.json:regularized_001"
    "configs/experiments/lognormal_sweep.json:lognormal_001"
    "configs/experiments/categorical_sweep.json:categorical_001"
    "configs/experiments/nonlinear_sweep.json:nonlinear_001"
)

# Create temp configs directory
mkdir -p outputs/experiments/temp_configs

# =============================================================================
# PHASE 1: Run all main experiments in parallel
# =============================================================================
log "Phase 1: Running main experiments with $PARALLEL_JOBS parallel jobs..."

# Build list of all experiments that need to run and create their config files
declare -a PENDING_CONFIGS=()

for config_pair in "${CONFIGS[@]}"; do
    config="${config_pair%%:*}"
    base_id="${config_pair##*:}"

    for seed in "${SEEDS[@]}"; do
        PROGRESS_FILE="outputs/experiments/${base_id}_seed_${seed}/progress.json"

        # Check if experiment is complete (11 widths for main experiments)
        if [ -f "$PROGRESS_FILE" ]; then
            COMPLETED=$(grep -o '"n_completed": [0-9]*' "$PROGRESS_FILE" 2>/dev/null | grep -o '[0-9]*' || echo "0")
            if [ "$COMPLETED" = "11" ]; then
                continue  # Already complete
            fi
        fi

        # Create temp config with seed embedded
        temp_config="outputs/experiments/temp_configs/${base_id}_seed_${seed}.json"
        $VENV -c "
import json
with open('$config') as f:
    cfg = json.load(f)
cfg['experiment_id'] = '${base_id}_seed_${seed}'
cfg['seed'] = $seed
with open('$temp_config', 'w') as f:
    json.dump(cfg, f, indent=2)
"
        PENDING_CONFIGS+=("$temp_config")
    done
done

# Count jobs
TOTAL_JOBS=${#PENDING_CONFIGS[@]}
if [ "$TOTAL_JOBS" -gt 0 ]; then
    log "  Found $TOTAL_JOBS experiments to run..."

    # Run experiments in batches using background processes
    RUNNING=0
    STARTED=0

    for temp_config in "${PENDING_CONFIGS[@]}"; do
        # Extract experiment ID from config filename
        exp_id=$(basename "$temp_config" .json)

        # Wait if we have too many running jobs
        while [ "$RUNNING" -ge "$PARALLEL_JOBS" ]; do
            wait -n 2>/dev/null || true
            RUNNING=$((RUNNING - 1))
        done

        STARTED=$((STARTED + 1))

        # Start new job in background
        (
            log "  [$STARTED/$TOTAL_JOBS] Starting: $exp_id"
            $VENV -m src.cli.run_experiment --config "$temp_config" --output-dir outputs/experiments --resume 2>&1 | \
                while IFS= read -r line; do echo "[$exp_id] $line"; done >> "$LOG_FILE"
            log "  Completed: $exp_id"
        ) &
        RUNNING=$((RUNNING + 1))
    done

    # Wait for all remaining jobs
    wait
    log "  All main experiments complete!"
else
    log "  All main experiments already complete!"
fi

log "Phase 1 complete!"

# =============================================================================
# PHASE 2: Run extended width experiments in parallel
# =============================================================================
log "Phase 2: Running extended width experiments..."

declare -a EXTENDED_CONFIGS=()

for seed in "${SEEDS[@]}"; do
    PROGRESS_FILE="outputs/experiments/nonlinear_extended_001_seed_${seed}/progress.json"

    if [ -f "$PROGRESS_FILE" ]; then
        COMPLETED=$(grep -o '"n_completed": [0-9]*' "$PROGRESS_FILE" 2>/dev/null | grep -o '[0-9]*' || echo "0")
        if [ "$COMPLETED" = "2" ]; then
            continue
        fi
    fi

    # Create temp config with seed embedded
    temp_config="outputs/experiments/temp_configs/nonlinear_extended_001_seed_${seed}.json"
    $VENV -c "
import json
with open('configs/experiments/nonlinear_extended_sweep.json') as f:
    cfg = json.load(f)
cfg['experiment_id'] = 'nonlinear_extended_001_seed_${seed}'
cfg['seed'] = $seed
with open('$temp_config', 'w') as f:
    json.dump(cfg, f, indent=2)
"
    EXTENDED_CONFIGS+=("$temp_config")
done

TOTAL_JOBS=${#EXTENDED_CONFIGS[@]}
if [ "$TOTAL_JOBS" -gt 0 ]; then
    log "  Found $TOTAL_JOBS extended experiments to run..."

    RUNNING=0
    STARTED=0

    for temp_config in "${EXTENDED_CONFIGS[@]}"; do
        exp_id=$(basename "$temp_config" .json)

        while [ "$RUNNING" -ge "$PARALLEL_JOBS" ]; do
            wait -n 2>/dev/null || true
            RUNNING=$((RUNNING - 1))
        done

        STARTED=$((STARTED + 1))

        (
            log "  [$STARTED/$TOTAL_JOBS] Starting: $exp_id"
            $VENV -m src.cli.run_experiment --config "$temp_config" --output-dir outputs/experiments --resume 2>&1 | \
                while IFS= read -r line; do echo "[$exp_id] $line"; done >> "$LOG_FILE"
            log "  Completed: $exp_id"
        ) &
        RUNNING=$((RUNNING + 1))
    done
    wait
    log "  All extended experiments complete!"
else
    log "  All extended experiments already complete!"
fi

log "Phase 2 complete!"

# =============================================================================
# PHASE 3: Aggregate results
# =============================================================================
log "Phase 3: Aggregating results..."

for config_pair in "${CONFIGS[@]}"; do
    base_id="${config_pair##*:}"

    log "  Aggregating $base_id..."
    $VENV -c "
from src.experiments.aggregation import aggregate_multi_seed_results
experiment_ids = [f'${base_id}_seed_{s}' for s in [42, 123, 456, 789, 1011, 2024, 2025, 2026, 3141, 2718, 1618, 1414, 1732, 9999, 8888, 7777, 6666, 5555, 4444, 3333]]
result = aggregate_multi_seed_results(experiment_ids, 'outputs/experiments', '${base_id}')
if result:
    print(f'  Aggregated {result[\"n_seeds\"]} seeds')
else:
    print('  No results to aggregate')
" 2>&1 | tee -a "$LOG_FILE"
done

# Aggregate extended experiments
log "  Aggregating nonlinear_extended_001..."
$VENV -c "
from src.experiments.aggregation import aggregate_multi_seed_results
experiment_ids = [f'nonlinear_extended_001_seed_{s}' for s in [42, 123, 456, 789, 1011, 2024, 2025, 2026, 3141, 2718, 1618, 1414, 1732, 9999, 8888, 7777, 6666, 5555, 4444, 3333]]
result = aggregate_multi_seed_results(experiment_ids, 'outputs/experiments', 'nonlinear_extended_001')
if result:
    print(f'  Aggregated {result[\"n_seeds\"]} seeds')
" 2>&1 | tee -a "$LOG_FILE"

log "Phase 3 complete!"

# =============================================================================
# PHASE 4: Run baselines in parallel
# =============================================================================
log "Phase 4: Running baselines..."

declare -a BASELINE_JOBS=()

for config_pair in "${CONFIGS[@]}"; do
    base_id="${config_pair##*:}"

    for seed in "${SEEDS[@]}"; do
        EXPERIMENT_ID="${base_id}_seed_${seed}"
        BASELINE_FILE="outputs/experiments/${EXPERIMENT_ID}/baselines.csv"

        if [ ! -f "$BASELINE_FILE" ] && [ -d "outputs/experiments/${EXPERIMENT_ID}" ]; then
            BASELINE_JOBS+=("$EXPERIMENT_ID:$seed")
        fi
    done
done

TOTAL_JOBS=${#BASELINE_JOBS[@]}
if [ "$TOTAL_JOBS" -gt 0 ]; then
    log "  Found $TOTAL_JOBS baseline runs needed..."

    RUNNING=0
    for job in "${BASELINE_JOBS[@]}"; do
        exp_id="${job%%:*}"
        seed="${job##*:}"

        while [ "$RUNNING" -ge "$PARALLEL_JOBS" ]; do
            wait -n 2>/dev/null || true
            RUNNING=$((RUNNING - 1))
        done

        (
            $VENV scripts/run_baselines.py --experiment "$exp_id" --seed "$seed" 2>&1 | \
                while IFS= read -r line; do echo "[baseline:$exp_id] $line"; done >> "$LOG_FILE"
        ) &
        RUNNING=$((RUNNING + 1))
    done
    wait
    log "  All baselines complete!"
else
    log "  All baselines already complete!"
fi

log "Phase 4 complete!"

# =============================================================================
# PHASE 5: Generate figures and build manuscript
# =============================================================================
log "Phase 5: Generating paper figures..."
$VENV scripts/regenerate_manuscript_figures.py 2>&1 | tee -a "$LOG_FILE"

log "Phase 6: Building manuscript..."
cd manuscript && ./build.sh 2>&1 | tee -a "../$LOG_FILE" && cd ..

log "=== Pipeline Complete ==="
log "Results in: outputs/experiments/, outputs/paper_figures/, manuscript/main.pdf"

# Cleanup temp configs
rm -rf outputs/experiments/temp_configs/
