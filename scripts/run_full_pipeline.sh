#!/bin/bash
# Full experiment pipeline - runs all remaining experiments and post-processing
# Usage: nohup ./scripts/run_full_pipeline.sh > pipeline.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

VENV=".venv/bin/python"
LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Starting Full Pipeline ==="

# 1. Wait for any running nonlinear experiments to complete
log "Step 1: Checking/waiting for nonlinear experiments..."
while pgrep -f "run_multi_seed.*nonlinear" > /dev/null; do
    log "  Nonlinear experiments still running, waiting 60s..."
    sleep 60
done

# 2. Check if nonlinear experiments need to be run/completed
NONLINEAR_COMPLETE=true
for seed in 42 123 456 789 1011; do
    PROGRESS_FILE="outputs/experiments/nonlinear_001_seed_${seed}/progress.json"
    if [ ! -f "$PROGRESS_FILE" ]; then
        NONLINEAR_COMPLETE=false
        break
    fi
    COMPLETED=$(grep -o '"n_completed": [0-9]*' "$PROGRESS_FILE" | grep -o '[0-9]*')
    if [ "$COMPLETED" != "11" ]; then
        NONLINEAR_COMPLETE=false
        break
    fi
done

if [ "$NONLINEAR_COMPLETE" = false ]; then
    log "  Running remaining nonlinear experiments..."
    $VENV scripts/run_multi_seed.py --config configs/experiments/nonlinear_sweep.json --seeds 5 --resume
fi
log "  Nonlinear experiments complete!"

# 3. Wait for any running embedding experiments to complete
log "Step 2: Checking/waiting for embedding experiments..."
while pgrep -f "run_embedding_experiment" > /dev/null; do
    log "  Embedding experiments still running, waiting 60s..."
    sleep 60
done

# 4. Check if embedding experiments completed, if not run them
if [ ! -f "outputs/experiments/categorical_embedding_embedding_aggregated/results/summary.csv" ]; then
    log "  Running embedding experiments..."
    $VENV scripts/run_embedding_experiment.py --scenario categorical_embedding --seeds 5 --epochs 10000
fi
log "  Embedding experiments complete!"

# 5. Run extended width experiments (4096, 8192) for full double descent recovery
log "Step 3: Running extended width experiments..."
EXTENDED_COMPLETE=true
for seed in 42 123 456 789 1011; do
    PROGRESS_FILE="outputs/experiments/nonlinear_extended_001_seed_${seed}/progress.json"
    if [ ! -f "$PROGRESS_FILE" ]; then
        EXTENDED_COMPLETE=false
        break
    fi
    COMPLETED=$(grep -o '"n_completed": [0-9]*' "$PROGRESS_FILE" | grep -o '[0-9]*')
    if [ "$COMPLETED" != "2" ]; then
        EXTENDED_COMPLETE=false
        break
    fi
done

if [ "$EXTENDED_COMPLETE" = false ]; then
    log "  Running extended width experiments (4096, 8192)..."
    $VENV scripts/run_multi_seed.py --config configs/experiments/nonlinear_extended_sweep.json --seeds 5 --resume
fi
log "  Extended width experiments complete!"

# 6. Aggregate nonlinear results
log "Step 4: Aggregating nonlinear results..."
$VENV -c "
from src.experiments.aggregation import aggregate_multi_seed_results
aggregate_multi_seed_results(
    experiment_ids=['nonlinear_001_seed_42', 'nonlinear_001_seed_123', 'nonlinear_001_seed_456', 'nonlinear_001_seed_789', 'nonlinear_001_seed_1011'],
    output_dir='outputs/experiments',
    base_id='nonlinear_001'
)
print('Aggregation complete')
"

# 7. Run baselines for nonlinear experiments
log "Step 5: Running baselines for nonlinear experiments..."
for seed in 42 123 456 789 1011; do
    EXPERIMENT_ID="nonlinear_001_seed_${seed}"
    BASELINE_FILE="outputs/experiments/${EXPERIMENT_ID}/baselines.csv"
    if [ ! -f "$BASELINE_FILE" ]; then
        log "  Running baselines for ${EXPERIMENT_ID}..."
        $VENV scripts/run_baselines.py --experiment "$EXPERIMENT_ID" --seed $seed
    fi
done
log "  Baselines complete!"

# 8. Generate updated figures
log "Step 6: Generating paper figures..."
$VENV scripts/generate_paper_figures.py --output-dir outputs/paper_figures

# 9. Build manuscript
log "Step 7: Building manuscript..."
cd manuscript && ./build.sh && cd ..

log "=== Pipeline Complete ==="
log "Results in: outputs/experiments/, outputs/paper_figures/, manuscript/main.pdf"
