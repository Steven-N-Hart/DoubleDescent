#!/bin/bash
# run_all_analyses.sh - Run all experiments and analyses for the double descent paper
#
# This script orchestrates:
#   1. Multi-seed experiments for all scenarios
#   2. Classical baseline comparisons
#   3. Figure generation
#
# Usage:
#   ./scripts/run_all_analyses.sh              # Run everything (sequential)
#   ./scripts/run_all_analyses.sh --parallel   # Run experiments in parallel
#   ./scripts/run_all_analyses.sh -j 4         # Parallel with max 4 concurrent jobs
#   ./scripts/run_all_analyses.sh --dry-run    # Show what would be run
#   ./scripts/run_all_analyses.sh --resume     # Resume interrupted runs

set -e  # Exit on error

source .venv/bin/activate

# Configuration
SEEDS="5"                          # Number of seeds per scenario
OUTPUT_DIR="outputs/experiments"
FIGURES_DIR="outputs/paper_figures"
BASELINES_DIR="outputs/baselines"
DEVICE="auto"                      # cuda, cpu, or auto

# Parse arguments
DRY_RUN=false
RESUME=false
SKIP_EXPERIMENTS=false
SKIP_BASELINES=false
SKIP_FIGURES=false
PARALLEL=false
MAX_PARALLEL=5  # Default: run up to 5 experiments simultaneously

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --skip-experiments)
            SKIP_EXPERIMENTS=true
            shift
            ;;
        --skip-baselines)
            SKIP_BASELINES=true
            shift
            ;;
        --skip-figures)
            SKIP_FIGURES=true
            shift
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --parallel|-p)
            PARALLEL=true
            shift
            ;;
        --max-parallel|-j)
            PARALLEL=true
            MAX_PARALLEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] $1"
    else
        eval "$1"
    fi
}

# ============================================================================
# PHASE 1: Run Multi-Seed Experiments
# ============================================================================

run_single_experiment() {
    local config="$1"
    local description="$2"
    local config_path="configs/experiments/${config}.json"
    local log_file="$OUTPUT_DIR/logs/${config}.log"

    mkdir -p "$OUTPUT_DIR/logs"

    if [ ! -f "$config_path" ]; then
        echo "[WARNING] Config not found: $config_path - skipping" >> "$log_file"
        return 1
    fi

    echo "[INFO] Starting: $description ($config)" >> "$log_file"

    RESUME_FLAG=""
    if [ "$RESUME" = true ]; then
        RESUME_FLAG="--resume"
    fi

    python scripts/run_multi_seed.py \
        --config "$config_path" \
        --seeds "$SEEDS" \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        $RESUME_FLAG >> "$log_file" 2>&1

    local status=$?
    if [ $status -eq 0 ]; then
        echo "[SUCCESS] Completed: $description" >> "$log_file"
    else
        echo "[ERROR] Failed: $description (exit code: $status)" >> "$log_file"
    fi
    return $status
}

export -f run_single_experiment
export SEEDS OUTPUT_DIR DEVICE RESUME

run_experiments() {
    log_info "=========================================="
    log_info "PHASE 1: Running Multi-Seed Experiments"
    log_info "=========================================="

    # Define experiments
    EXPERIMENTS=(
        "baseline_sweep:Baseline (Gaussian, 30% censoring)"
        "high_censoring_sweep:High Censoring (90%)"
        "regularized_sweep:Regularized (weight decay)"
        "lognormal_sweep:Log-Normal Covariates"
        "categorical_sweep:Categorical Features"
    )

    if [ "$PARALLEL" = true ] && [ "$DRY_RUN" = false ]; then
        log_info "Running experiments in PARALLEL (max $MAX_PARALLEL jobs)"
        mkdir -p "$OUTPUT_DIR/logs"

        # Build job list
        job_args=()
        for exp in "${EXPERIMENTS[@]}"; do
            config="${exp%%:*}"
            description="${exp##*:}"
            config_path="configs/experiments/${config}.json"
            if [ -f "$config_path" ]; then
                job_args+=("$config" "$description")
                log_info "  Queued: $description"
            else
                log_warn "Config not found: $config_path - skipping"
            fi
        done

        # Run in parallel using GNU parallel or xargs
        if command -v parallel &> /dev/null; then
            printf '%s\n' "${job_args[@]}" | parallel -N2 -j "$MAX_PARALLEL" run_single_experiment {1} {2}
        else
            # Fallback: simple background jobs with job control
            local pids=()
            local running=0
            for ((i=0; i<${#job_args[@]}; i+=2)); do
                config="${job_args[$i]}"
                description="${job_args[$((i+1))]}"

                run_single_experiment "$config" "$description" &
                pids+=($!)
                ((running++))

                # Wait if we've hit max parallel jobs
                if [ $running -ge $MAX_PARALLEL ]; then
                    wait -n 2>/dev/null || wait "${pids[0]}"
                    pids=("${pids[@]:1}")
                    ((running--))
                fi
            done

            # Wait for remaining jobs
            for pid in "${pids[@]}"; do
                wait "$pid"
            done
        fi

        log_success "All parallel experiments completed. Logs in: $OUTPUT_DIR/logs/"
    else
        # Sequential execution (original behavior)
        RESUME_FLAG=""
        if [ "$RESUME" = true ]; then
            RESUME_FLAG="--resume"
        fi

        for exp in "${EXPERIMENTS[@]}"; do
            config="${exp%%:*}"
            description="${exp##*:}"

            config_path="configs/experiments/${config}.json"

            if [ ! -f "$config_path" ]; then
                log_warn "Config not found: $config_path - skipping"
                continue
            fi

            log_info "Running: $description ($config)"
            log_info "  Config: $config_path"
            log_info "  Seeds: $SEEDS"

            cmd="python scripts/run_multi_seed.py \\
                --config $config_path \\
                --seeds $SEEDS \\
                --output-dir $OUTPUT_DIR \\
                --device $DEVICE \\
                $RESUME_FLAG"

            run_cmd "$cmd"

            if [ "$DRY_RUN" = false ]; then
                log_success "Completed: $description"
            fi
            echo ""
        done
    fi
}

# ============================================================================
# PHASE 2: Run Classical Baselines
# ============================================================================

run_baselines() {
    log_info "=========================================="
    log_info "PHASE 2: Running Classical Baselines"
    log_info "=========================================="

    # Get all experiment directories (including aggregated)
    log_info "Running Cox PH and RSF baselines for all experiments..."

    cmd="python scripts/run_baselines.py \\
        --all \\
        --experiments-dir $OUTPUT_DIR \\
        --output-dir $BASELINES_DIR"

    run_cmd "$cmd"

    if [ "$DRY_RUN" = false ]; then
        log_success "Baselines complete"
    fi
}

# ============================================================================
# PHASE 3: Generate Figures
# ============================================================================

generate_figures() {
    log_info "=========================================="
    log_info "PHASE 3: Generating Paper Figures"
    log_info "=========================================="

    cmd="python scripts/generate_paper_figures.py"
    run_cmd "$cmd"

    if [ "$DRY_RUN" = false ]; then
        log_success "Figures saved to $FIGURES_DIR"
    fi
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    echo ""
    echo "========================================================"
    echo "  Double Descent Paper - Complete Analysis Pipeline"
    echo "========================================================"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        log_warn "DRY RUN MODE - No commands will be executed"
        echo ""
    fi

    log_info "Configuration:"
    log_info "  Seeds per scenario: $SEEDS"
    log_info "  Output directory: $OUTPUT_DIR"
    log_info "  Device: $DEVICE"
    log_info "  Resume: $RESUME"
    log_info "  Parallel: $PARALLEL (max $MAX_PARALLEL jobs)"
    echo ""

    START_TIME=$(date +%s)

    # Phase 1: Experiments
    if [ "$SKIP_EXPERIMENTS" = false ]; then
        run_experiments
    else
        log_warn "Skipping experiments (--skip-experiments)"
    fi

    # Phase 2: Baselines
    if [ "$SKIP_BASELINES" = false ]; then
        run_baselines
    else
        log_warn "Skipping baselines (--skip-baselines)"
    fi

    # Phase 3: Figures
    if [ "$SKIP_FIGURES" = false ]; then
        generate_figures
    else
        log_warn "Skipping figures (--skip-figures)"
    fi

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo ""
    echo "========================================================"
    log_success "Analysis pipeline complete!"
    log_info "Total time: ${ELAPSED}s"
    echo "========================================================"
    echo ""
    echo "Outputs:"
    echo "  - Experiment results: $OUTPUT_DIR/"
    echo "  - Baseline results: $BASELINES_DIR/"
    echo "  - Paper figures: $FIGURES_DIR/"
    echo ""
}

main
