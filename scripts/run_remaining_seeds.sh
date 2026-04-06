#!/bin/bash
# Run remaining 19 seeds for new experiments (B2-B4).
# Seed 42 already ran. This generates temp configs and runs each.
#
# Usage:
#   bash scripts/run_remaining_seeds.sh              # all configs, parallel
#   bash scripts/run_remaining_seeds.sh mixed_sweep   # filter to one config
#   JOBS=4 bash scripts/run_remaining_seeds.sh        # limit parallelism
#
# Requires: GNU parallel (sudo apt install parallel)
# Falls back to xargs -P if parallel is not installed.

set -e

SEEDS=(123 456 789 1011 1414 1618 1732 2024 2025 2026 2718 3141 3333 4444 5555 6666 7777 8888 9999)

CONFIGS=(
    "configs/experiments/mixed_sweep.json"
    "configs/experiments/high_dimensional_sweep.json"
    "configs/experiments/depth_sweep.json"
    "configs/experiments/depth4_width_sweep.json"
)

# Allow filtering to a single config
FILTER="${1:-}"

# Max parallel jobs (default: number of CPU cores)
MAX_JOBS="${JOBS:-$(nproc)}"

TMPDIR="outputs/experiments/temp_configs"
mkdir -p "$TMPDIR"

# Phase 1: Generate all temp configs, collect jobs to run
JOBLIST=$(mktemp)
trap "rm -f $JOBLIST" EXIT

for config in "${CONFIGS[@]}"; do
    base=$(basename "$config" .json)

    # Optional filter
    if [ -n "$FILTER" ] && [[ "$base" != *"$FILTER"* ]]; then
        continue
    fi

    # Extract experiment_id from config
    exp_id=$(python3 -c "import json; print(json.load(open('$config'))['experiment_id'])")

    for seed in "${SEEDS[@]}"; do
        run_id="${exp_id}_seed_${seed}"
        outdir="outputs/experiments/${run_id}"

        # Skip if already completed
        if [ -f "${outdir}/progress.json" ]; then
            done=$(python3 -c "import json; p=json.load(open('${outdir}/progress.json')); print(p['n_completed'] == p['n_total'])" 2>/dev/null || echo "False")
            if [ "$done" = "True" ]; then
                echo "SKIP: ${run_id} (already completed)"
                continue
            fi
        fi

        # Generate temp config with modified seed and experiment_id
        tmp_config="${TMPDIR}/${base}_seed_${seed}.json"
        python3 -c "
import json
with open('$config') as f:
    cfg = json.load(f)
cfg['seed'] = $seed
cfg['experiment_id'] = '${run_id}'
with open('$tmp_config', 'w') as f:
    json.dump(cfg, f, indent=2)
"
        echo "$tmp_config" >> "$JOBLIST"
    done
done

N_JOBS=$(wc -l < "$JOBLIST")
echo ""
echo "Launching $N_JOBS experiments with up to $MAX_JOBS parallel workers..."
echo ""

# Phase 2: Run in parallel
if command -v parallel &>/dev/null; then
    parallel -j "$MAX_JOBS" --bar --halt soon,fail=10% \
        "echo 'RUN: {}' && python -m src.cli.run_experiment --config {} 2>&1 | tail -1" \
        :::: "$JOBLIST"
else
    echo "(GNU parallel not found, falling back to xargs -P)"
    cat "$JOBLIST" | xargs -P "$MAX_JOBS" -I{} bash -c \
        "echo 'RUN: {}' && python -m src.cli.run_experiment --config {} --device cpu 2>&1 | tail -1 || echo 'FAILED: {}'"
fi

echo ""
echo "All seeds complete. Run: python scripts/aggregate_seeds.py"
