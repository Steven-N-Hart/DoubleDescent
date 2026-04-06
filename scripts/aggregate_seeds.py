"""Aggregate results across seeds for new experiments.

Usage:
    python scripts/aggregate_seeds.py
    python scripts/aggregate_seeds.py --experiment mixed_001
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


EXPERIMENTS = [
    "mixed_001",
    "high_dim_001",
    "depth_001",
    "depth4_width_001",
    "metabric",
    "support",
]


def aggregate_experiment(exp_prefix: str, output_base: Path) -> None:
    """Aggregate results across seeds for one experiment."""
    # Find all seed directories
    seed_dirs = sorted(output_base.glob(f"{exp_prefix}_seed_*"))
    if not seed_dirs:
        # Check if there's a single-seed run (no _seed_ suffix)
        single = output_base / exp_prefix
        if single.exists() and (single / "results" / "summary.csv").exists():
            seed_dirs = [single]
        else:
            print(f"  No results found for {exp_prefix}")
            return

    print(f"\n{'='*60}")
    print(f"Aggregating {exp_prefix}: {len(seed_dirs)} seed(s)")

    # Collect all results
    all_results = {}  # key -> list of metric dicts
    for sd in seed_dirs:
        summary = sd / "results" / "summary.csv"
        if not summary.exists():
            print(f"  WARNING: Missing {summary}")
            continue
        with open(summary) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (int(row["width"]), int(row["depth"]))
                if key not in all_results:
                    all_results[key] = []
                all_results[key].append({
                    k: float(v) if k != "width" and k != "depth" else int(float(v))
                    for k, v in row.items()
                })

    if not all_results:
        print(f"  No data to aggregate")
        return

    # Compute means and SDs
    agg_dir = output_base / f"{exp_prefix}_aggregated" / "results"
    agg_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key in sorted(all_results.keys()):
        entries = all_results[key]
        width, depth = key
        n_params = int(entries[0]["n_parameters"])

        c_vals = [e["c_index"] for e in entries if not np.isnan(e["c_index"])]
        ibs_vals = [e["ibs"] for e in entries if not np.isnan(e["ibs"])]
        nll_vals = [e["nll"] for e in entries if not np.isnan(e["nll"])]

        row = {
            "width": width,
            "depth": depth,
            "n_parameters": n_params,
            "n_seeds": len(entries),
            "c_index_mean": np.mean(c_vals) if c_vals else float("nan"),
            "c_index_std": np.std(c_vals, ddof=1) if len(c_vals) > 1 else 0.0,
            "ibs_mean": np.mean(ibs_vals) if ibs_vals else float("nan"),
            "ibs_std": np.std(ibs_vals, ddof=1) if len(ibs_vals) > 1 else 0.0,
            "nll_mean": np.mean(nll_vals) if nll_vals else float("nan"),
            "nll_std": np.std(nll_vals, ddof=1) if len(nll_vals) > 1 else 0.0,
        }
        rows.append(row)

        print(f"  w={width:>4d} d={depth} (n={len(entries):>2d}): "
              f"C={row['c_index_mean']:.3f}±{row['c_index_std']:.3f}  "
              f"IBS={row['ibs_mean']:.3f}±{row['ibs_std']:.3f}")

    # Save aggregated CSV
    out_path = agg_dir / "summary_aggregated.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed results")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Specific experiment to aggregate (default: all)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/experiments"),
                        help="Output directory")
    args = parser.parse_args()

    exps = [args.experiment] if args.experiment else EXPERIMENTS
    for exp in exps:
        aggregate_experiment(exp, args.output_dir)


if __name__ == "__main__":
    main()
