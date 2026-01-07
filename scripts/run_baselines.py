#!/usr/bin/env python3
"""Run classical baseline models on experiment data.

Loads data from existing experiments and evaluates Cox PH and RSF baselines
for comparison with neural network results.

Usage:
    python scripts/run_baselines.py --experiment baseline_001
    python scripts/run_baselines.py --all
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generator import SurvivalData
from src.models.baselines import run_baselines, BaselineResults


def load_experiment_data(experiment_dir: Path) -> tuple:
    """Load train and test data from an experiment directory.

    Args:
        experiment_dir: Path to experiment output directory.

    Returns:
        Tuple of (train_data, test_data) as SurvivalData objects.
    """
    train_path = experiment_dir / "data" / "train.npz"
    test_path = experiment_dir / "data" / "test.npz"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Data not found in {experiment_dir}")

    train_data = SurvivalData.load(train_path)
    test_data = SurvivalData.load(test_path)

    return train_data, test_data


def run_baselines_for_experiment(
    experiment_dir: Path,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, BaselineResults]:
    """Run baseline models for a single experiment.

    Args:
        experiment_dir: Path to experiment output directory.
        seed: Random seed for RSF.
        verbose: Whether to print progress.

    Returns:
        Dictionary mapping model names to results.
    """
    if verbose:
        print(f"Loading data from {experiment_dir}...")

    train_data, test_data = load_experiment_data(experiment_dir)

    if verbose:
        print(f"  Train: {len(train_data.T)} samples, {np.mean(train_data.E):.1%} events")
        print(f"  Test: {len(test_data.T)} samples")
        print("Running baselines...")

    results = run_baselines(
        X_train=train_data.X,
        T_train=train_data.T,
        E_train=train_data.E,
        X_test=test_data.X,
        T_test=test_data.T,
        E_test=test_data.E,
        seed=seed,
    )

    if verbose:
        for name, result in results.items():
            ibs_str = f", IBS={result.ibs:.4f}" if result.ibs else ""
            print(f"  {name}: C-index={result.c_index:.4f}{ibs_str}")

    return results


def save_baseline_results(
    results: Dict[str, BaselineResults],
    output_dir: Path,
    experiment_id: str,
) -> None:
    """Save baseline results to files.

    Args:
        results: Dictionary of baseline results.
        output_dir: Output directory.
        experiment_id: Experiment identifier.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    csv_path = output_dir / f"{experiment_id}_baselines.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "c_index", "ibs"])
        writer.writeheader()
        for name, result in results.items():
            writer.writerow({
                "model": name,
                "c_index": result.c_index,
                "ibs": result.ibs if result.ibs else "",
            })

    # Save as JSON (for easier loading)
    json_path = output_dir / f"{experiment_id}_baselines.json"
    json_data = {
        name: {
            "model_name": result.model_name,
            "c_index": result.c_index,
            "ibs": result.ibs,
        }
        for name, result in results.items()
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Results saved to {csv_path}")


def get_all_experiments(base_dir: Path) -> List[str]:
    """Get list of all experiment IDs.

    Args:
        base_dir: Base experiments directory.

    Returns:
        List of experiment IDs.
    """
    experiments = []
    for exp_dir in base_dir.iterdir():
        if exp_dir.is_dir() and (exp_dir / "data" / "train.npz").exists():
            experiments.append(exp_dir.name)
    return sorted(experiments)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run classical baseline models on experiment data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run baselines for a specific experiment
    python scripts/run_baselines.py --experiment baseline_001

    # Run baselines for all experiments
    python scripts/run_baselines.py --all

    # Run for aggregated experiments
    python scripts/run_baselines.py --experiment baseline_001_aggregated
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment",
        type=str,
        help="Experiment ID to run baselines for",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run baselines for all experiments",
    )

    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("outputs/experiments"),
        help="Base experiments directory",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/baselines"),
        help="Output directory for baseline results",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for RSF",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Get experiments to process
    if args.all:
        experiments = get_all_experiments(args.experiments_dir)
        if not experiments:
            print(f"No experiments found in {args.experiments_dir}", file=sys.stderr)
            return 1
    else:
        experiments = [args.experiment]

    if verbose:
        print(f"Processing {len(experiments)} experiment(s)...")
        print()

    # Run baselines for each experiment
    all_results = {}
    for exp_id in experiments:
        exp_dir = args.experiments_dir / exp_id

        if not exp_dir.exists():
            print(f"Warning: Experiment directory not found: {exp_dir}", file=sys.stderr)
            continue

        try:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Experiment: {exp_id}")
                print('='*60)

            results = run_baselines_for_experiment(
                experiment_dir=exp_dir,
                seed=args.seed,
                verbose=verbose,
            )

            save_baseline_results(results, args.output_dir, exp_id)
            all_results[exp_id] = results

        except Exception as e:
            print(f"Error processing {exp_id}: {e}", file=sys.stderr)
            continue

    # Summary
    if verbose and len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Summary")
        print('='*60)

        for exp_id, results in all_results.items():
            print(f"\n{exp_id}:")
            for name, result in results.items():
                ibs_str = f", IBS={result.ibs:.4f}" if result.ibs else ""
                print(f"  {name}: C-index={result.c_index:.4f}{ibs_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
