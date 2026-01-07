#!/usr/bin/env python3
"""Multi-seed experiment runner for uncertainty quantification.

Runs the same experiment configuration with multiple random seeds and
aggregates results for statistical analysis.

Usage:
    python scripts/run_multi_seed.py --config configs/experiments/baseline_sweep.json --seeds 5
    python scripts/run_multi_seed.py --config configs/experiments/baseline_sweep.json --seed-list 42,123,456
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.runner import run_experiment
from src.experiments.aggregation import aggregate_multi_seed_results


DEFAULT_SEEDS = [42, 123, 456, 789, 1011]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run experiment with multiple seeds for uncertainty quantification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with 5 seeds (default seed list)
    python scripts/run_multi_seed.py --config configs/experiments/baseline_sweep.json --seeds 5

    # Run with specific seeds
    python scripts/run_multi_seed.py --config configs/experiments/baseline_sweep.json --seed-list 42,123,456

    # Resume partially completed multi-seed run
    python scripts/run_multi_seed.py --config configs/experiments/baseline_sweep.json --seeds 5 --resume
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment config JSON",
    )

    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument(
        "--seeds",
        type=int,
        help="Number of seeds to use (uses default seed list)",
    )
    seed_group.add_argument(
        "--seed-list",
        type=str,
        help="Comma-separated list of seeds (e.g., '42,123,456')",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments"),
        help="Base output directory (default: outputs/experiments/)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint for each seed",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device: cuda, cpu, auto (default: auto)",
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training, only aggregate existing results",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    return parser.parse_args()


def get_seed_list(args: argparse.Namespace) -> List[int]:
    """Get list of seeds from arguments."""
    if args.seed_list:
        return [int(s.strip()) for s in args.seed_list.split(",")]
    else:
        return DEFAULT_SEEDS[: args.seeds]


def create_seeded_config(
    config_path: Path, seed: int, output_dir: Path
) -> Path:
    """Create a modified config file with the specified seed.

    Args:
        config_path: Path to original config.
        seed: New seed value.
        output_dir: Directory to write modified config.

    Returns:
        Path to the modified config file.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    # Modify experiment ID to include seed
    base_id = config["experiment_id"].rsplit("_seed_", 1)[0]
    config["experiment_id"] = f"{base_id}_seed_{seed}"
    config["seed"] = seed

    # Update description
    if "description" in config:
        config["description"] = f"{config['description']} (seed={seed})"

    # Write modified config
    output_dir.mkdir(parents=True, exist_ok=True)
    modified_path = output_dir / f"{base_id}_seed_{seed}.json"
    with open(modified_path, "w") as f:
        json.dump(config, f, indent=2)

    return modified_path


def run_multi_seed(
    config_path: Path,
    seeds: List[int],
    output_dir: Path,
    device: str = "auto",
    resume: bool = False,
    verbose: bool = True,
) -> dict:
    """Run experiment with multiple seeds.

    Args:
        config_path: Path to experiment config JSON.
        seeds: List of random seeds to use.
        output_dir: Base output directory.
        device: Device string.
        resume: Whether to resume from checkpoint.
        verbose: Whether to print progress.

    Returns:
        Dictionary with run results for each seed.
    """
    # Load base config to get experiment ID
    with open(config_path, "r") as f:
        base_config = json.load(f)

    base_id = base_config["experiment_id"]
    temp_config_dir = output_dir / "temp_configs"

    results = {}
    n_success = 0
    n_failed = 0

    if verbose:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Multi-Seed Experiment: {base_config.get('name', base_id)}", file=sys.stderr)
        print(f"Seeds: {seeds}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n[Seed {i+1}/{len(seeds)}] Running with seed={seed}", file=sys.stderr)
            print("-" * 40, file=sys.stderr)

        # Create seeded config
        seeded_config = create_seeded_config(config_path, seed, temp_config_dir)

        # Run experiment
        exit_code = run_experiment(
            config_path=seeded_config,
            output_dir=output_dir,
            device=device,
            resume=resume,
            dry_run=False,
            verbose=verbose,
        )

        results[seed] = {
            "exit_code": exit_code,
            "experiment_id": f"{base_id}_seed_{seed}",
            "success": exit_code == 0,
        }

        if exit_code == 0:
            n_success += 1
        else:
            n_failed += 1

    if verbose:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Multi-Seed Summary: {n_success} succeeded, {n_failed} failed", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

    return results


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Check config exists
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        return 1

    verbose = args.verbose and not args.quiet
    seeds = get_seed_list(args)

    if verbose:
        print(f"Seeds to run: {seeds}", file=sys.stderr)

    # Load base config to get experiment ID
    with open(args.config, "r") as f:
        base_config = json.load(f)
    base_id = base_config["experiment_id"]

    # Run experiments (unless aggregate-only)
    if not args.aggregate_only:
        results = run_multi_seed(
            config_path=args.config,
            seeds=seeds,
            output_dir=args.output_dir,
            device=args.device,
            resume=args.resume,
            verbose=verbose,
        )

        # Check for failures
        n_failed = sum(1 for r in results.values() if not r["success"])
        if n_failed == len(results):
            print("ERROR: All seed runs failed", file=sys.stderr)
            return 2

    # Aggregate results
    if verbose:
        print("\nAggregating results across seeds...", file=sys.stderr)

    experiment_ids = [f"{base_id}_seed_{seed}" for seed in seeds]
    aggregated = aggregate_multi_seed_results(
        experiment_ids=experiment_ids,
        output_dir=args.output_dir,
        base_id=base_id,
    )

    if aggregated is None:
        print("WARNING: Aggregation failed or no results to aggregate", file=sys.stderr)
        return 1

    if verbose:
        print(f"Aggregated results saved to: {args.output_dir}/{base_id}_aggregated/", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
