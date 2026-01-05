"""CLI for generating synthetic survival data."""

import argparse
import sys
from pathlib import Path

from ..data.generator import SurvivalDataGenerator
from ..data.scenarios import DataScenario, get_scenario, PREDEFINED_SCENARIOS


def main() -> int:
    """Main entry point for generate_data CLI.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.generate_data",
        description="Generate synthetic survival data without training.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scenario",
        type=str,
        choices=list(PREDEFINED_SCENARIOS.keys()),
        help="Predefined scenario name",
    )
    group.add_argument(
        "--config",
        type=Path,
        help="Path to custom scenario config file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory or file path",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        help="Override sample count",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["npz", "csv"],
        default="npz",
        help="Output format: npz, csv (default: npz)",
    )

    args = parser.parse_args()

    # Load scenario
    if args.scenario:
        scenario = get_scenario(args.scenario)
    else:
        if not args.config.exists():
            print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
            return 1
        scenario = DataScenario.from_json(args.config)

    # Override n_samples if specified
    if args.n_samples:
        scenario.n_samples = args.n_samples

    # Generate data
    print(f"Generating {scenario.name} data with {scenario.n_samples} samples...")
    generator = SurvivalDataGenerator(scenario, seed=args.seed)
    data = generator.generate()

    # Determine output path
    output_path = args.output
    if output_path.is_dir() or not output_path.suffix:
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f"{scenario.name}.{args.format}"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save data
    if args.format == "npz":
        data.save(output_path)
    elif args.format == "csv":
        import numpy as np
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header
            feature_cols = [f"X{i}" for i in range(data.X.shape[1])]
            writer.writerow(feature_cols + ["T", "E", "T_true"])
            # Data
            for i in range(len(data.T)):
                row = list(data.X[i]) + [data.T[i], data.E[i], data.T_true[i]]
                writer.writerow(row)

    print(f"Data saved to: {output_path}")
    print(f"  Samples: {len(data.T)}")
    print(f"  Features: {data.X.shape[1]}")
    print(f"  Event rate: {data.E.mean():.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
