"""CLI for generating visualizations from experiment results."""

import argparse
import json
import sys
from pathlib import Path

from ..visualization.curves import (
    DoubleDescentCurve,
    plot_double_descent_curve,
    plot_multi_metric_curves,
)


def main() -> int:
    """Main entry point for visualize CLI.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.visualize",
        description="Generate visualizations from experiment results.",
    )

    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Path to experiment output directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for figures (default: {experiment}/figures/)",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="png,pdf",
        help="Output formats, comma-separated (default: png,pdf)",
    )

    parser.add_argument(
        "--plots",
        type=str,
        default="all",
        help="Plot types to generate: double_descent, all (default: all)",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for raster formats (default: 300)",
    )

    args = parser.parse_args()

    # Validate experiment directory
    if not args.experiment.exists():
        print(f"ERROR: Experiment not found: {args.experiment}", file=sys.stderr)
        return 1

    curves_path = args.experiment / "results" / "curves.json"
    if not curves_path.exists():
        print(f"ERROR: No results found. Run the experiment first.", file=sys.stderr)
        return 1

    # Setup output directory
    output_dir = args.output or (args.experiment / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load curves
    with open(curves_path, "r") as f:
        curves_data = json.load(f)

    curves = {
        name: DoubleDescentCurve.from_dict(data)
        for name, data in curves_data.items()
    }

    print(f"Generating visualizations for experiment: {args.experiment.name}")

    # Generate double descent curves
    if args.plots in ("all", "double_descent"):
        for metric_name, curve in curves.items():
            output_path = output_dir / f"double_descent_{metric_name}"
            plot_double_descent_curve(
                curve,
                output_path=output_path,
                dpi=args.dpi,
            )
            print(f"  Saved: {output_path}.png, {output_path}.pdf")

        # Multi-metric plot
        output_path = output_dir / "double_descent_all"
        plot_multi_metric_curves(
            curves,
            output_path=output_path,
            dpi=args.dpi,
        )
        print(f"  Saved: {output_path}.png, {output_path}.pdf")

    print(f"\nVisualization complete. Output: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
