"""CLI for generating visualizations from experiment results."""

import argparse
import json
import sys
from pathlib import Path

from ..visualization.curves import (
    DoubleDescentCurve,
    plot_double_descent_curve,
    plot_multi_metric_curves,
    plot_train_test_comparison,
    plot_generalization_gap,
    plot_learning_dynamics,
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
        help="Plot types: double_descent, train_test, gap, dynamics, paper, all (default: all)",
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

    plot_types = args.plots.split(",") if "," in args.plots else [args.plots]
    if "all" in plot_types:
        plot_types = ["double_descent", "train_test", "gap", "dynamics"]
    if "paper" in plot_types:
        plot_types = ["double_descent", "train_test", "gap", "dynamics"]

    # Generate double descent curves
    if "double_descent" in plot_types:
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

    # Generate train vs test comparison
    if "train_test" in plot_types:
        # Need train curves - load from tensorboard or separate file
        train_curves_path = args.experiment / "results" / "train_curves.json"
        if train_curves_path.exists():
            with open(train_curves_path, "r") as f:
                train_curves_data = json.load(f)
            train_curves = {
                name: DoubleDescentCurve.from_dict(data)
                for name, data in train_curves_data.items()
            }

            for metric_name in curves:
                if metric_name in train_curves:
                    output_path = output_dir / f"train_test_{metric_name}"
                    plot_train_test_comparison(
                        train_curves[metric_name],
                        curves[metric_name],
                        output_path=output_path,
                        dpi=args.dpi,
                    )
                    print(f"  Saved: {output_path}.png, {output_path}.pdf")
        else:
            print("  Skipping train_test plots (no train_curves.json found)")

    # Generate generalization gap plots
    if "gap" in plot_types:
        train_curves_path = args.experiment / "results" / "train_curves.json"
        if train_curves_path.exists():
            with open(train_curves_path, "r") as f:
                train_curves_data = json.load(f)
            train_curves = {
                name: DoubleDescentCurve.from_dict(data)
                for name, data in train_curves_data.items()
            }

            for metric_name in curves:
                if metric_name in train_curves:
                    output_path = output_dir / f"gap_{metric_name}"
                    plot_generalization_gap(
                        train_curves[metric_name],
                        curves[metric_name],
                        output_path=output_path,
                        dpi=args.dpi,
                    )
                    print(f"  Saved: {output_path}.png, {output_path}.pdf")
        else:
            print("  Skipping gap plots (no train_curves.json found)")

    # Generate learning dynamics plots
    if "dynamics" in plot_types:
        tb_dir = args.experiment / "tensorboard"
        if tb_dir.exists():
            # Get widths from curves
            widths = curves.get("c_index", list(curves.values())[0]).capacities if curves else []

            for metric in ["val/c_index", "train/c_index"]:
                metric_name = metric.replace("/", "_")
                output_path = output_dir / f"dynamics_{metric_name}"
                try:
                    plot_learning_dynamics(
                        tb_dir,
                        widths,
                        metric=metric,
                        output_path=output_path,
                        dpi=args.dpi,
                        max_epoch=1000,  # Show first 1000 epochs for clarity
                    )
                    print(f"  Saved: {output_path}.png, {output_path}.pdf")
                except Exception as e:
                    print(f"  Skipping {metric} dynamics: {e}")
        else:
            print("  Skipping dynamics plots (no tensorboard directory found)")

    print(f"\nVisualization complete. Output: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
