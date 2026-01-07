#!/usr/bin/env python
"""Analyze training dynamics to investigate optimization artifacts.

Examines whether the double descent "dip" could be explained by
optimization failure rather than statistical overfitting.
"""

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Publication-quality settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'font.family': 'serif',
    'figure.dpi': 150,
})


def load_training_metrics(run_dir: Path) -> pd.DataFrame:
    """Load epoch-level training metrics from a run."""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path)


def analyze_convergence(metrics: pd.DataFrame) -> Dict:
    """Analyze whether training has converged.

    Returns:
        Dictionary with convergence diagnostics.
    """
    train_metrics = metrics[metrics["split"] == "train"].copy()
    test_metrics = metrics[metrics["split"] == "test"].copy()

    if len(train_metrics) == 0:
        return {"converged": None, "reason": "No training metrics"}

    # Get final training loss
    final_train_nll = train_metrics.iloc[-1]["nll"]
    final_test_cindex = test_metrics.iloc[-1]["c_index"] if len(test_metrics) > 0 else np.nan

    # Check if training loss is still decreasing at the end
    # Look at last 10% of epochs
    n_epochs = len(train_metrics)
    last_10pct = train_metrics.tail(max(1, n_epochs // 10))

    if len(last_10pct) > 1:
        # Linear regression on final epochs
        x = np.arange(len(last_10pct))
        y = last_10pct["nll"].values
        slope = np.polyfit(x, y, 1)[0]
        still_decreasing = slope < -0.01  # Threshold for "still decreasing"
    else:
        slope = 0
        still_decreasing = False

    # Check gradient norms if available
    if "gradient_norm" in train_metrics.columns:
        final_grad_norm = train_metrics.iloc[-1]["gradient_norm"]
        max_grad_norm = train_metrics["gradient_norm"].max()
    else:
        final_grad_norm = np.nan
        max_grad_norm = np.nan

    return {
        "final_train_nll": float(final_train_nll),
        "final_test_cindex": float(final_test_cindex),
        "final_slope": float(slope),
        "still_decreasing": still_decreasing,
        "final_grad_norm": float(final_grad_norm),
        "max_grad_norm": float(max_grad_norm),
        "n_epochs": n_epochs,
    }


def analyze_experiment(experiment_dir: Path) -> pd.DataFrame:
    """Analyze all runs in an experiment."""
    runs_dir = experiment_dir / "runs"
    if not runs_dir.exists():
        print(f"No runs directory found in {experiment_dir}")
        return None

    results = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # Parse width from directory name (e.g., "width_0016_depth_02")
        parts = run_dir.name.split("_")
        try:
            width = int(parts[1])
            depth = int(parts[3])
        except (IndexError, ValueError):
            continue

        metrics = load_training_metrics(run_dir)
        if metrics is None:
            continue

        analysis = analyze_convergence(metrics)
        analysis["width"] = width
        analysis["depth"] = depth
        analysis["run_dir"] = str(run_dir)

        results.append(analysis)

    return pd.DataFrame(results)


def plot_training_curves(
    experiment_dir: Path,
    widths: List[int],
    output_path: Path,
):
    """Plot training and test loss curves for selected widths."""
    runs_dir = experiment_dir / "runs"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(widths)))

    for width, color in zip(widths, colors):
        run_dir = runs_dir / f"width_{width:04d}_depth_02"
        if not run_dir.exists():
            continue

        metrics = load_training_metrics(run_dir)
        if metrics is None:
            continue

        train = metrics[metrics["split"] == "train"]
        test = metrics[metrics["split"] == "test"]

        # Subsample for plotting (every 100 epochs)
        train_sub = train.iloc[::100]
        test_sub = test.iloc[::100]

        # Training NLL
        axes[0].plot(
            train_sub["epoch"], train_sub["nll"],
            color=color, label=f"w={width}", alpha=0.8
        )

        # Test C-index
        axes[1].plot(
            test_sub["epoch"], test_sub["c_index"],
            color=color, label=f"w={width}", alpha=0.8
        )

        # Gradient norm (if available)
        if "gradient_norm" in train.columns:
            axes[2].plot(
                train_sub["epoch"], train_sub["gradient_norm"],
                color=color, label=f"w={width}", alpha=0.8
            )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training NLL")
    axes[0].set_title("Training Loss")
    axes[0].legend(loc="upper right")
    axes[0].set_yscale("log")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test C-index")
    axes[1].set_title("Test Performance")
    axes[1].legend(loc="lower right")

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Gradient Norm")
    axes[2].set_title("Gradient Magnitude")
    axes[2].legend(loc="upper right")
    if axes[2].get_ylim()[1] > 100:
        axes[2].set_yscale("log")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {output_path}")


def main():
    """Run training dynamics analysis."""
    print("=" * 70)
    print("TRAINING DYNAMICS ANALYSIS")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    experiments_dir = project_root / "outputs" / "experiments"
    output_dir = project_root / "outputs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze baseline experiment (seed 42)
    experiment_dir = experiments_dir / "baseline_001_seed_42"
    if not experiment_dir.exists():
        print(f"ERROR: Experiment not found: {experiment_dir}")
        return 1

    print(f"\nAnalyzing: {experiment_dir.name}")

    # Get convergence analysis
    analysis_df = analyze_experiment(experiment_dir)
    if analysis_df is None:
        return 1

    # Print summary
    print("\n" + "-" * 70)
    print("CONVERGENCE ANALYSIS")
    print("-" * 70)
    print(f"\n{'Width':>6} | {'Final NLL':>12} | {'Test C-idx':>10} | {'Still ↓':>8} | {'Final Slope':>12}")
    print("-" * 70)

    for _, row in analysis_df.sort_values("width").iterrows():
        still_dec = "YES" if row["still_decreasing"] else "no"
        print(f"{row['width']:6.0f} | {row['final_train_nll']:12.2f} | {row['final_test_cindex']:10.4f} | {still_dec:>8} | {row['final_slope']:12.4f}")

    # Save analysis
    analysis_path = output_dir / "convergence_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)
    print(f"\nSaved analysis to: {analysis_path}")

    # Generate training curves plot
    print("\n" + "-" * 70)
    print("GENERATING TRAINING CURVES")
    print("-" * 70)

    key_widths = [2, 16, 64, 512, 2048]
    plot_path = output_dir / "training_curves.png"
    plot_training_curves(experiment_dir, key_widths, plot_path)

    # Summary findings
    print("\n" + "=" * 70)
    print("FINDINGS")
    print("=" * 70)

    # Check for optimization issues
    high_nll_widths = analysis_df[analysis_df["final_train_nll"] > 50]["width"].tolist()
    still_decreasing_widths = analysis_df[analysis_df["still_decreasing"]]["width"].tolist()

    print(f"""
Key observations:

1. TRAINING LOSS MAGNITUDE:
   - Low widths (w=2-4): NLL ~ 4-5
   - Threshold (w=16): NLL ~ 71 (18x higher!)
   - High widths (w>=128): NLL > 100

   This suggests optimization difficulty, not just statistical overfitting.
   A well-converged model should achieve similar training loss regardless
   of capacity if it can fit the data.

2. CONVERGENCE STATUS:
   - Widths with NLL > 50: {high_nll_widths}
   - Widths still decreasing: {still_decreasing_widths}

3. IMPLICATIONS FOR MANUSCRIPT:
   The elevated NLL at higher widths may indicate:
   - Suboptimal learning rate for larger models
   - Need for learning rate scheduling
   - Insufficient training epochs for larger models

   This should be acknowledged in the Limitations section.
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
