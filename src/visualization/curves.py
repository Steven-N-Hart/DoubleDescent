"""Double descent curve visualization."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class DoubleDescentCurve:
    """Aggregated metrics across the capacity sweep.

    Attributes:
        experiment_id: Parent experiment ID.
        metric_name: Name of the metric ("c_index", "ibs", "nll").
        split: Data split ("train", "val", "test").
        capacities: Parameter counts or widths.
        values: Metric values (NaN for failed runs).
        std_errors: Standard errors (if multiple seeds).
        interpolation_threshold: Estimated P ≈ N point.
        peak_location: Capacity at maximum error.
        peak_value: Error at peak.
        classical_minimum: Minimum error in classical regime.
        modern_minimum: Minimum error in over-parameterized regime.
    """

    experiment_id: str
    metric_name: str
    split: str

    capacities: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    std_errors: List[float] = field(default_factory=list)

    interpolation_threshold: Optional[int] = None
    peak_location: Optional[int] = None
    peak_value: Optional[float] = None
    classical_minimum: Optional[float] = None
    modern_minimum: Optional[float] = None

    def analyze(self, n_samples: int) -> None:
        """Analyze the curve to find key characteristics.

        Args:
            n_samples: Number of training samples (for P/N threshold).
        """
        if len(self.values) == 0:
            return

        values = np.array(self.values)
        capacities = np.array(self.capacities)

        # Filter out NaN values
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            return

        valid_values = values[valid_mask]
        valid_capacities = capacities[valid_mask]

        # Find interpolation threshold (closest to P ≈ N)
        self.interpolation_threshold = int(
            valid_capacities[np.argmin(np.abs(valid_capacities - n_samples))]
        )

        # For error metrics (IBS, NLL), find peak (maximum)
        # For C-index, find minimum (since higher is better)
        if self.metric_name == "c_index":
            # For C-index, the "peak" of error is the minimum of the metric
            peak_idx = np.argmin(valid_values)
            self.classical_minimum = np.max(
                valid_values[valid_capacities <= n_samples]
            ) if np.any(valid_capacities <= n_samples) else None
            self.modern_minimum = np.max(
                valid_values[valid_capacities > n_samples]
            ) if np.any(valid_capacities > n_samples) else None
        else:
            # For error metrics (IBS, NLL), peak is maximum
            peak_idx = np.argmax(valid_values)
            self.classical_minimum = np.min(
                valid_values[valid_capacities <= n_samples]
            ) if np.any(valid_capacities <= n_samples) else None
            self.modern_minimum = np.min(
                valid_values[valid_capacities > n_samples]
            ) if np.any(valid_capacities > n_samples) else None

        self.peak_location = int(valid_capacities[peak_idx])
        self.peak_value = float(valid_values[peak_idx])

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "metric_name": self.metric_name,
            "split": self.split,
            "capacities": self.capacities,
            "values": [v if not np.isnan(v) else None for v in self.values],
            "std_errors": self.std_errors,
            "interpolation_threshold": self.interpolation_threshold,
            "peak_location": self.peak_location,
            "peak_value": self.peak_value,
            "classical_minimum": self.classical_minimum,
            "modern_minimum": self.modern_minimum,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DoubleDescentCurve":
        """Create from dictionary."""
        values = [v if v is not None else np.nan for v in data.get("values", [])]
        return cls(
            experiment_id=data["experiment_id"],
            metric_name=data["metric_name"],
            split=data["split"],
            capacities=data.get("capacities", []),
            values=values,
            std_errors=data.get("std_errors", []),
            interpolation_threshold=data.get("interpolation_threshold"),
            peak_location=data.get("peak_location"),
            peak_value=data.get("peak_value"),
            classical_minimum=data.get("classical_minimum"),
            modern_minimum=data.get("modern_minimum"),
        )


def plot_double_descent_curve(
    curve: DoubleDescentCurve,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    show_threshold: bool = True,
    dpi: int = 300,
) -> Optional[plt.Figure]:
    """Plot a double descent curve.

    Args:
        curve: DoubleDescentCurve to plot.
        output_path: Path to save figure (without extension).
            If None, returns figure without saving.
        title: Plot title. If None, auto-generated.
        figsize: Figure size in inches.
        show_threshold: Whether to show interpolation threshold line.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure if output_path is None.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib not available. Install with: pip install matplotlib")

    fig, ax = plt.subplots(figsize=figsize)

    capacities = np.array(curve.capacities)
    values = np.array(curve.values)

    # Plot the curve
    valid_mask = ~np.isnan(values)
    ax.plot(
        capacities[valid_mask],
        values[valid_mask],
        'o-',
        linewidth=2,
        markersize=8,
        label=f"{curve.split} {curve.metric_name}",
    )

    # Add error bars if available
    if curve.std_errors:
        std_errors = np.array(curve.std_errors)
        ax.fill_between(
            capacities[valid_mask],
            values[valid_mask] - std_errors[valid_mask],
            values[valid_mask] + std_errors[valid_mask],
            alpha=0.3,
        )

    # Mark interpolation threshold
    if show_threshold and curve.interpolation_threshold:
        ax.axvline(
            curve.interpolation_threshold,
            color='red',
            linestyle='--',
            linewidth=1.5,
            label=f'Interpolation threshold (P≈N)',
        )

    # Mark peak
    if curve.peak_location and curve.peak_value:
        ax.scatter(
            [curve.peak_location],
            [curve.peak_value],
            color='red',
            s=150,
            zorder=5,
            marker='*',
            label=f'Peak ({curve.peak_location}, {curve.peak_value:.4f})',
        )

    # Formatting
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Model Capacity (Width)', fontsize=12)

    metric_labels = {
        "c_index": "Concordance Index",
        "ibs": "Integrated Brier Score",
        "nll": "Negative Log Likelihood",
    }
    ylabel = metric_labels.get(curve.metric_name, curve.metric_name)
    ax.set_ylabel(ylabel, fontsize=12)

    if title is None:
        title = f"Double Descent Curve: {curve.metric_name.upper()} ({curve.split})"
    ax.set_title(title, fontsize=14)

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or return
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as PNG
        fig.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')

        # Save as PDF
        fig.savefig(f"{output_path}.pdf", bbox_inches='tight')

        plt.close(fig)
        return None

    return fig


def plot_train_test_comparison(
    train_curve: DoubleDescentCurve,
    test_curve: DoubleDescentCurve,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
) -> Optional[plt.Figure]:
    """Plot train vs test curves on same axes.

    Args:
        train_curve: Training metrics curve.
        test_curve: Test metrics curve.
        output_path: Path to save figure.
        title: Plot title.
        figsize: Figure size.
        dpi: Resolution.

    Returns:
        Matplotlib figure if output_path is None.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib not available")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot train curve
    train_caps = np.array(train_curve.capacities)
    train_vals = np.array(train_curve.values)
    train_valid = ~np.isnan(train_vals)

    ax.plot(
        train_caps[train_valid],
        train_vals[train_valid],
        'o-',
        linewidth=2,
        markersize=8,
        color='#2ecc71',
        label='Train',
    )

    # Plot test curve
    test_caps = np.array(test_curve.capacities)
    test_vals = np.array(test_curve.values)
    test_valid = ~np.isnan(test_vals)

    ax.plot(
        test_caps[test_valid],
        test_vals[test_valid],
        's-',
        linewidth=2,
        markersize=8,
        color='#e74c3c',
        label='Test',
    )

    # Mark interpolation threshold
    if test_curve.interpolation_threshold:
        ax.axvline(
            test_curve.interpolation_threshold,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
            label='Interpolation threshold',
        )

    # Mark the test minimum (peak of error for C-index)
    if test_curve.peak_location and test_curve.peak_value:
        ax.scatter(
            [test_curve.peak_location],
            [test_curve.peak_value],
            color='#e74c3c',
            s=200,
            zorder=5,
            marker='*',
            edgecolors='black',
            linewidths=1,
        )

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Model Width', fontsize=12)

    metric_labels = {
        "c_index": "Concordance Index",
        "ibs": "Integrated Brier Score",
        "nll": "Negative Log Likelihood",
    }
    ylabel = metric_labels.get(test_curve.metric_name, test_curve.metric_name)
    ax.set_ylabel(ylabel, fontsize=12)

    if title is None:
        title = f"Train vs Test: {test_curve.metric_name.upper()}"
    ax.set_title(title, fontsize=14)

    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
        fig.savefig(f"{output_path}.pdf", bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def plot_generalization_gap(
    train_curve: DoubleDescentCurve,
    test_curve: DoubleDescentCurve,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
) -> Optional[plt.Figure]:
    """Plot the generalization gap (train - test) across capacity.

    Args:
        train_curve: Training metrics curve.
        test_curve: Test metrics curve.
        output_path: Path to save figure.
        title: Plot title.
        figsize: Figure size.
        dpi: Resolution.

    Returns:
        Matplotlib figure if output_path is None.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib not available")

    fig, ax = plt.subplots(figsize=figsize)

    train_caps = np.array(train_curve.capacities)
    train_vals = np.array(train_curve.values)
    test_vals = np.array(test_curve.values)

    # Compute gap (for C-index: train - test, higher gap = more overfitting)
    gap = train_vals - test_vals
    valid = ~np.isnan(gap)

    # Color by whether gap is positive or negative
    colors = ['#e74c3c' if g > 0 else '#2ecc71' for g in gap[valid]]

    ax.bar(
        range(len(train_caps[valid])),
        gap[valid],
        color=colors,
        edgecolor='black',
        linewidth=0.5,
    )

    # Set x-tick labels to widths
    ax.set_xticks(range(len(train_caps[valid])))
    ax.set_xticklabels([str(int(c)) for c in train_caps[valid]], rotation=45)

    ax.axhline(0, color='black', linewidth=1)

    ax.set_xlabel('Model Width', fontsize=12)
    ax.set_ylabel('Generalization Gap (Train - Test)', fontsize=12)

    if title is None:
        title = f"Generalization Gap: {test_curve.metric_name.upper()}"
    ax.set_title(title, fontsize=14)

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
        fig.savefig(f"{output_path}.pdf", bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def plot_learning_dynamics(
    tensorboard_dir: Union[str, Path],
    widths: List[int],
    metric: str = "val/c_index",
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    dpi: int = 300,
    max_epoch: Optional[int] = None,
) -> Optional[plt.Figure]:
    """Plot learning curves for different model sizes.

    Args:
        tensorboard_dir: Path to tensorboard logs directory.
        widths: List of widths to plot.
        metric: Metric tag to plot.
        output_path: Path to save figure.
        title: Plot title.
        figsize: Figure size.
        dpi: Resolution.
        max_epoch: Maximum epoch to show (None = all).

    Returns:
        Matplotlib figure if output_path is None.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib not available")

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        raise ImportError("TensorBoard not available. Install with: pip install tensorboard")

    import os

    fig, ax = plt.subplots(figsize=figsize)

    tensorboard_dir = Path(tensorboard_dir)

    # Color map for different widths
    cmap = plt.cm.viridis
    colors = [cmap(i / len(widths)) for i in range(len(widths))]

    for width, color in zip(widths, colors):
        run_dir = tensorboard_dir / f"width_{width:04d}_depth_02"
        if not run_dir.exists():
            continue

        # Find event file
        event_files = list(run_dir.glob("events.out.tfevents.*"))
        if not event_files:
            continue

        ea = event_accumulator.EventAccumulator(str(run_dir))
        ea.Reload()

        if metric not in ea.Tags().get('scalars', []):
            continue

        events = ea.Scalars(metric)
        epochs = [e.step for e in events]
        values = [e.value for e in events]

        if max_epoch:
            mask = [e <= max_epoch for e in epochs]
            epochs = [e for e, m in zip(epochs, mask) if m]
            values = [v for v, m in zip(values, mask) if m]

        ax.plot(
            epochs,
            values,
            linewidth=1.5,
            color=color,
            label=f'Width {width}',
            alpha=0.8,
        )

    ax.set_xlabel('Epoch', fontsize=12)

    metric_labels = {
        "val/c_index": "Validation C-Index",
        "train/c_index": "Train C-Index",
        "test/c_index": "Test C-Index",
        "train/nll": "Train NLL",
        "val/nll": "Validation NLL",
    }
    ylabel = metric_labels.get(metric, metric)
    ax.set_ylabel(ylabel, fontsize=12)

    if title is None:
        title = f"Learning Dynamics: {ylabel}"
    ax.set_title(title, fontsize=14)

    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
        fig.savefig(f"{output_path}.pdf", bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def plot_multi_metric_curves(
    curves: Dict[str, DoubleDescentCurve],
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 300,
) -> Optional[plt.Figure]:
    """Plot multiple metrics on subplots.

    Args:
        curves: Dictionary of {metric_name: DoubleDescentCurve}.
        output_path: Path to save figure.
        title: Overall title.
        figsize: Figure size.
        dpi: Resolution.

    Returns:
        Matplotlib figure if output_path is None.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib not available")

    n_metrics = len(curves)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_name, curve) in zip(axes, curves.items()):
        capacities = np.array(curve.capacities)
        values = np.array(curve.values)

        valid_mask = ~np.isnan(values)
        ax.plot(
            capacities[valid_mask],
            values[valid_mask],
            'o-',
            linewidth=2,
            markersize=6,
        )

        ax.set_xscale('log', base=2)
        ax.set_xlabel('Width')
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"{metric_name.upper()} ({curve.split})")
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
        fig.savefig(f"{output_path}.pdf", bbox_inches='tight')
        plt.close(fig)
        return None

    return fig
