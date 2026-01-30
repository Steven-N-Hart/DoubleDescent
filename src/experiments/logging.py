"""Experiment logging utilities for TensorBoard and CSV."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from ..metrics.results import MetricResult


class TensorBoardLogger:
    """TensorBoard logging wrapper.

    Provides a simplified interface for logging training metrics,
    model weights, and other diagnostics to TensorBoard.

    Args:
        log_dir: Directory for TensorBoard logs.
        run_name: Name for this run (creates subdirectory).
    """

    def __init__(self, log_dir: Union[str, Path], run_name: Optional[str] = None):
        if not HAS_TENSORBOARD:
            raise ImportError(
                "TensorBoard not available. Install with: pip install tensorboard"
            )

        self.log_dir = Path(log_dir)
        if run_name:
            self.log_dir = self.log_dir / run_name

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            tag: Metric name.
            value: Metric value.
            step: Training step (epoch).
        """
        if not np.isnan(value) and not np.isinf(value):
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_values: Dict[str, float], step: int) -> None:
        """Log multiple scalar values.

        Args:
            main_tag: Main tag (e.g., "loss").
            tag_values: Dictionary of {tag: value}.
            step: Training step.
        """
        # Filter out NaN/Inf values
        filtered = {k: v for k, v in tag_values.items() if not np.isnan(v) and not np.isinf(v)}
        if filtered:
            self.writer.add_scalars(main_tag, filtered, step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log a histogram of values.

        Args:
            tag: Histogram name.
            values: Array of values.
            step: Training step.
        """
        self.writer.add_histogram(tag, values, step)

    def log_metrics(self, metrics: MetricResult, prefix: str = "") -> None:
        """Log a MetricResult object.

        Args:
            metrics: MetricResult to log.
            prefix: Prefix for tags.
        """
        epoch = metrics.epoch
        split = metrics.split
        tag_prefix = f"{prefix}/{split}" if prefix else split

        self.log_scalar(f"{tag_prefix}/c_index", metrics.c_index, epoch)
        self.log_scalar(f"{tag_prefix}/ibs", metrics.integrated_brier_score, epoch)
        self.log_scalar(f"{tag_prefix}/nll", metrics.neg_log_likelihood, epoch)

        if metrics.gradient_norm is not None:
            self.log_scalar(f"{tag_prefix}/grad_norm", metrics.gradient_norm, epoch)
        if metrics.weight_norm is not None:
            self.log_scalar(f"{tag_prefix}/weight_norm", metrics.weight_norm, epoch)
        if metrics.learning_rate is not None:
            self.log_scalar(f"{tag_prefix}/learning_rate", metrics.learning_rate, epoch)
        if metrics.batch_loss_variance is not None:
            self.log_scalar(f"{tag_prefix}/batch_loss_var", metrics.batch_loss_variance, epoch)

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()

    def flush(self) -> None:
        """Flush pending writes."""
        self.writer.flush()


class CSVMetricsWriter:
    """CSV writer for experiment metrics.

    Writes metrics to a CSV file with one row per evaluation.

    Args:
        output_path: Path to CSV file.
        append: If True, append to existing file.
    """

    FIELDNAMES = [
        "epoch",
        "width",
        "depth",
        "split",
        "c_index",
        "ibs",
        "nll",
        "cal_slope",
        "cal_large",
        "ici",
        "grad_norm",
        "weight_norm",
        "lr",
        "batch_var",
        "timestamp",
    ]

    def __init__(
        self,
        output_path: Union[str, Path],
        width: int = 0,
        depth: int = 0,
        append: bool = False,
    ):
        self.output_path = Path(output_path)
        self.width = width
        self.depth = depth
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append and self.output_path.exists() else "w"
        self.file = open(self.output_path, mode, newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDNAMES)

        # Write header if new file
        if mode == "w":
            self.writer.writeheader()

    def write(self, metrics: MetricResult) -> None:
        """Write a metric result to CSV.

        Args:
            metrics: MetricResult to write.
        """
        row = {
            "epoch": metrics.epoch,
            "width": self.width,
            "depth": self.depth,
            "split": metrics.split,
            "c_index": metrics.c_index,
            "ibs": metrics.integrated_brier_score,
            "nll": metrics.neg_log_likelihood,
            "cal_slope": metrics.calibration_slope if metrics.calibration_slope is not None else "",
            "cal_large": metrics.calibration_in_the_large if metrics.calibration_in_the_large is not None else "",
            "ici": metrics.ici if metrics.ici is not None else "",
            "grad_norm": metrics.gradient_norm if metrics.gradient_norm is not None else "",
            "weight_norm": metrics.weight_norm if metrics.weight_norm is not None else "",
            "lr": metrics.learning_rate if metrics.learning_rate is not None else "",
            "batch_var": metrics.batch_loss_variance if metrics.batch_loss_variance is not None else "",
            "timestamp": metrics.timestamp.isoformat() if metrics.timestamp else "",
        }
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        """Close the CSV file."""
        self.file.close()


class ExperimentLogger:
    """Combined logger for TensorBoard and CSV.

    Provides a unified interface for logging to both TensorBoard
    and CSV simultaneously.

    Args:
        experiment_dir: Base directory for experiment outputs.
        run_id: Run identifier.
        width: Network width for this run.
        depth: Network depth for this run.
        use_tensorboard: Whether to log to TensorBoard.
    """

    def __init__(
        self,
        experiment_dir: Union[str, Path],
        run_id: str,
        width: int,
        depth: int,
        use_tensorboard: bool = True,
    ):
        self.experiment_dir = Path(experiment_dir)
        self.run_id = run_id
        self.width = width
        self.depth = depth

        # Setup run directory
        self.run_dir = self.experiment_dir / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV writer
        csv_path = self.run_dir / "metrics.csv"
        self.csv_writer = CSVMetricsWriter(csv_path, width=width, depth=depth)

        # Initialize TensorBoard writer
        self.tb_writer: Optional[TensorBoardLogger] = None
        if use_tensorboard and HAS_TENSORBOARD:
            tb_dir = self.experiment_dir / "tensorboard"
            self.tb_writer = TensorBoardLogger(tb_dir, run_id)

    def log_metrics(
        self,
        metrics: MetricResult,
        prefix: str = "",
    ) -> None:
        """Log metrics to both TensorBoard and CSV.

        Args:
            metrics: MetricResult to log.
            prefix: Optional prefix for TensorBoard tags.
        """
        # Log to CSV
        self.csv_writer.write(metrics)

        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.log_metrics(metrics, prefix)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar to TensorBoard only.

        Args:
            tag: Metric name.
            value: Metric value.
            step: Training step.
        """
        if self.tb_writer:
            self.tb_writer.log_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log a histogram to TensorBoard.

        Args:
            tag: Histogram name.
            values: Array of values.
            step: Training step.
        """
        if self.tb_writer:
            self.tb_writer.log_histogram(tag, values, step)

    def log_run_info(self, info: Dict[str, Any]) -> None:
        """Log run information to JSON file.

        Args:
            info: Dictionary of run information.
        """
        info_path = self.run_dir / "run_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2, default=str)

    def close(self) -> None:
        """Close all writers."""
        self.csv_writer.close()
        if self.tb_writer:
            self.tb_writer.close()

    def flush(self) -> None:
        """Flush all writers."""
        if self.tb_writer:
            self.tb_writer.flush()
