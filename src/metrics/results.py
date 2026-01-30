"""Metric result containers for experiment evaluation."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class MetricResult:
    """Metrics captured at a single evaluation point.

    Attributes:
        run_id: Associated run identifier.
        epoch: Training epoch number.
        split: Data split ("train", "val", or "test").
        c_index: Concordance index [0, 1].
        integrated_brier_score: IBS (lower is better).
        neg_log_likelihood: Cox partial likelihood loss.
        calibration_in_the_large: O/E ratio (1.0 = perfect).
        calibration_slope: Cox regression coefficient (1.0 = perfect).
        ici: Integrated Calibration Index (lower is better).
        gradient_norm: L2 norm of gradients.
        weight_norm: L2 norm of all weights.
        learning_rate: Current learning rate.
        batch_loss_variance: Variance across batches.
        timestamp: When metrics were computed.
    """

    # Context
    run_id: str
    epoch: int
    split: str

    # Core Survival Metrics
    c_index: float
    integrated_brier_score: float
    neg_log_likelihood: float

    # Calibration Decomposition Metrics
    calibration_in_the_large: Optional[float] = None
    calibration_slope: Optional[float] = None
    ici: Optional[float] = None

    # Training Diagnostics (optional for val/test splits)
    gradient_norm: Optional[float] = None
    weight_norm: Optional[float] = None
    learning_rate: Optional[float] = None
    batch_loss_variance: Optional[float] = None

    # Timestamp
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def validate(self) -> bool:
        """Validate metric values.

        Returns:
            True if all metrics are valid, False otherwise.
        """
        # Check for NaN values
        import math

        if math.isnan(self.c_index) or math.isnan(self.integrated_brier_score):
            return False

        # C-index should be in [0, 1]
        if not 0.0 <= self.c_index <= 1.0:
            return False

        # IBS should be non-negative
        if self.integrated_brier_score < 0:
            return False

        # Gradient and weight norms should be non-negative if present
        if self.gradient_norm is not None and self.gradient_norm < 0:
            return False

        if self.weight_norm is not None and self.weight_norm < 0:
            return False

        return True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "epoch": self.epoch,
            "split": self.split,
            "c_index": self.c_index,
            "ibs": self.integrated_brier_score,
            "nll": self.neg_log_likelihood,
            "cal_large": self.calibration_in_the_large,
            "cal_slope": self.calibration_slope,
            "ici": self.ici,
            "grad_norm": self.gradient_norm,
            "weight_norm": self.weight_norm,
            "lr": self.learning_rate,
            "batch_var": self.batch_loss_variance,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    def to_csv_row(self) -> dict:
        """Convert to CSV row format."""
        return {
            "epoch": self.epoch,
            "split": self.split,
            "c_index": self.c_index,
            "ibs": self.integrated_brier_score,
            "nll": self.neg_log_likelihood,
            "cal_large": self.calibration_in_the_large if self.calibration_in_the_large is not None else "",
            "cal_slope": self.calibration_slope if self.calibration_slope is not None else "",
            "ici": self.ici if self.ici is not None else "",
            "grad_norm": self.gradient_norm if self.gradient_norm is not None else "",
            "weight_norm": self.weight_norm if self.weight_norm is not None else "",
            "lr": self.learning_rate if self.learning_rate is not None else "",
            "batch_var": self.batch_loss_variance if self.batch_loss_variance is not None else "",
            "timestamp": self.timestamp.isoformat() if self.timestamp else "",
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetricResult":
        """Create from dictionary.

        Args:
            data: Dictionary with metric data.

        Returns:
            MetricResult instance.
        """
        timestamp = data.get("timestamp")
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            run_id=data["run_id"],
            epoch=data["epoch"],
            split=data["split"],
            c_index=data["c_index"],
            integrated_brier_score=data.get("ibs", data.get("integrated_brier_score")),
            neg_log_likelihood=data.get("nll", data.get("neg_log_likelihood")),
            calibration_in_the_large=data.get("cal_large", data.get("calibration_in_the_large")),
            calibration_slope=data.get("cal_slope", data.get("calibration_slope")),
            ici=data.get("ici"),
            gradient_norm=data.get("grad_norm", data.get("gradient_norm")),
            weight_norm=data.get("weight_norm"),
            learning_rate=data.get("lr", data.get("learning_rate")),
            batch_loss_variance=data.get("batch_var", data.get("batch_loss_variance")),
            timestamp=timestamp,
        )


@dataclass
class RunSummary:
    """Summary of metrics for a completed run.

    Attributes:
        run_id: Associated run identifier.
        width: Network width.
        depth: Network depth.
        final_train_c_index: Final training C-index.
        final_val_c_index: Final validation C-index.
        final_test_c_index: Final test C-index.
        final_train_ibs: Final training IBS.
        final_val_ibs: Final validation IBS.
        final_test_ibs: Final test IBS.
        final_test_cal_large: Final test calibration-in-the-large (O/E).
        final_test_cal_slope: Final test calibration slope.
        final_test_ici: Final test ICI.
        best_val_epoch: Epoch with best validation metric.
        total_epochs: Total epochs trained.
        n_parameters: Number of model parameters.
    """

    run_id: str
    width: int
    depth: int

    # Final metrics
    final_train_c_index: float
    final_val_c_index: float
    final_test_c_index: float
    final_train_ibs: float
    final_val_ibs: float
    final_test_ibs: float

    # Calibration metrics (test set)
    final_test_cal_large: Optional[float] = None
    final_test_cal_slope: Optional[float] = None
    final_test_ici: Optional[float] = None

    # Training info
    best_val_epoch: int = 0
    total_epochs: int = 0
    n_parameters: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "width": self.width,
            "depth": self.depth,
            "final_train_c_index": self.final_train_c_index,
            "final_val_c_index": self.final_val_c_index,
            "final_test_c_index": self.final_test_c_index,
            "final_train_ibs": self.final_train_ibs,
            "final_val_ibs": self.final_val_ibs,
            "final_test_ibs": self.final_test_ibs,
            "final_test_cal_large": self.final_test_cal_large,
            "final_test_cal_slope": self.final_test_cal_slope,
            "final_test_ici": self.final_test_ici,
            "best_val_epoch": self.best_val_epoch,
            "total_epochs": self.total_epochs,
            "n_parameters": self.n_parameters,
        }

    def to_csv_row(self) -> dict:
        """Convert to CSV row format for summary table."""
        data = self.to_dict()
        # Handle None values for CSV
        for key in ["final_test_cal_large", "final_test_cal_slope", "final_test_ici"]:
            if data[key] is None:
                data[key] = ""
        return data
