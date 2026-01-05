"""Experiment run tracking and management."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..data.types import RunStatus
from ..models.config import ModelConfiguration


@dataclass
class ExperimentRun:
    """A single model training run within an experiment.

    Attributes:
        run_id: Unique identifier within experiment.
        experiment_id: Parent experiment identifier.
        width: Network width for this run.
        depth: Network depth for this run.
        model_config: Full configuration for this run.
        status: Current run status.
        failure_reason: Reason if FAILED or SKIPPED.
        started_at: Start timestamp.
        completed_at: Completion timestamp.
        epochs_completed: Number of epochs completed.
        checkpoint_path: Path to checkpoint file.
        best_checkpoint_epoch: Epoch with best validation metric.
    """

    # Identity
    run_id: str
    experiment_id: str

    # Configuration
    width: int
    depth: int
    model_config: ModelConfiguration

    # Status
    status: RunStatus = RunStatus.PENDING
    failure_reason: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    epochs_completed: int = 0

    # Checkpoints
    checkpoint_path: Optional[Path] = None
    best_checkpoint_epoch: Optional[int] = None

    @classmethod
    def create(
        cls,
        experiment_id: str,
        width: int,
        depth: int,
        model_config: ModelConfiguration,
    ) -> "ExperimentRun":
        """Create a new experiment run.

        Args:
            experiment_id: Parent experiment ID.
            width: Network width.
            depth: Network depth.
            model_config: Model configuration.

        Returns:
            New ExperimentRun instance.
        """
        run_id = f"width_{width:04d}_depth_{depth:02d}"
        return cls(
            run_id=run_id,
            experiment_id=experiment_id,
            width=width,
            depth=depth,
            model_config=model_config,
        )

    def start(self) -> None:
        """Mark run as started."""
        self.status = RunStatus.RUNNING
        self.started_at = datetime.now()

    def complete(self, epochs: int, best_epoch: Optional[int] = None) -> None:
        """Mark run as completed.

        Args:
            epochs: Total epochs completed.
            best_epoch: Epoch with best validation metric.
        """
        self.status = RunStatus.COMPLETED
        self.completed_at = datetime.now()
        self.epochs_completed = epochs
        self.best_checkpoint_epoch = best_epoch

    def fail(self, reason: str) -> None:
        """Mark run as failed.

        Args:
            reason: Failure reason.
        """
        self.status = RunStatus.FAILED
        self.completed_at = datetime.now()
        self.failure_reason = reason

    def skip(self, reason: str) -> None:
        """Mark run as skipped.

        Args:
            reason: Skip reason.
        """
        self.status = RunStatus.SKIPPED
        self.completed_at = datetime.now()
        self.failure_reason = reason

    @property
    def is_complete(self) -> bool:
        """Check if run is complete (success or failure)."""
        return self.status in (
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.SKIPPED,
        )

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get run duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "width": self.width,
            "depth": self.depth,
            "model_config": self.model_config.to_dict(),
            "status": self.status.name,
            "failure_reason": self.failure_reason,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "epochs_completed": self.epochs_completed,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "best_checkpoint_epoch": self.best_checkpoint_epoch,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentRun":
        """Create from dictionary.

        Args:
            data: Dictionary with run data.

        Returns:
            ExperimentRun instance.
        """
        model_config = ModelConfiguration.from_dict(data["model_config"])
        status = RunStatus[data["status"]]

        started_at = data.get("started_at")
        if started_at and isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)

        completed_at = data.get("completed_at")
        if completed_at and isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)

        checkpoint_path = data.get("checkpoint_path")
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)

        return cls(
            run_id=data["run_id"],
            experiment_id=data["experiment_id"],
            width=data["width"],
            depth=data["depth"],
            model_config=model_config,
            status=status,
            failure_reason=data.get("failure_reason"),
            started_at=started_at,
            completed_at=completed_at,
            epochs_completed=data.get("epochs_completed", 0),
            checkpoint_path=checkpoint_path,
            best_checkpoint_epoch=data.get("best_checkpoint_epoch"),
        )
