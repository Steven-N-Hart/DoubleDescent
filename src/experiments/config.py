"""Experiment configuration and management."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from ..data.scenarios import DataScenario
from ..data.types import ExperimentStatus, SweepType
from ..models.config import ModelConfiguration


@dataclass
class Experiment:
    """Complete experiment configuration.

    Attributes:
        experiment_id: Unique identifier (auto-generated if not provided).
        name: Human-readable experiment name.
        description: Optional description.
        seed: Random seed for all operations.
        data_scenario: Data generation configuration.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.
        width_sweep: List of widths to test.
        depth_sweep: List of depths to test.
        base_model_config: Base configuration (width/depth overridden per run).
        sweep_type: Type of sweep (WIDTH or DEPTH).
        created_at: Creation timestamp.
        started_at: Start timestamp.
        completed_at: Completion timestamp.
        status: Current experiment status.
    """

    # Identity
    name: str
    seed: int

    # Data
    data_scenario: DataScenario

    # Splits
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # Model Sweep
    width_sweep: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64, 128])
    depth_sweep: List[int] = field(default_factory=lambda: [2])
    base_model_config: Optional[ModelConfiguration] = None

    # Experiment Type
    sweep_type: SweepType = SweepType.WIDTH

    # Identity (auto-generated)
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Status
    status: ExperimentStatus = ExperimentStatus.PENDING

    def __post_init__(self) -> None:
        """Validate and set defaults after initialization."""
        if self.base_model_config is None:
            self.base_model_config = ModelConfiguration(width=self.width_sweep[0])

        self.validate()

    def validate(self) -> None:
        """Validate the experiment configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )

        if not self.width_sweep:
            raise ValueError("width_sweep cannot be empty")

        if not self.depth_sweep:
            raise ValueError("depth_sweep cannot be empty")

        if any(w < 1 for w in self.width_sweep):
            raise ValueError("All widths in width_sweep must be >= 1")

        if any(d < 1 for d in self.depth_sweep):
            raise ValueError("All depths in depth_sweep must be >= 1")

    def get_sweep_configurations(self) -> List[ModelConfiguration]:
        """Get all model configurations for the sweep.

        Returns:
            List of ModelConfiguration instances for each sweep point.
        """
        configs = []

        if self.sweep_type == SweepType.WIDTH:
            for width in self.width_sweep:
                for depth in self.depth_sweep:
                    config = self.base_model_config.with_width(width).with_depth(depth)
                    configs.append(config)
        else:  # DEPTH sweep
            for depth in self.depth_sweep:
                for width in self.width_sweep:
                    config = self.base_model_config.with_depth(depth).with_width(width)
                    configs.append(config)

        return configs

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "data": self.data_scenario.to_dict(),
            "splits": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            "model": {
                "widths": self.width_sweep,
                "depths": self.depth_sweep,
                "sweep_type": self.sweep_type.name.lower(),
                **{
                    k: v
                    for k, v in self.base_model_config.to_dict().items()
                    if k not in ("width", "depth")
                },
            },
            "training": {
                "epochs": self.base_model_config.epochs,
                "batch_size": self.base_model_config.batch_size,
                "learning_rate": self.base_model_config.learning_rate,
                "optimizer": self.base_model_config.optimizer,
            },
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Experiment":
        """Create from dictionary.

        Args:
            data: Dictionary with experiment configuration.

        Returns:
            Experiment instance.
        """
        # Parse data scenario
        data_config = data.get("data", {})
        if isinstance(data_config, dict):
            data_scenario = DataScenario.from_dict(data_config)
        else:
            # Assume it's a scenario name
            from ..data.scenarios import get_scenario
            data_scenario = get_scenario(data_config)

        # Parse model configuration
        model_config = data.get("model", {})
        training_config = data.get("training", {})

        base_config = ModelConfiguration(
            width=model_config.get("widths", [64])[0],
            depth=model_config.get("depths", [2])[0],
            activation=model_config.get("activation", "relu"),
            dropout=model_config.get("dropout", 0.0),
            weight_decay=model_config.get("weight_decay", 0.0),
            epochs=training_config.get("epochs", 50000),
            batch_size=training_config.get("batch_size", 256),
            learning_rate=training_config.get("learning_rate", 0.001),
            optimizer=training_config.get("optimizer", "adam"),
        )

        # Parse sweep type
        sweep_type_str = model_config.get("sweep_type", "width")
        sweep_type = SweepType[sweep_type_str.upper()]

        # Parse splits
        splits = data.get("splits", {})

        # Parse status
        status_str = data.get("status", "PENDING")
        status = ExperimentStatus[status_str.upper()]

        # Parse timestamps
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()

        started_at = data.get("started_at")
        if started_at and isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)

        completed_at = data.get("completed_at")
        if completed_at and isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)

        return cls(
            experiment_id=data.get("experiment_id", str(uuid.uuid4())[:8]),
            name=data.get("name", "Unnamed Experiment"),
            description=data.get("description", ""),
            seed=data["seed"],
            data_scenario=data_scenario,
            train_ratio=splits.get("train", 0.6),
            val_ratio=splits.get("val", 0.2),
            test_ratio=splits.get("test", 0.2),
            width_sweep=model_config.get("widths", [2, 4, 8, 16, 32, 64, 128]),
            depth_sweep=model_config.get("depths", [2]),
            base_model_config=base_config,
            sweep_type=sweep_type,
            status=status,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Experiment":
        """Load experiment from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            Experiment instance.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save experiment to JSON file.

        Args:
            path: Path to save JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def start(self) -> None:
        """Mark experiment as started."""
        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.now()

    def complete(self) -> None:
        """Mark experiment as completed."""
        self.status = ExperimentStatus.COMPLETED
        self.completed_at = datetime.now()

    def fail(self) -> None:
        """Mark experiment as failed."""
        self.status = ExperimentStatus.FAILED
        self.completed_at = datetime.now()
