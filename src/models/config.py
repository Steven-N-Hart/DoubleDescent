"""Model configuration for DeepSurv architecture and training."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfiguration:
    """Configuration for a single DeepSurv model.

    Attributes:
        width: Neurons per hidden layer.
        depth: Number of hidden layers.
        activation: Activation function name.
        dropout: Dropout rate (0 = disabled).
        weight_decay: L2 regularization coefficient (0 = disabled).
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        optimizer: Optimizer type.
        retry_lr_factor: Learning rate reduction factor on failure retry.
    """

    # Architecture
    width: int
    depth: int = 2
    activation: str = "relu"
    dropout: float = 0.0

    # Regularization
    weight_decay: float = 0.0

    # Training
    epochs: int = 50000
    batch_size: int = 256
    learning_rate: float = 0.001
    optimizer: str = "adam"

    # Retry settings
    retry_lr_factor: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate the model configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.width < 1:
            raise ValueError(f"width must be >= 1, got {self.width}")

        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")

        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")

        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")

        if self.activation not in ("relu", "tanh", "selu"):
            raise ValueError(
                f"activation must be one of (relu, tanh, selu), got {self.activation}"
            )

        if self.optimizer not in ("adam", "sgd", "adamw"):
            raise ValueError(
                f"optimizer must be one of (adam, sgd, adamw), got {self.optimizer}"
            )

    def n_parameters(self, n_features: int) -> int:
        """Calculate approximate parameter count for P/N ratio.

        Args:
            n_features: Number of input features.

        Returns:
            Approximate number of trainable parameters.
        """
        # Input layer: n_features * width + width (bias)
        params = n_features * self.width + self.width

        # Hidden layers: (depth-1) * (width * width + width)
        for _ in range(self.depth - 1):
            params += self.width * self.width + self.width

        # Output layer: width * 1 + 1 (bias)
        params += self.width + 1

        return params

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "width": self.width,
            "depth": self.depth,
            "activation": self.activation,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "retry_lr_factor": self.retry_lr_factor,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfiguration":
        """Create from dictionary.

        Args:
            data: Dictionary with model configuration.

        Returns:
            ModelConfiguration instance.
        """
        return cls(
            width=data["width"],
            depth=data.get("depth", 2),
            activation=data.get("activation", "relu"),
            dropout=data.get("dropout", 0.0),
            weight_decay=data.get("weight_decay", 0.0),
            epochs=data.get("epochs", 50000),
            batch_size=data.get("batch_size", 256),
            learning_rate=data.get("learning_rate", 0.001),
            optimizer=data.get("optimizer", "adam"),
            retry_lr_factor=data.get("retry_lr_factor", 0.1),
        )

    def with_width(self, width: int) -> "ModelConfiguration":
        """Create a copy with a different width.

        Args:
            width: New width value.

        Returns:
            New ModelConfiguration with updated width.
        """
        config_dict = self.to_dict()
        config_dict["width"] = width
        return ModelConfiguration.from_dict(config_dict)

    def with_depth(self, depth: int) -> "ModelConfiguration":
        """Create a copy with a different depth.

        Args:
            depth: New depth value.

        Returns:
            New ModelConfiguration with updated depth.
        """
        config_dict = self.to_dict()
        config_dict["depth"] = depth
        return ModelConfiguration.from_dict(config_dict)

    def with_reduced_lr(self) -> "ModelConfiguration":
        """Create a copy with reduced learning rate for retry.

        Returns:
            New ModelConfiguration with learning_rate * retry_lr_factor.
        """
        config_dict = self.to_dict()
        config_dict["learning_rate"] = self.learning_rate * self.retry_lr_factor
        return ModelConfiguration.from_dict(config_dict)
