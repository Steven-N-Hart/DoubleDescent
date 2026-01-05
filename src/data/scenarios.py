"""Data scenario configuration for synthetic survival data generation."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .types import CovariateType


@dataclass
class DataScenario:
    """Configuration for synthetic survival data generation.

    Attributes:
        name: Unique identifier (e.g., "baseline", "skewed")
        description: Human-readable description
        n_samples: Total number of samples (default: 1000)
        n_features: Total number of features (default: 20)
        n_predictive: Number of non-zero coefficients (default: 10)
        covariate_type: Type of covariate distribution
        correlation_matrix: Correlation structure (default: identity)
        n_categorical_features: Number of categorical features (for categorical/mixed)
        cardinality: Number of levels per categorical feature
        censoring_rate: Target censoring proportion (0.0 to 0.95)
        censoring_distribution: Censoring distribution type
        weibull_scale: Weibull lambda (scale) parameter
        weibull_shape: Weibull nu (shape) parameter
        coefficient_range: Range [min, max] for ground truth coefficients
    """

    # Identity
    name: str
    description: str = ""

    # Sample Configuration
    n_samples: int = 1000
    n_features: int = 20
    n_predictive: int = 10

    # Covariate Distribution
    covariate_type: CovariateType = CovariateType.GAUSSIAN
    correlation_matrix: Optional[np.ndarray] = None

    # For categorical covariates
    n_categorical_features: int = 0
    cardinality: int = 100

    # Censoring
    censoring_rate: float = 0.3
    censoring_distribution: str = "exponential"

    # Weibull Parameters
    weibull_scale: float = 0.5
    weibull_shape: float = 2.0

    # Ground Truth
    coefficient_range: Tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate the scenario configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.n_predictive > self.n_features:
            raise ValueError(
                f"n_predictive ({self.n_predictive}) cannot exceed "
                f"n_features ({self.n_features})"
            )

        if not 0.0 <= self.censoring_rate <= 0.95:
            raise ValueError(
                f"censoring_rate must be in [0.0, 0.95], got {self.censoring_rate}"
            )

        if self.n_samples < 100:
            raise ValueError(f"n_samples must be >= 100, got {self.n_samples}")

        if self.weibull_scale <= 0:
            raise ValueError(f"weibull_scale must be > 0, got {self.weibull_scale}")

        if self.weibull_shape <= 0:
            raise ValueError(f"weibull_shape must be > 0, got {self.weibull_shape}")

        if self.covariate_type in (CovariateType.CATEGORICAL, CovariateType.MIXED):
            if self.n_categorical_features <= 0:
                raise ValueError(
                    "n_categorical_features must be > 0 for categorical/mixed types"
                )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_predictive": self.n_predictive,
            "covariate_type": self.covariate_type.name.lower(),
            "n_categorical_features": self.n_categorical_features,
            "cardinality": self.cardinality,
            "censoring_rate": self.censoring_rate,
            "weibull_scale": self.weibull_scale,
            "weibull_shape": self.weibull_shape,
            "coefficient_range": list(self.coefficient_range),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DataScenario":
        """Create from dictionary.

        Args:
            data: Dictionary with scenario configuration.

        Returns:
            DataScenario instance.
        """
        # Convert string covariate_type to enum
        covariate_type_str = data.get("covariate_type", "gaussian")
        covariate_type = CovariateType[covariate_type_str.upper()]

        # Convert coefficient_range to tuple
        coef_range = data.get("coefficient_range", [-1.0, 1.0])
        if isinstance(coef_range, list):
            coef_range = tuple(coef_range)

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            n_samples=data.get("n_samples", 1000),
            n_features=data.get("n_features", 20),
            n_predictive=data.get("n_predictive", 10),
            covariate_type=covariate_type,
            n_categorical_features=data.get("n_categorical_features", 0),
            cardinality=data.get("cardinality", 100),
            censoring_rate=data.get("censoring_rate", 0.3),
            weibull_scale=data.get("weibull_scale", 0.5),
            weibull_shape=data.get("weibull_shape", 2.0),
            coefficient_range=coef_range,
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DataScenario":
        """Load scenario from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            DataScenario instance.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save scenario to JSON file.

        Args:
            path: Path to save JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Predefined scenarios
PREDEFINED_SCENARIOS = {
    "baseline": DataScenario(
        name="baseline",
        description="Baseline scenario with Gaussian covariates and 30% censoring",
        covariate_type=CovariateType.GAUSSIAN,
        censoring_rate=0.3,
    ),
    "skewed": DataScenario(
        name="skewed",
        description="Skewed scenario with log-normal covariates and 30% censoring",
        covariate_type=CovariateType.LOGNORMAL,
        censoring_rate=0.3,
    ),
    "high_cardinality": DataScenario(
        name="high_cardinality",
        description="High-cardinality categorical features and 30% censoring",
        covariate_type=CovariateType.CATEGORICAL,
        n_categorical_features=5,
        cardinality=100,
        censoring_rate=0.3,
    ),
    "imbalanced": DataScenario(
        name="imbalanced",
        description="Imbalanced scenario with 90% censoring (rare events)",
        covariate_type=CovariateType.GAUSSIAN,
        censoring_rate=0.9,
    ),
}


def get_scenario(name: str) -> DataScenario:
    """Get a predefined scenario by name.

    Args:
        name: Scenario name.

    Returns:
        DataScenario instance.

    Raises:
        ValueError: If scenario name is not found.
    """
    if name not in PREDEFINED_SCENARIOS:
        raise ValueError(
            f"Unknown scenario: {name}. "
            f"Available: {list(PREDEFINED_SCENARIOS.keys())}"
        )
    return PREDEFINED_SCENARIOS[name]
