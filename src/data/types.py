"""Core enumerations for the experiment framework."""

from enum import Enum, auto


class CovariateType(Enum):
    """Type of covariate distribution for synthetic data generation."""

    GAUSSIAN = auto()
    LOGNORMAL = auto()
    CATEGORICAL = auto()
    MIXED = auto()


class SweepType(Enum):
    """Type of model capacity sweep."""

    WIDTH = auto()  # Vary width, fix depth
    DEPTH = auto()  # Vary depth, fix width


class ExperimentStatus(Enum):
    """Status of an experiment."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class RunStatus(Enum):
    """Status of a single model training run."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
