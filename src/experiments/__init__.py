"""Experiment orchestration and management."""

from .config import Experiment
from .run import ExperimentRun
from .logging import TensorBoardLogger, CSVMetricsWriter, ExperimentLogger
from .sweep import CapacitySweep, SweepPoint
from .runner import ExperimentRunner, run_experiment
from .aggregation import (
    aggregate_multi_seed_results,
    load_aggregated_curves,
    compare_scenarios,
)

__all__ = [
    "Experiment",
    "ExperimentRun",
    "TensorBoardLogger",
    "CSVMetricsWriter",
    "ExperimentLogger",
    "CapacitySweep",
    "SweepPoint",
    "ExperimentRunner",
    "run_experiment",
    "aggregate_multi_seed_results",
    "load_aggregated_curves",
    "compare_scenarios",
]
