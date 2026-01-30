"""Unified metric evaluator for survival models."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .concordance import calculate_c_index
from .brier import calculate_integrated_brier_score
from .likelihood import (
    calculate_negative_log_likelihood,
    breslow_estimator,
    compute_survival_function,
)
from .calibration import compute_calibration_metrics
from .results import MetricResult


@dataclass
class EvaluationData:
    """Container for data needed for evaluation.

    Attributes:
        X: Covariate matrix.
        T: Observed survival times.
        E: Event indicators.
    """

    X: np.ndarray
    T: np.ndarray
    E: np.ndarray

    def __post_init__(self):
        self.X = np.asarray(self.X)
        self.T = np.asarray(self.T).ravel()
        self.E = np.asarray(self.E).ravel()

    def __len__(self) -> int:
        return len(self.T)


class MetricEvaluator:
    """Evaluator for computing all survival metrics.

    This class provides a unified interface for computing C-index,
    Integrated Brier Score, and Negative Log Likelihood.

    Args:
        train_data: Training data for reference (needed for IBS censoring weights).
        n_time_points: Number of time points for survival function evaluation.
    """

    def __init__(
        self,
        train_data: EvaluationData,
        n_time_points: int = 100,
    ):
        self.train_data = train_data
        self.n_time_points = n_time_points

        # Precompute time grid based on training data
        self.time_grid = self._compute_time_grid()

    def _compute_time_grid(self) -> np.ndarray:
        """Compute time grid for survival function evaluation."""
        # Use quantiles of event times
        event_times = self.train_data.T[self.train_data.E == 1]
        if len(event_times) == 0:
            event_times = self.train_data.T

        min_time = np.percentile(event_times, 1)
        max_time = np.percentile(event_times, 99)

        return np.linspace(min_time, max_time, self.n_time_points)

    def evaluate(
        self,
        risk_scores: np.ndarray,
        eval_data: EvaluationData,
        split: str,
        run_id: str,
        epoch: int,
        compute_ibs: bool = True,
        compute_calibration: bool = True,
        train_risk_scores: Optional[np.ndarray] = None,
    ) -> MetricResult:
        """Compute all metrics for given risk scores.

        Args:
            risk_scores: Predicted log-hazard ratios.
            eval_data: Evaluation data (X, T, E).
            split: Data split name ("train", "val", "test").
            run_id: Run identifier.
            epoch: Training epoch.
            compute_ibs: Whether to compute IBS (slower).
            compute_calibration: Whether to compute calibration metrics.
            train_risk_scores: Risk scores on training data (for baseline hazard
                estimation in IBS). If None, uses zeros (less accurate).

        Returns:
            MetricResult with all computed metrics.
        """
        risk_scores = np.asarray(risk_scores).ravel()

        # Compute C-index
        c_index = calculate_c_index(
            risk_scores,
            eval_data.T,
            eval_data.E,
        )

        # Compute negative log likelihood
        try:
            nll = calculate_negative_log_likelihood(
                risk_scores,
                eval_data.T,
                eval_data.E,
            )
        except ValueError:
            # No events in data
            nll = np.nan

        # Compute IBS if requested
        ibs = np.nan
        survival_functions = None
        if compute_ibs:
            ibs, survival_functions = self._compute_ibs_with_survival(
                risk_scores, eval_data, train_risk_scores
            )

        # Compute calibration metrics if requested
        cal_large = None
        cal_slope = None
        ici = None
        if compute_calibration:
            cal_result = self._compute_calibration(
                risk_scores, eval_data, train_risk_scores, survival_functions
            )
            if cal_result is not None:
                cal_large = cal_result.calibration_in_the_large
                cal_slope = cal_result.calibration_slope
                ici = cal_result.ici

        return MetricResult(
            run_id=run_id,
            epoch=epoch,
            split=split,
            c_index=c_index,
            integrated_brier_score=ibs,
            neg_log_likelihood=nll,
            calibration_in_the_large=cal_large,
            calibration_slope=cal_slope,
            ici=ici,
        )

    def _compute_ibs(
        self,
        risk_scores: np.ndarray,
        eval_data: EvaluationData,
        train_risk_scores: Optional[np.ndarray] = None,
    ) -> float:
        """Compute Integrated Brier Score.

        Args:
            risk_scores: Predicted log-hazard ratios for evaluation data.
            eval_data: Evaluation data.
            train_risk_scores: Risk scores for training data (for baseline hazard).
                If None, uses zeros (original behavior, but less accurate).

        Returns:
            IBS value.
        """
        ibs, _ = self._compute_ibs_with_survival(risk_scores, eval_data, train_risk_scores)
        return ibs

    def _compute_ibs_with_survival(
        self,
        risk_scores: np.ndarray,
        eval_data: EvaluationData,
        train_risk_scores: Optional[np.ndarray] = None,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Compute Integrated Brier Score and return survival functions.

        Args:
            risk_scores: Predicted log-hazard ratios for evaluation data.
            eval_data: Evaluation data.
            train_risk_scores: Risk scores for training data (for baseline hazard).
                If None, uses zeros (original behavior, but less accurate).

        Returns:
            Tuple of (IBS value, survival_functions array or None on error).
        """
        try:
            # Estimate baseline hazard from training data
            # Use training risk scores if provided (proper Breslow method)
            # Otherwise fall back to zeros (original behavior)
            baseline_risk = (
                train_risk_scores if train_risk_scores is not None
                else np.zeros(len(self.train_data))
            )
            _, baseline_cumhaz = breslow_estimator(
                risk_scores=baseline_risk,
                event_times=self.train_data.T,
                event_indicators=self.train_data.E,
                time_points=self.time_grid,
            )

            # Compute survival functions for evaluation data
            survival_functions = compute_survival_function(
                risk_scores,
                baseline_cumhaz,
                self.time_grid,
            )

            # Compute IBS
            ibs = calculate_integrated_brier_score(
                survival_functions=survival_functions,
                time_points=self.time_grid,
                event_times_train=self.train_data.T,
                event_indicators_train=self.train_data.E,
                event_times_test=eval_data.T,
                event_indicators_test=eval_data.E,
            )

            return ibs, survival_functions

        except Exception:
            return np.nan, None

    def _compute_calibration(
        self,
        risk_scores: np.ndarray,
        eval_data: EvaluationData,
        train_risk_scores: Optional[np.ndarray] = None,
        survival_functions: Optional[np.ndarray] = None,
    ):
        """Compute calibration decomposition metrics.

        Args:
            risk_scores: Predicted log-hazard ratios for evaluation data.
            eval_data: Evaluation data.
            train_risk_scores: Risk scores for training data (for baseline hazard).
            survival_functions: Pre-computed survival functions (optional).

        Returns:
            CalibrationResult or None on error.
        """
        try:
            # Compute survival functions if not provided
            if survival_functions is None:
                baseline_risk = (
                    train_risk_scores if train_risk_scores is not None
                    else np.zeros(len(self.train_data))
                )
                _, baseline_cumhaz = breslow_estimator(
                    risk_scores=baseline_risk,
                    event_times=self.train_data.T,
                    event_indicators=self.train_data.E,
                    time_points=self.time_grid,
                )
                survival_functions = compute_survival_function(
                    risk_scores,
                    baseline_cumhaz,
                    self.time_grid,
                )

            return compute_calibration_metrics(
                risk_scores=risk_scores,
                survival_functions=survival_functions,
                time_grid=self.time_grid,
                event_times=eval_data.T,
                event_indicators=eval_data.E,
            )

        except Exception:
            return None

    def evaluate_all_splits(
        self,
        risk_scores_train: np.ndarray,
        risk_scores_val: np.ndarray,
        risk_scores_test: np.ndarray,
        val_data: EvaluationData,
        test_data: EvaluationData,
        run_id: str,
        epoch: int,
        compute_ibs: bool = True,
        compute_calibration: bool = True,
    ) -> Dict[str, MetricResult]:
        """Evaluate metrics on all data splits.

        Args:
            risk_scores_train: Risk scores for training data.
            risk_scores_val: Risk scores for validation data.
            risk_scores_test: Risk scores for test data.
            val_data: Validation data.
            test_data: Test data.
            run_id: Run identifier.
            epoch: Training epoch.
            compute_ibs: Whether to compute IBS.
            compute_calibration: Whether to compute calibration metrics.

        Returns:
            Dictionary mapping split name to MetricResult.
        """
        # Pass training risk scores for proper baseline hazard estimation in IBS
        train_risk = np.asarray(risk_scores_train).ravel()

        return {
            "train": self.evaluate(
                risk_scores_train,
                self.train_data,
                "train",
                run_id,
                epoch,
                compute_ibs,
                compute_calibration,
                train_risk_scores=train_risk,
            ),
            "val": self.evaluate(
                risk_scores_val,
                val_data,
                "val",
                run_id,
                epoch,
                compute_ibs,
                compute_calibration,
                train_risk_scores=train_risk,
            ),
            "test": self.evaluate(
                risk_scores_test,
                test_data,
                "test",
                run_id,
                epoch,
                compute_ibs,
                compute_calibration,
                train_risk_scores=train_risk,
            ),
        }
