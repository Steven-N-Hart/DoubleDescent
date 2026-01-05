"""Unit tests for metric calculations (T036)."""

import numpy as np
import pytest

from src.metrics.concordance import calculate_c_index
from src.metrics.likelihood import (
    calculate_negative_log_likelihood,
    breslow_estimator,
    compute_survival_function,
)
from src.metrics.results import MetricResult, RunSummary


class TestCIndex:
    """Tests for concordance index calculation."""

    def test_perfect_concordance(self):
        """Test C-index with perfect ranking."""
        # Higher risk = earlier event
        risk_scores = np.array([3.0, 2.0, 1.0, 0.0])
        event_times = np.array([1.0, 2.0, 3.0, 4.0])
        event_indicators = np.array([1, 1, 1, 1])

        c_index = calculate_c_index(risk_scores, event_times, event_indicators)

        assert c_index == 1.0

    def test_inverse_concordance(self):
        """Test C-index with inverse ranking."""
        # Higher risk = later event (wrong)
        risk_scores = np.array([0.0, 1.0, 2.0, 3.0])
        event_times = np.array([1.0, 2.0, 3.0, 4.0])
        event_indicators = np.array([1, 1, 1, 1])

        c_index = calculate_c_index(risk_scores, event_times, event_indicators)

        assert c_index == 0.0

    def test_random_concordance(self):
        """Test C-index with random ranking."""
        np.random.seed(42)
        n = 100
        risk_scores = np.random.randn(n)
        event_times = np.random.exponential(1.0, n)
        event_indicators = np.ones(n)

        c_index = calculate_c_index(risk_scores, event_times, event_indicators)

        # Random should be around 0.5
        assert 0.3 < c_index < 0.7

    def test_with_censoring(self):
        """Test C-index with censored observations."""
        risk_scores = np.array([2.0, 1.0, 0.0])
        event_times = np.array([1.0, 2.0, 3.0])
        event_indicators = np.array([1, 0, 1])  # Middle is censored

        c_index = calculate_c_index(risk_scores, event_times, event_indicators)

        # Should still work with censoring
        assert 0.0 <= c_index <= 1.0

    def test_all_censored(self):
        """Test C-index when all observations are censored."""
        risk_scores = np.array([1.0, 2.0, 3.0])
        event_times = np.array([1.0, 2.0, 3.0])
        event_indicators = np.array([0, 0, 0])

        c_index = calculate_c_index(risk_scores, event_times, event_indicators)

        # No comparable pairs, should return 0.5
        assert c_index == 0.5

    def test_empty_arrays_raise_error(self):
        """Test that empty arrays raise error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_c_index(np.array([]), np.array([]), np.array([]))

    def test_mismatched_shapes_raise_error(self):
        """Test that mismatched shapes raise error."""
        with pytest.raises(ValueError, match="same length"):
            calculate_c_index(
                np.array([1, 2]),
                np.array([1, 2, 3]),
                np.array([1, 1]),
            )


class TestNegativeLogLikelihood:
    """Tests for negative log partial likelihood."""

    def test_nll_basic(self):
        """Test basic NLL calculation."""
        risk_scores = np.array([1.0, 0.5, 0.0])
        event_times = np.array([1.0, 2.0, 3.0])
        event_indicators = np.array([1, 1, 1])

        nll = calculate_negative_log_likelihood(
            risk_scores, event_times, event_indicators
        )

        assert nll > 0
        assert not np.isnan(nll)

    def test_nll_with_censoring(self):
        """Test NLL with censored observations."""
        risk_scores = np.array([1.0, 0.5, 0.0])
        event_times = np.array([1.0, 2.0, 3.0])
        event_indicators = np.array([1, 0, 1])

        nll = calculate_negative_log_likelihood(
            risk_scores, event_times, event_indicators
        )

        assert nll > 0
        assert not np.isnan(nll)

    def test_nll_no_events_raises_error(self):
        """Test that no events raises error."""
        risk_scores = np.array([1.0, 0.5])
        event_times = np.array([1.0, 2.0])
        event_indicators = np.array([0, 0])

        with pytest.raises(ValueError, match="No events"):
            calculate_negative_log_likelihood(
                risk_scores, event_times, event_indicators
            )

    def test_nll_better_for_correct_ordering(self):
        """Test that correct ordering gives lower NLL."""
        event_times = np.array([1.0, 2.0, 3.0, 4.0])
        event_indicators = np.array([1, 1, 1, 1])

        # Correct: higher risk for earlier events
        correct_risk = np.array([2.0, 1.0, 0.0, -1.0])
        # Wrong: lower risk for earlier events
        wrong_risk = np.array([-1.0, 0.0, 1.0, 2.0])

        nll_correct = calculate_negative_log_likelihood(
            correct_risk, event_times, event_indicators
        )
        nll_wrong = calculate_negative_log_likelihood(
            wrong_risk, event_times, event_indicators
        )

        assert nll_correct < nll_wrong


class TestBreslowEstimator:
    """Tests for Breslow estimator."""

    def test_breslow_basic(self):
        """Test basic Breslow estimator."""
        risk_scores = np.array([0.0, 0.0, 0.0])
        event_times = np.array([1.0, 2.0, 3.0])
        event_indicators = np.array([1, 1, 1])

        times, cumhaz = breslow_estimator(risk_scores, event_times, event_indicators)

        # Cumulative hazard should be non-decreasing
        assert np.all(np.diff(cumhaz) >= 0)
        # Should have entries at event times
        assert len(times) == 3

    def test_breslow_with_time_points(self):
        """Test Breslow with specified time points."""
        risk_scores = np.array([0.0, 0.0, 0.0])
        event_times = np.array([1.0, 2.0, 3.0])
        event_indicators = np.array([1, 1, 1])
        time_points = np.array([0.5, 1.5, 2.5, 3.5])

        times, cumhaz = breslow_estimator(
            risk_scores, event_times, event_indicators, time_points
        )

        assert len(times) == 4
        assert len(cumhaz) == 4


class TestSurvivalFunction:
    """Tests for survival function computation."""

    def test_survival_shape(self):
        """Test survival function output shape."""
        risk_scores = np.array([0.0, 0.5, 1.0])
        baseline_cumhaz = np.array([0.1, 0.3, 0.5, 0.8])
        time_points = np.array([1.0, 2.0, 3.0, 4.0])

        survival = compute_survival_function(
            risk_scores, baseline_cumhaz, time_points
        )

        assert survival.shape == (3, 4)

    def test_survival_range(self):
        """Test survival function values are in [0, 1]."""
        risk_scores = np.array([0.0, 0.5, 1.0])
        baseline_cumhaz = np.array([0.1, 0.3, 0.5])
        time_points = np.array([1.0, 2.0, 3.0])

        survival = compute_survival_function(
            risk_scores, baseline_cumhaz, time_points
        )

        assert np.all(survival >= 0)
        assert np.all(survival <= 1)

    def test_survival_decreasing(self):
        """Test survival function is non-increasing over time."""
        risk_scores = np.array([0.5])
        baseline_cumhaz = np.array([0.1, 0.2, 0.4, 0.8])
        time_points = np.array([1.0, 2.0, 3.0, 4.0])

        survival = compute_survival_function(
            risk_scores, baseline_cumhaz, time_points
        )

        # Each row should be non-increasing
        assert np.all(np.diff(survival, axis=1) <= 0)

    def test_higher_risk_lower_survival(self):
        """Test that higher risk gives lower survival."""
        baseline_cumhaz = np.array([0.1, 0.3, 0.5])
        time_points = np.array([1.0, 2.0, 3.0])

        low_risk = compute_survival_function(
            np.array([0.0]), baseline_cumhaz, time_points
        )
        high_risk = compute_survival_function(
            np.array([2.0]), baseline_cumhaz, time_points
        )

        # Higher risk should have lower survival at all times
        assert np.all(high_risk <= low_risk)


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_create_result(self):
        """Test creating a metric result."""
        result = MetricResult(
            run_id="width_0064",
            epoch=100,
            split="train",
            c_index=0.75,
            integrated_brier_score=0.15,
            neg_log_likelihood=2.5,
        )

        assert result.run_id == "width_0064"
        assert result.epoch == 100
        assert result.c_index == 0.75

    def test_validation_valid_metrics(self):
        """Test validation with valid metrics."""
        result = MetricResult(
            run_id="test",
            epoch=0,
            split="test",
            c_index=0.75,
            integrated_brier_score=0.15,
            neg_log_likelihood=2.5,
        )

        assert result.validate()

    def test_validation_invalid_c_index(self):
        """Test validation with invalid C-index."""
        result = MetricResult(
            run_id="test",
            epoch=0,
            split="test",
            c_index=1.5,  # Invalid: > 1
            integrated_brier_score=0.15,
            neg_log_likelihood=2.5,
        )

        assert not result.validate()

    def test_validation_nan_values(self):
        """Test validation with NaN values."""
        result = MetricResult(
            run_id="test",
            epoch=0,
            split="test",
            c_index=np.nan,
            integrated_brier_score=0.15,
            neg_log_likelihood=2.5,
        )

        assert not result.validate()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = MetricResult(
            run_id="test",
            epoch=0,
            split="val",
            c_index=0.8,
            integrated_brier_score=0.12,
            neg_log_likelihood=1.5,
            gradient_norm=0.5,
        )

        data = result.to_dict()

        assert data["run_id"] == "test"
        assert data["c_index"] == 0.8
        assert data["grad_norm"] == 0.5

    def test_to_csv_row(self):
        """Test conversion to CSV row format."""
        result = MetricResult(
            run_id="test",
            epoch=0,
            split="test",
            c_index=0.75,
            integrated_brier_score=0.15,
            neg_log_likelihood=2.5,
        )

        row = result.to_csv_row()

        assert "c_index" in row
        assert "ibs" in row
        assert row["c_index"] == 0.75


class TestRunSummary:
    """Tests for RunSummary dataclass."""

    def test_create_summary(self):
        """Test creating a run summary."""
        summary = RunSummary(
            run_id="width_0064_depth_02",
            width=64,
            depth=2,
            final_train_c_index=0.85,
            final_val_c_index=0.78,
            final_test_c_index=0.76,
            final_train_ibs=0.10,
            final_val_ibs=0.12,
            final_test_ibs=0.13,
            best_val_epoch=5000,
            total_epochs=10000,
            n_parameters=5569,
        )

        assert summary.width == 64
        assert summary.depth == 2
        assert summary.final_test_c_index == 0.76

    def test_to_dict(self):
        """Test conversion to dictionary."""
        summary = RunSummary(
            run_id="test",
            width=32,
            depth=2,
            final_train_c_index=0.85,
            final_val_c_index=0.78,
            final_test_c_index=0.76,
            final_train_ibs=0.10,
            final_val_ibs=0.12,
            final_test_ibs=0.13,
            best_val_epoch=5000,
            total_epochs=10000,
            n_parameters=1000,
        )

        data = summary.to_dict()

        assert data["width"] == 32
        assert data["n_parameters"] == 1000
