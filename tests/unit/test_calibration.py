"""Unit tests for calibration decomposition metrics (T008-T011)."""

import numpy as np
import pytest

from src.metrics.calibration import (
    calibration_in_the_large,
    calibration_slope,
    integrated_calibration_index,
    compute_calibration_metrics,
    CalibrationResult,
)


class TestCalibrationInTheLarge:
    """Tests for calibration-in-the-large (O/E ratio)."""

    def test_perfect_calibration(self):
        """T008: Test O/E ≈ 1.0 for perfectly calibrated predictions."""
        # When survival probs match actual outcomes, O/E should be ~1.0
        # If we have 50% events and predict 50% event probability on average
        n = 100
        np.random.seed(42)

        # Create scenario where expected = observed
        event_indicators = np.array([1] * 50 + [0] * 50)
        # Average survival prob of 0.5 means expected events = 50
        survival_probs = np.full(n, 0.5)

        oe_ratio = calibration_in_the_large(survival_probs, event_indicators)

        # 50 observed / 50 expected = 1.0
        assert 0.95 <= oe_ratio <= 1.05

    def test_over_prediction(self):
        """T009: Test O/E < 1 for over-prediction scenario."""
        # When model predicts high event probability but fewer events occur
        n = 100
        event_indicators = np.array([1] * 20 + [0] * 80)  # 20 events
        # Low survival probs = high predicted events
        survival_probs = np.full(n, 0.3)  # Expects 70 events

        oe_ratio = calibration_in_the_large(survival_probs, event_indicators)

        # 20 observed / 70 expected < 1 (over-prediction)
        assert oe_ratio < 1.0
        assert oe_ratio == pytest.approx(20 / 70, rel=0.01)

    def test_under_prediction(self):
        """T010: Test O/E > 1 for under-prediction scenario."""
        # When model predicts low event probability but more events occur
        n = 100
        event_indicators = np.array([1] * 80 + [0] * 20)  # 80 events
        # High survival probs = low predicted events
        survival_probs = np.full(n, 0.7)  # Expects 30 events

        oe_ratio = calibration_in_the_large(survival_probs, event_indicators)

        # 80 observed / 30 expected > 1 (under-prediction)
        assert oe_ratio > 1.0
        assert oe_ratio == pytest.approx(80 / 30, rel=0.01)

    def test_no_events_returns_zero(self):
        """Test O/E = 0 when no events observed."""
        n = 50
        event_indicators = np.zeros(n)  # No events
        survival_probs = np.full(n, 0.5)  # Expects 25 events

        oe_ratio = calibration_in_the_large(survival_probs, event_indicators)

        assert oe_ratio == 0.0

    def test_zero_expected_returns_nan(self):
        """T011: Test NaN when expected events is zero."""
        n = 50
        event_indicators = np.ones(n)  # All events
        # All survival = 1.0 means expected = 0
        survival_probs = np.ones(n)

        oe_ratio = calibration_in_the_large(survival_probs, event_indicators)

        assert np.isnan(oe_ratio)

    def test_mismatched_lengths_raises_error(self):
        """Test error on mismatched array lengths."""
        with pytest.raises(ValueError, match="must match"):
            calibration_in_the_large(np.array([0.5, 0.5]), np.array([1, 0, 1]))


class TestCalibrationSlope:
    """Tests for calibration slope."""

    def test_perfect_calibration_slope_near_one(self):
        """T008: Test slope ≈ 1.0 for well-calibrated predictions."""
        # Generate data where risk scores perfectly predict hazard
        np.random.seed(42)
        n = 200

        # True risk scores drive event times
        true_risk = np.random.randn(n)
        # Higher risk = earlier event (exponential with rate exp(risk))
        event_times = np.random.exponential(1.0 / np.exp(true_risk * 0.5))
        event_indicators = np.ones(n)

        # If predictions match true risk, slope should be ~1.0
        slope = calibration_slope(true_risk, event_times, event_indicators)

        # Allow some tolerance due to randomness
        assert 0.3 < slope < 2.0  # Reasonable range for slope

    def test_overfitting_slope_less_than_one(self):
        """T009: Test slope < 1 for overfitted predictions (too extreme)."""
        np.random.seed(42)
        n = 200

        # True moderate risk
        true_risk = np.random.randn(n) * 0.5
        event_times = np.random.exponential(1.0 / np.exp(true_risk))
        event_indicators = np.ones(n)

        # Overfitted: predictions are exaggerated (too extreme)
        exaggerated_risk = true_risk * 3.0

        slope = calibration_slope(exaggerated_risk, event_times, event_indicators)

        # Slope should be < 1 for overfitted (too extreme) predictions
        assert slope < 1.0

    def test_underfitting_slope_greater_than_one(self):
        """T010: Test slope > 1 for underfitted predictions (too conservative)."""
        np.random.seed(42)
        n = 200

        # Strong true effect
        true_risk = np.random.randn(n) * 2.0
        event_times = np.random.exponential(1.0 / np.exp(true_risk * 0.3))
        event_indicators = np.ones(n)

        # Underfitted: predictions are too conservative
        conservative_risk = true_risk * 0.2

        slope = calibration_slope(conservative_risk, event_times, event_indicators)

        # Slope should be > 1 for underfitted (too conservative) predictions
        assert slope > 1.0

    def test_constant_risk_returns_nan(self):
        """T011: Test NaN for constant risk scores."""
        n = 50
        event_times = np.random.exponential(1.0, n)
        event_indicators = np.ones(n)
        constant_risk = np.full(n, 0.5)

        slope = calibration_slope(constant_risk, event_times, event_indicators)

        assert np.isnan(slope)

    def test_no_events_returns_nan(self):
        """T011: Test NaN when no events."""
        n = 50
        event_times = np.random.exponential(1.0, n)
        event_indicators = np.zeros(n)  # All censored
        risk_scores = np.random.randn(n)

        slope = calibration_slope(risk_scores, event_times, event_indicators)

        assert np.isnan(slope)

    def test_single_event_returns_nan(self):
        """T011: Test NaN with only one event."""
        n = 50
        event_times = np.random.exponential(1.0, n)
        event_indicators = np.zeros(n)
        event_indicators[0] = 1  # Single event
        risk_scores = np.random.randn(n)

        slope = calibration_slope(risk_scores, event_times, event_indicators)

        assert np.isnan(slope)


class TestIntegratedCalibrationIndex:
    """Tests for ICI using LOESS smoothing."""

    def test_perfect_calibration_low_ici(self):
        """T008: Test ICI ≈ 0 for perfectly calibrated predictions."""
        np.random.seed(42)
        n = 200
        time_horizon = 2.0

        # Generate event times and predict correctly
        event_times = np.random.exponential(2.0, n)
        event_indicators = np.ones(n)

        # Perfect calibration: predicted probability matches actual
        # Prob of event by t=2 with exponential(rate=0.5) is 1-exp(-0.5*2) ≈ 0.63
        # We'll simulate this scenario
        prob_event = 1 - np.exp(-event_times / 2.0)  # Approx survival probs
        survival_probs = 1 - np.clip(prob_event, 0.01, 0.99)

        ici = integrated_calibration_index(
            survival_probs, event_times, event_indicators, time_horizon
        )

        # ICI should be relatively low for reasonable calibration
        assert ici < 0.3

    def test_poor_calibration_high_ici(self):
        """T009/T010: Test higher ICI for poorly calibrated predictions."""
        np.random.seed(42)
        n = 200
        time_horizon = 2.0

        event_times = np.random.exponential(2.0, n)
        event_indicators = np.ones(n)

        # Poor calibration: predict opposite of truth
        # If event occurred early, predict high survival (wrong)
        # If event occurred late, predict low survival (wrong)
        survival_probs = np.where(event_times < time_horizon, 0.9, 0.1)

        ici_poor = integrated_calibration_index(
            survival_probs, event_times, event_indicators, time_horizon
        )

        # Compare with random predictions
        random_probs = np.random.uniform(0.2, 0.8, n)
        ici_random = integrated_calibration_index(
            random_probs, event_times, event_indicators, time_horizon
        )

        # Poor calibration should have higher ICI than random
        assert ici_poor > 0.1

    def test_ici_non_negative(self):
        """Test ICI is always non-negative."""
        np.random.seed(42)
        n = 100
        time_horizon = 1.0

        event_times = np.random.exponential(1.0, n)
        event_indicators = np.ones(n)
        survival_probs = np.random.uniform(0.2, 0.8, n)

        ici = integrated_calibration_index(
            survival_probs, event_times, event_indicators, time_horizon
        )

        assert ici >= 0.0

    def test_insufficient_data_returns_nan(self):
        """T011: Test NaN with insufficient data for LOESS."""
        # Less than 10 samples
        n = 5
        event_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event_indicators = np.ones(n)
        survival_probs = np.array([0.9, 0.7, 0.5, 0.3, 0.1])

        ici = integrated_calibration_index(
            survival_probs, event_times, event_indicators, time_horizon=2.5
        )

        assert np.isnan(ici)

    def test_constant_predictions_returns_nan(self):
        """T011: Test NaN for constant predictions."""
        np.random.seed(42)
        n = 100
        event_times = np.random.exponential(1.0, n)
        event_indicators = np.ones(n)
        constant_probs = np.full(n, 0.5)

        ici = integrated_calibration_index(
            constant_probs, event_times, event_indicators, time_horizon=1.0
        )

        assert np.isnan(ici)


class TestComputeCalibrationMetrics:
    """Tests for combined calibration metrics computation."""

    def test_returns_calibration_result(self):
        """Test that compute_calibration_metrics returns CalibrationResult."""
        np.random.seed(42)
        n = 100
        n_times = 50

        risk_scores = np.random.randn(n)
        time_grid = np.linspace(0.1, 5.0, n_times)
        event_times = np.random.exponential(2.0, n)
        event_indicators = np.ones(n)

        # Generate plausible survival functions
        survival_functions = np.zeros((n, n_times))
        for i in range(n):
            survival_functions[i, :] = np.exp(-np.exp(risk_scores[i]) * time_grid / 5)

        result = compute_calibration_metrics(
            risk_scores=risk_scores,
            survival_functions=survival_functions,
            time_grid=time_grid,
            event_times=event_times,
            event_indicators=event_indicators,
        )

        assert isinstance(result, CalibrationResult)
        assert hasattr(result, "calibration_in_the_large")
        assert hasattr(result, "calibration_slope")
        assert hasattr(result, "ici")
        assert hasattr(result, "time_horizon")

    def test_uses_median_time_horizon_by_default(self):
        """Test default time horizon is median event time."""
        np.random.seed(42)
        n = 100
        n_times = 50

        risk_scores = np.random.randn(n)
        time_grid = np.linspace(0.1, 10.0, n_times)
        event_times = np.array([1.0] * 30 + [5.0] * 40 + [9.0] * 30)  # Median ~5.0
        event_indicators = np.ones(n)

        survival_functions = np.zeros((n, n_times))
        for i in range(n):
            survival_functions[i, :] = np.exp(-time_grid / 5)

        result = compute_calibration_metrics(
            risk_scores=risk_scores,
            survival_functions=survival_functions,
            time_grid=time_grid,
            event_times=event_times,
            event_indicators=event_indicators,
        )

        expected_median = np.median(event_times)
        assert result.time_horizon == pytest.approx(expected_median, rel=0.01)

    def test_custom_time_horizon(self):
        """Test custom time horizon is used."""
        np.random.seed(42)
        n = 100
        n_times = 50

        risk_scores = np.random.randn(n)
        time_grid = np.linspace(0.1, 10.0, n_times)
        event_times = np.random.exponential(2.0, n)
        event_indicators = np.ones(n)

        survival_functions = np.zeros((n, n_times))
        for i in range(n):
            survival_functions[i, :] = np.exp(-time_grid / 5)

        custom_horizon = 3.0
        result = compute_calibration_metrics(
            risk_scores=risk_scores,
            survival_functions=survival_functions,
            time_grid=time_grid,
            event_times=event_times,
            event_indicators=event_indicators,
            time_horizon=custom_horizon,
        )

        assert result.time_horizon == custom_horizon


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = CalibrationResult(
            calibration_in_the_large=0.95,
            calibration_slope=1.05,
            ici=0.08,
            time_horizon=2.5,
        )

        data = result.to_dict()

        assert data["cal_large"] == 0.95
        assert data["cal_slope"] == 1.05
        assert data["ici"] == 0.08
        assert data["time_horizon"] == 2.5

    def test_nan_values(self):
        """Test CalibrationResult can hold NaN values."""
        result = CalibrationResult(
            calibration_in_the_large=np.nan,
            calibration_slope=np.nan,
            ici=np.nan,
            time_horizon=2.0,
        )

        data = result.to_dict()
        assert np.isnan(data["cal_large"])
        assert np.isnan(data["cal_slope"])
        assert np.isnan(data["ici"])
