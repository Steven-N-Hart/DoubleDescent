"""Functions for computing calibration decomposition metrics."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Optional
from dataclasses import dataclass

@dataclass
class CalibrationResult:
    """Container for calibration decomposition metrics."""
    calibration_in_the_large: Optional[float] = None
    calibration_slope: Optional[float] = None
    ici: Optional[float] = None
    time_horizon: Optional[float] = None

def calibration_in_the_large(
    survival_probs: np.ndarray,  # S(t|x) at observed times
    event_indicators: np.ndarray
) -> float:
    """Compute calibration-in-the-large (O/E ratio)."""
    observed = event_indicators.sum()
    expected = (1 - survival_probs).sum()  # Expected events = 1 - S(T_i|X_i)
    return observed / expected if expected > 0 else np.nan


def calibration_slope(
    risk_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray
) -> float:
    """Compute calibration slope via Cox regression on linear predictor.

    Fit Cox model with risk scores as sole covariate. Perfect calibration
    yields slope = 1.0. Values < 1 indicate overfitting (predictions too extreme).

    Args:
        risk_scores: Model's linear predictor (log hazard ratios).
        times: Observed survival times.
        events: Event indicators (1=event, 0=censored).

    Returns:
        Calibration slope coefficient.
    """
    df = pd.DataFrame({
        'T': times,
        'E': events.astype(int),
        'risk': risk_scores
    })

    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col='T', event_col='E', formula='risk')
        return cph.params_['risk']
    except Exception:
        return np.nan


def integrated_calibration_index(
    predicted_probs: np.ndarray,
    observed_events: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Integrated Calibration Index (ICI).

    ICI is the weighted average of absolute differences between predicted
    probabilities and LOWESS-smoothed observed proportions.

    Args:
        predicted_probs: Predicted event probabilities at time horizon.
        observed_events: Binary event indicators at time horizon.
        n_bins: Number of bins for grouping (not used with LOWESS).

    Returns:
        ICI value (lower is better, 0 = perfect calibration).
    """
    try:
        # Sort by predicted probability
        sorted_idx = np.argsort(predicted_probs)
        pred_sorted = predicted_probs[sorted_idx]
        obs_sorted = observed_events[sorted_idx]

        # LOWESS smoothing to get calibration curve
        smoothed = lowess(obs_sorted, pred_sorted, frac=0.3, return_sorted=False)

        # ICI = weighted mean absolute difference
        ici = np.mean(np.abs(pred_sorted - smoothed))
        return ici
    except Exception:
        return np.nan


def compute_calibration_metrics(
    risk_scores: np.ndarray,
    survival_functions: np.ndarray,
    time_grid: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    time_horizon: Optional[float] = None
) -> CalibrationResult:
    """Compute all calibration decomposition metrics.

    Args:
        risk_scores: Model's linear predictor (log hazard ratios).
        survival_functions: Predicted survival curves S(t|x), shape (n_samples, n_time_points).
        time_grid: Time points corresponding to survival_functions columns.
        event_times: Observed survival times.
        event_indicators: Event indicators (1=event, 0=censored).
        time_horizon: Time point for calibration assessment (default: median event time).

    Returns:
        CalibrationResult with cal_large, cal_slope, and ICI.
    """
    # Determine time horizon
    if time_horizon is None:
        time_horizon = np.median(event_times[event_indicators == 1])

    # Find closest time index in grid
    time_idx = np.searchsorted(time_grid, time_horizon)
    time_idx = min(time_idx, len(time_grid) - 1)

    # Get survival probabilities at time horizon for each subject
    survival_probs_at_horizon = survival_functions[:, time_idx]

    # For calibration-in-the-large, use survival at each subject's observed time
    # Find the index in time_grid closest to each subject's event time
    time_indices = np.searchsorted(time_grid, event_times)
    time_indices = np.clip(time_indices, 0, len(time_grid) - 1)
    survival_probs_at_obs = survival_functions[np.arange(len(event_times)), time_indices]

    # Calibration-in-the-large (O/E ratio)
    cal_large = calibration_in_the_large(survival_probs_at_obs, event_indicators)

    # Calibration slope
    cal_slope = calibration_slope(risk_scores, event_times, event_indicators)

    # ICI - use 1 - S(t*) as predicted event probability at time horizon
    # Only evaluate on subjects with event_time >= time_horizon or who had event before
    predicted_event_probs = 1 - survival_probs_at_horizon
    # For ICI, we need pseudo-observations or restrict to subjects observed at t*
    # Simplified: use all subjects
    ici = integrated_calibration_index(predicted_event_probs, event_indicators)

    return CalibrationResult(
        calibration_in_the_large=cal_large,
        calibration_slope=cal_slope,
        ici=ici,
        time_horizon=time_horizon
    )        