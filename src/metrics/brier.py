"""Integrated Brier Score (IBS) calculation for survival analysis."""

import numpy as np
from typing import Optional, Tuple, Union

try:
    from sksurv.metrics import integrated_brier_score as sksurv_ibs
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False


def calculate_integrated_brier_score(
    survival_functions: np.ndarray,
    time_points: np.ndarray,
    event_times_train: np.ndarray,
    event_indicators_train: np.ndarray,
    event_times_test: np.ndarray,
    event_indicators_test: np.ndarray,
    times: Optional[np.ndarray] = None,
) -> float:
    """Calculate the Integrated Brier Score (IBS).

    The IBS is a measure of calibration that assesses how well predicted
    survival probabilities match observed outcomes. Lower values are better.

    Args:
        survival_functions: Predicted survival functions.
            Shape: (n_test_samples, n_time_points)
        time_points: Time points at which survival was evaluated.
            Shape: (n_time_points,)
        event_times_train: Training set observed survival times.
            Shape: (n_train_samples,)
        event_indicators_train: Training set event indicators.
            Shape: (n_train_samples,)
        event_times_test: Test set observed survival times.
            Shape: (n_test_samples,)
        event_indicators_test: Test set event indicators.
            Shape: (n_test_samples,)
        times: Time points at which to evaluate IBS.
            If None, uses quantiles of test event times.

    Returns:
        Integrated Brier Score value (lower is better).

    Raises:
        ValueError: If input arrays have invalid shapes.
    """
    survival_functions = np.asarray(survival_functions)
    time_points = np.asarray(time_points)
    event_times_train = np.asarray(event_times_train).ravel()
    event_indicators_train = np.asarray(event_indicators_train).ravel()
    event_times_test = np.asarray(event_times_test).ravel()
    event_indicators_test = np.asarray(event_indicators_test).ravel()

    n_test = len(event_times_test)

    if survival_functions.shape[0] != n_test:
        raise ValueError(
            f"survival_functions rows ({survival_functions.shape[0]}) must match "
            f"test samples ({n_test})"
        )

    if survival_functions.shape[1] != len(time_points):
        raise ValueError(
            f"survival_functions columns ({survival_functions.shape[1]}) must match "
            f"time_points ({len(time_points)})"
        )

    # Use scikit-survival if available
    if HAS_SKSURV:
        return _calculate_ibs_sksurv(
            survival_functions,
            time_points,
            event_times_train,
            event_indicators_train,
            event_times_test,
            event_indicators_test,
            times,
        )

    # Fallback: manual calculation
    return _calculate_ibs_manual(
        survival_functions,
        time_points,
        event_times_train,
        event_indicators_train,
        event_times_test,
        event_indicators_test,
        times,
    )


def _calculate_ibs_sksurv(
    survival_functions: np.ndarray,
    time_points: np.ndarray,
    event_times_train: np.ndarray,
    event_indicators_train: np.ndarray,
    event_times_test: np.ndarray,
    event_indicators_test: np.ndarray,
    times: Optional[np.ndarray],
) -> float:
    """Calculate IBS using scikit-survival."""
    # Create structured arrays for sksurv
    train_struct = np.array(
        [(bool(e), t) for e, t in zip(event_indicators_train, event_times_train)],
        dtype=[("event", bool), ("time", float)],
    )
    test_struct = np.array(
        [(bool(e), t) for e, t in zip(event_indicators_test, event_times_test)],
        dtype=[("event", bool), ("time", float)],
    )

    # Determine evaluation times
    if times is None:
        event_times_with_events = event_times_test[event_indicators_test == 1]
        if len(event_times_with_events) > 0:
            times = np.percentile(event_times_with_events, [10, 25, 50, 75, 90])
        else:
            times = np.percentile(event_times_test, [10, 25, 50, 75, 90])

    # Ensure times are within valid range
    min_time = max(time_points.min(), event_times_test.min())
    max_time = min(time_points.max(), event_times_test.max())
    times = times[(times >= min_time) & (times <= max_time)]

    if len(times) == 0:
        return np.nan

    # Interpolate survival functions at evaluation times
    # sksurv expects shape (n_samples, n_times)
    n_samples = survival_functions.shape[0]
    n_times = len(times)
    estimate = np.zeros((n_samples, n_times))

    for i, t in enumerate(times):
        # Find the index in time_points closest to t
        idx = np.searchsorted(time_points, t)
        idx = min(idx, len(time_points) - 1)
        estimate[:, i] = survival_functions[:, idx]

    try:
        ibs = sksurv_ibs(train_struct, test_struct, estimate, times)
        return ibs
    except Exception:
        return np.nan


def _calculate_ibs_manual(
    survival_functions: np.ndarray,
    time_points: np.ndarray,
    event_times_train: np.ndarray,
    event_indicators_train: np.ndarray,
    event_times_test: np.ndarray,
    event_indicators_test: np.ndarray,
    times: Optional[np.ndarray],
) -> float:
    """Manual IBS calculation without external dependencies.

    Uses inverse probability of censoring weighting (IPCW).
    """
    n_test = len(event_times_test)

    # Determine evaluation times
    if times is None:
        event_times_with_events = event_times_test[event_indicators_test == 1]
        if len(event_times_with_events) > 0:
            times = np.percentile(event_times_with_events, [10, 25, 50, 75, 90])
        else:
            times = np.percentile(event_times_test, [10, 25, 50, 75, 90])

    # Estimate censoring distribution from training data using Kaplan-Meier
    censoring_probs = _estimate_censoring_km(
        event_times_train,
        event_indicators_train,
        times,
    )

    # Calculate Brier scores at each time point
    brier_scores = []

    for t_idx, t in enumerate(times):
        # Get survival predictions at time t
        # Find closest time point in time_points
        time_idx = np.searchsorted(time_points, t)
        time_idx = min(time_idx, len(time_points) - 1)
        S_t = survival_functions[:, time_idx]

        # Calculate Brier score at time t
        bs = 0.0
        n_valid = 0

        for i in range(n_test):
            T_i = event_times_test[i]
            delta_i = event_indicators_test[i]

            # Case 1: Event before time t
            if T_i <= t and delta_i == 1:
                G_Ti = _get_censoring_prob(censoring_probs, times, T_i)
                # Use minimum threshold to prevent division by very small numbers
                if G_Ti > 0.01:
                    bs += (S_t[i] ** 2) / G_Ti
                    n_valid += 1 / G_Ti

            # Case 2: Still at risk at time t
            elif T_i > t:
                G_t = censoring_probs[t_idx]
                # Use minimum threshold to prevent division by very small numbers
                if G_t > 0.01:
                    bs += ((1 - S_t[i]) ** 2) / G_t
                    n_valid += 1 / G_t

        if n_valid > 0:
            score = bs / n_valid
            # Clip to prevent inf values
            if np.isfinite(score):
                brier_scores.append(score)

    if len(brier_scores) == 0:
        return np.nan

    # Integrate over time (trapezoidal rule)
    if len(brier_scores) == 1:
        return brier_scores[0]

    ibs = np.trapz(brier_scores, times[: len(brier_scores)])
    time_range = times[len(brier_scores) - 1] - times[0]

    if time_range > 0:
        ibs /= time_range

    return ibs


def _estimate_censoring_km(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Estimate censoring survival function using Kaplan-Meier.

    For censoring, we treat censoring as the "event" and actual events as "censored".
    """
    # Reverse indicators for censoring distribution
    censoring_indicators = 1 - event_indicators

    # Sort by time
    order = np.argsort(event_times)
    sorted_times = event_times[order]
    sorted_indicators = censoring_indicators[order]

    # Kaplan-Meier estimation
    unique_times = np.unique(sorted_times)
    n = len(event_times)
    survival = 1.0
    km_times = [0.0]
    km_survival = [1.0]

    for t in unique_times:
        at_risk = np.sum(sorted_times >= t)
        events_at_t = np.sum((sorted_times == t) & (sorted_indicators == 1))

        if at_risk > 0:
            survival *= (at_risk - events_at_t) / at_risk

        km_times.append(t)
        km_survival.append(survival)

    km_times = np.array(km_times)
    km_survival = np.array(km_survival)

    # Interpolate at requested times
    result = np.zeros(len(times))
    for i, t in enumerate(times):
        idx = np.searchsorted(km_times, t, side="right") - 1
        idx = max(0, min(idx, len(km_survival) - 1))
        result[i] = km_survival[idx]

    return result


def _get_censoring_prob(
    censoring_probs: np.ndarray,
    times: np.ndarray,
    t: float,
) -> float:
    """Get censoring probability at a specific time by interpolation."""
    if t <= times[0]:
        return 1.0
    if t >= times[-1]:
        return censoring_probs[-1]

    idx = np.searchsorted(times, t, side="right") - 1
    return censoring_probs[idx]


def integrated_brier_score_simple(
    test_times: np.ndarray,
    test_events: np.ndarray,
    risk_scores: np.ndarray,
    train_times: np.ndarray,
    train_events: np.ndarray,
    train_risk_scores: Optional[np.ndarray] = None,
) -> float:
    """Simplified IBS calculation directly from risk scores.

    This function estimates survival functions from risk scores using the
    Breslow estimator, then calculates IBS. Useful when only risk scores
    are available (e.g., from DeepSurvEmbedding).

    Args:
        test_times: Test set observed survival times.
        test_events: Test set event indicators.
        risk_scores: Predicted risk scores for test set.
        train_times: Training set times (for baseline hazard estimation).
        train_events: Training set event indicators.
        train_risk_scores: Risk scores on training data for Breslow estimation.
            If None, uses zeros (less accurate but works).

    Returns:
        Integrated Brier Score value.
    """
    from .likelihood import breslow_estimator, compute_survival_function

    # If no training risk scores provided, use zeros
    if train_risk_scores is None:
        train_risk_scores = np.zeros(len(train_times))

    # Estimate baseline cumulative hazard using Breslow's method
    time_points, baseline_cumhaz = breslow_estimator(
        risk_scores=train_risk_scores,
        event_times=train_times,
        event_indicators=train_events,
    )

    # Compute survival functions for test samples
    survival_funcs = compute_survival_function(
        risk_scores=risk_scores,
        baseline_cumulative_hazard=baseline_cumhaz,
        time_points=time_points,
    )

    # Calculate IBS
    return calculate_integrated_brier_score(
        survival_functions=survival_funcs,
        time_points=time_points,
        event_times_train=train_times,
        event_indicators_train=train_events,
        event_times_test=test_times,
        event_indicators_test=test_events,
    )
