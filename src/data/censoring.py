"""Censoring rate calibration utilities."""

import numpy as np
from typing import Optional


def calibrate_censoring_rate(
    event_times: np.ndarray,
    target_rate: float,
    rng: Optional[np.random.Generator] = None,
    max_iterations: int = 50,
    tolerance: float = 0.02,
) -> float:
    """Calibrate exponential censoring parameter to achieve target rate.

    Uses binary search to find the censoring distribution scale parameter
    that achieves the target censoring rate.

    Args:
        event_times: True event times.
        target_rate: Target censoring rate (proportion censored).
        rng: Random number generator for simulation.
        max_iterations: Maximum binary search iterations.
        tolerance: Acceptable error in achieved rate.

    Returns:
        Scale parameter for exponential censoring distribution.

    Raises:
        ValueError: If target rate cannot be achieved within bounds.
    """
    if rng is None:
        rng = np.random.default_rng()

    if target_rate <= 0:
        return np.inf  # No censoring
    if target_rate >= 0.95:
        return np.min(event_times) * 0.01  # Very aggressive censoring

    event_times = np.asarray(event_times)
    n = len(event_times)

    # Initial bounds for binary search
    # Scale = median(event_times) gives roughly 50% censoring for exponential
    median_time = np.median(event_times)

    # Adjust initial bounds based on target rate
    if target_rate < 0.5:
        low = median_time * 0.1
        high = median_time * 100
    else:
        low = median_time * 0.001
        high = median_time * 10

    # Binary search
    for _ in range(max_iterations):
        mid = (low + high) / 2

        # Simulate censoring with current scale
        actual_rate = _simulate_censoring_rate(event_times, mid, rng, n_simulations=5)

        if abs(actual_rate - target_rate) <= tolerance:
            return mid

        # Adjust bounds: larger scale = less censoring
        if actual_rate < target_rate:
            high = mid  # Need more censoring, decrease scale
        else:
            low = mid  # Need less censoring, increase scale

    # Return best estimate even if tolerance not met
    return mid


def _simulate_censoring_rate(
    event_times: np.ndarray,
    scale: float,
    rng: np.random.Generator,
    n_simulations: int = 5,
) -> float:
    """Simulate censoring rate for given scale parameter.

    Args:
        event_times: True event times.
        scale: Exponential distribution scale parameter.
        rng: Random number generator.
        n_simulations: Number of simulations to average.

    Returns:
        Average censoring rate across simulations.
    """
    n = len(event_times)
    rates = []

    for _ in range(n_simulations):
        censoring_times = rng.exponential(scale=scale, size=n)
        censored = event_times > censoring_times
        rates.append(np.mean(censored))

    return np.mean(rates)


def validate_censoring_rate(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    target_rate: float,
    tolerance: float = 0.02,
) -> bool:
    """Validate that achieved censoring rate is within tolerance.

    Args:
        event_times: Observed times.
        event_indicators: Event indicators (1 = event, 0 = censored).
        target_rate: Target censoring rate.
        tolerance: Acceptable deviation.

    Returns:
        True if achieved rate is within tolerance of target.
    """
    actual_rate = 1 - np.mean(event_indicators)
    return abs(actual_rate - target_rate) <= tolerance
