"""Concordance index (C-index) calculation for survival analysis."""

import numpy as np
from typing import Union

try:
    from sksurv.metrics import concordance_index_censored
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False


def calculate_c_index(
    risk_scores: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
) -> float:
    """Calculate the concordance index (C-index).

    The C-index measures the discriminative ability of a survival model.
    It represents the probability that, for a random pair of subjects,
    the subject with the higher risk score will experience the event first.

    Args:
        risk_scores: Predicted risk scores (higher = higher risk).
            Shape: (n_samples,)
        event_times: Observed survival times.
            Shape: (n_samples,)
        event_indicators: Event indicators (1 = event, 0 = censored).
            Shape: (n_samples,)

    Returns:
        C-index value in [0, 1]. Value of 0.5 indicates random predictions.

    Raises:
        ValueError: If input arrays have mismatched shapes.
    """
    risk_scores = np.asarray(risk_scores).ravel()
    event_times = np.asarray(event_times).ravel()
    event_indicators = np.asarray(event_indicators).ravel()

    if not (len(risk_scores) == len(event_times) == len(event_indicators)):
        raise ValueError(
            f"Input arrays must have the same length. Got: "
            f"risk_scores={len(risk_scores)}, event_times={len(event_times)}, "
            f"event_indicators={len(event_indicators)}"
        )

    if len(risk_scores) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Handle all-censored case
    if np.sum(event_indicators) == 0:
        return 0.5  # No comparable pairs

    # Use scikit-survival if available
    if HAS_SKSURV:
        event_indicators_bool = event_indicators.astype(bool)
        c_index, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
            event_indicators_bool,
            event_times,
            risk_scores,
        )
        return c_index

    # Fallback: manual calculation
    return _calculate_c_index_manual(risk_scores, event_times, event_indicators)


def _calculate_c_index_manual(
    risk_scores: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
) -> float:
    """Manual C-index calculation without external dependencies.

    This is a fallback implementation when scikit-survival is not available.

    Args:
        risk_scores: Predicted risk scores.
        event_times: Observed survival times.
        event_indicators: Event indicators.

    Returns:
        C-index value.
    """
    n = len(risk_scores)
    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Only compare if at least one has an event
            if event_indicators[i] == 0 and event_indicators[j] == 0:
                continue

            # Determine which subject has shorter observed time
            if event_times[i] < event_times[j]:
                shorter_idx, longer_idx = i, j
            elif event_times[i] > event_times[j]:
                shorter_idx, longer_idx = j, i
            else:
                # Tied times - only count if both have events
                if event_indicators[i] == 1 and event_indicators[j] == 1:
                    if risk_scores[i] == risk_scores[j]:
                        tied_risk += 1
                continue

            # Check if shorter time subject had an event
            if event_indicators[shorter_idx] == 0:
                continue

            # Compare risk scores
            if risk_scores[shorter_idx] > risk_scores[longer_idx]:
                concordant += 1
            elif risk_scores[shorter_idx] < risk_scores[longer_idx]:
                discordant += 1
            else:
                tied_risk += 1

    total_comparable = concordant + discordant + tied_risk
    if total_comparable == 0:
        return 0.5  # No comparable pairs

    # C-index formula: (concordant + 0.5 * tied_risk) / total
    return (concordant + 0.5 * tied_risk) / total_comparable
