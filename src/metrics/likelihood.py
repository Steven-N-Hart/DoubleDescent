"""Negative log partial likelihood (Cox loss) calculation."""

import numpy as np
from typing import Optional, Union

import torch
import torch.nn.functional as F


def calculate_negative_log_likelihood(
    risk_scores: Union[np.ndarray, torch.Tensor],
    event_times: Union[np.ndarray, torch.Tensor],
    event_indicators: Union[np.ndarray, torch.Tensor],
) -> float:
    """Calculate the negative log partial likelihood (Cox loss).

    This is the standard loss function for Cox proportional hazards models
    and DeepSurv. It measures how well the model ranks subjects by their
    risk of experiencing the event.

    Args:
        risk_scores: Predicted log-hazard ratios (output of DeepSurv).
            Shape: (n_samples,)
        event_times: Observed survival times.
            Shape: (n_samples,)
        event_indicators: Event indicators (1 = event, 0 = censored).
            Shape: (n_samples,)

    Returns:
        Negative log partial likelihood (loss value).

    Raises:
        ValueError: If no events are present in the data.
    """
    # Convert to numpy for calculation
    if isinstance(risk_scores, torch.Tensor):
        risk_scores = risk_scores.detach().cpu().numpy()
    if isinstance(event_times, torch.Tensor):
        event_times = event_times.detach().cpu().numpy()
    if isinstance(event_indicators, torch.Tensor):
        event_indicators = event_indicators.detach().cpu().numpy()

    risk_scores = np.asarray(risk_scores).ravel()
    event_times = np.asarray(event_times).ravel()
    event_indicators = np.asarray(event_indicators).ravel()

    n_events = np.sum(event_indicators)
    if n_events == 0:
        raise ValueError("No events in data - cannot compute Cox loss")

    # Sort by time (descending for efficient computation)
    order = np.argsort(-event_times)
    risk_scores = risk_scores[order]
    event_times = event_times[order]
    event_indicators = event_indicators[order]

    # Compute log partial likelihood
    # For numerical stability, use log-sum-exp trick
    log_lik = 0.0
    log_cumsum = -np.inf  # log(0)

    for i in range(len(risk_scores)):
        # Update cumulative sum (risk set)
        log_cumsum = np.logaddexp(log_cumsum, risk_scores[i])

        # If this is an event, add to likelihood
        if event_indicators[i] == 1:
            log_lik += risk_scores[i] - log_cumsum

    # Return negative log likelihood (for minimization)
    return -log_lik / n_events


def cox_partial_likelihood_loss(
    risk_scores: torch.Tensor,
    event_times: torch.Tensor,
    event_indicators: torch.Tensor,
) -> torch.Tensor:
    """Calculate Cox partial likelihood loss for PyTorch training.

    This is a differentiable version for use in training loops.

    Args:
        risk_scores: Predicted log-hazard ratios.
            Shape: (batch_size,) or (batch_size, 1)
        event_times: Observed survival times.
            Shape: (batch_size,)
        event_indicators: Event indicators.
            Shape: (batch_size,)

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError: If no events in batch.
    """
    risk_scores = risk_scores.view(-1)
    event_times = event_times.view(-1)
    event_indicators = event_indicators.view(-1).float()

    n_events = event_indicators.sum()
    if n_events == 0:
        # Return zero loss if no events (shouldn't happen in practice)
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

    # Sort by descending time
    order = torch.argsort(event_times, descending=True)
    risk_scores = risk_scores[order]
    event_indicators = event_indicators[order]

    # Compute log partial likelihood using cumulative logsumexp
    # This is more numerically stable than the naive approach
    log_cumsum = torch.logcumsumexp(risk_scores, dim=0)

    # Only count events
    event_mask = event_indicators == 1
    log_lik = (risk_scores - log_cumsum)[event_mask].sum()

    # Return negative log likelihood normalized by number of events
    return -log_lik / n_events


def breslow_estimator(
    risk_scores: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    time_points: Optional[np.ndarray] = None,
) -> tuple:
    """Estimate baseline cumulative hazard using Breslow's method.

    This is needed for computing survival functions from risk scores.

    Args:
        risk_scores: Predicted log-hazard ratios.
            Shape: (n_samples,)
        event_times: Observed survival times.
            Shape: (n_samples,)
        event_indicators: Event indicators.
            Shape: (n_samples,)
        time_points: Times at which to evaluate cumulative hazard.
            If None, uses unique event times.

    Returns:
        Tuple of (times, cumulative_hazard) arrays.
    """
    risk_scores = np.asarray(risk_scores).ravel()
    event_times = np.asarray(event_times).ravel()
    event_indicators = np.asarray(event_indicators).ravel()

    # Get unique event times (sorted)
    event_times_unique = np.unique(event_times[event_indicators == 1])
    event_times_unique = np.sort(event_times_unique)

    if len(event_times_unique) == 0:
        if time_points is not None:
            return time_points, np.zeros(len(time_points))
        return np.array([0.0]), np.array([0.0])

    # Compute exp(risk_scores) once
    exp_risks = np.exp(risk_scores)

    # Compute cumulative hazard at each event time
    cumulative_hazard = np.zeros(len(event_times_unique))
    cumsum = 0.0

    for i, t in enumerate(event_times_unique):
        # Count events at time t
        events_at_t = np.sum((event_times == t) & (event_indicators == 1))

        # Sum of risk scores for those at risk at time t
        at_risk_mask = event_times >= t
        risk_sum = np.sum(exp_risks[at_risk_mask])

        if risk_sum > 0:
            cumsum += events_at_t / risk_sum

        cumulative_hazard[i] = cumsum

    # Interpolate to requested time points if specified
    if time_points is not None:
        result = np.zeros(len(time_points))
        for i, t in enumerate(time_points):
            if t <= event_times_unique[0]:
                result[i] = 0.0
            elif t >= event_times_unique[-1]:
                result[i] = cumulative_hazard[-1]
            else:
                idx = np.searchsorted(event_times_unique, t, side="right") - 1
                result[i] = cumulative_hazard[idx]
        return time_points, result

    return event_times_unique, cumulative_hazard


def compute_survival_function(
    risk_scores: np.ndarray,
    baseline_cumulative_hazard: np.ndarray,
    time_points: np.ndarray,
) -> np.ndarray:
    """Compute survival functions from risk scores and baseline hazard.

    S(t|x) = exp(-H_0(t) * exp(risk_score))

    Args:
        risk_scores: Predicted log-hazard ratios.
            Shape: (n_samples,)
        baseline_cumulative_hazard: Baseline cumulative hazard at time_points.
            Shape: (n_time_points,)
        time_points: Time points.
            Shape: (n_time_points,)

    Returns:
        Survival functions.
            Shape: (n_samples, n_time_points)
    """
    risk_scores = np.asarray(risk_scores).ravel()
    baseline_cumulative_hazard = np.asarray(baseline_cumulative_hazard).ravel()

    # S(t|x) = exp(-H_0(t) * exp(risk_score))
    # Compute for all samples at all time points
    exp_risks = np.exp(risk_scores)  # (n_samples,)

    # Outer product: (n_samples, n_time_points)
    cumulative_hazards = np.outer(exp_risks, baseline_cumulative_hazard)

    # Survival function
    survival_functions = np.exp(-cumulative_hazards)

    return survival_functions
