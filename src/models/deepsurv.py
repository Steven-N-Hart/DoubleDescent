"""DeepSurv neural network model for survival analysis."""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfiguration


class DeepSurv(nn.Module):
    """DeepSurv: Deep Learning for Survival Analysis.

    A multi-layer perceptron that outputs a risk score (log-hazard ratio)
    for each input. Trained using Cox partial likelihood loss.

    Architecture:
        Input -> [Hidden Layer -> Activation -> Dropout] x depth -> Output (1D)

    Args:
        n_features: Number of input features.
        config: Model configuration specifying architecture.
    """

    def __init__(self, n_features: int, config: ModelConfiguration):
        super().__init__()

        self.n_features = n_features
        self.config = config

        # Build network layers
        layers: List[nn.Module] = []

        # Input layer
        in_features = n_features
        for i in range(config.depth):
            layers.append(nn.Linear(in_features, config.width))
            layers.append(self._get_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_features = config.width

        # Output layer (single output for risk score)
        layers.append(nn.Linear(config.width, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "selu":
            return nn.SELU()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, n_features).

        Returns:
            Risk scores of shape (batch_size, 1).
        """
        return self.network(x)

    def predict_risk(
        self,
        x: Union[np.ndarray, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """Predict risk scores.

        Args:
            x: Input features.
            device: Device to use for computation.

        Returns:
            Risk scores as numpy array.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x = x.to(device)

        with torch.no_grad():
            risk = self.forward(x)

        return risk.cpu().numpy().ravel()

    def n_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_weight_norm(self) -> float:
        """Compute L2 norm of all weights."""
        total_norm = 0.0
        for p in self.parameters():
            if p.requires_grad:
                total_norm += p.data.norm(2).item() ** 2
        return total_norm ** 0.5


def cox_ph_loss(
    risk_scores: torch.Tensor,
    event_times: torch.Tensor,
    event_indicators: torch.Tensor,
) -> torch.Tensor:
    """Compute Cox partial likelihood loss.

    This is the negative log partial likelihood, averaged over events.

    Args:
        risk_scores: Predicted log-hazard ratios of shape (batch_size,) or (batch_size, 1).
        event_times: Observed survival times of shape (batch_size,).
        event_indicators: Event indicators of shape (batch_size,).

    Returns:
        Scalar loss tensor.
    """
    risk_scores = risk_scores.view(-1)
    event_times = event_times.view(-1)
    event_indicators = event_indicators.view(-1).float()

    # Number of events in batch
    n_events = event_indicators.sum()

    if n_events == 0:
        # No events in batch - return zero loss
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

    # Sort by descending time for efficient computation
    order = torch.argsort(event_times, descending=True)
    risk_scores = risk_scores[order]
    event_indicators = event_indicators[order]

    # Compute cumulative logsumexp (risk set sums)
    log_cumsum = torch.logcumsumexp(risk_scores, dim=0)

    # Log partial likelihood: sum over events of (risk - log_cumsum)
    event_mask = event_indicators == 1
    log_lik = (risk_scores - log_cumsum)[event_mask].sum()

    # Return negative log likelihood normalized by number of events
    return -log_lik / n_events


def cox_ph_loss_breslow(
    risk_scores: torch.Tensor,
    event_times: torch.Tensor,
    event_indicators: torch.Tensor,
) -> torch.Tensor:
    """Compute Cox partial likelihood loss with Breslow tie handling.

    This version properly handles tied event times using Breslow's
    approximation.

    Args:
        risk_scores: Predicted log-hazard ratios.
        event_times: Observed survival times.
        event_indicators: Event indicators.

    Returns:
        Scalar loss tensor.
    """
    risk_scores = risk_scores.view(-1)
    event_times = event_times.view(-1)
    event_indicators = event_indicators.view(-1).float()

    n_events = event_indicators.sum()
    if n_events == 0:
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

    # Get unique event times
    unique_times = torch.unique(event_times[event_indicators == 1])

    log_lik = torch.tensor(0.0, device=risk_scores.device)

    for t in unique_times:
        # Events at time t
        events_at_t = (event_times == t) & (event_indicators == 1)
        n_events_at_t = events_at_t.sum()

        # Risk set at time t (all with time >= t)
        at_risk = event_times >= t

        # Sum of risk scores for events at t
        risk_sum_events = risk_scores[events_at_t].sum()

        # Log of sum of exp(risk) for risk set
        log_risk_set_sum = torch.logsumexp(risk_scores[at_risk], dim=0)

        # Breslow: subtract n_events_at_t times the log risk set sum
        log_lik = log_lik + risk_sum_events - n_events_at_t * log_risk_set_sum

    return -log_lik / n_events


class DeepSurvEmbedding(nn.Module):
    """DeepSurv with embedding layers for categorical features.

    Instead of one-hot encoding categorical features (which creates p >> n),
    this model learns dense embeddings for each categorical variable.

    Args:
        n_continuous: Number of continuous input features.
        categorical_cardinalities: List of cardinalities for each categorical feature.
        config: Model configuration specifying architecture.
        embedding_dim: Embedding dimension per feature. If "auto", uses sqrt rule.
    """

    def __init__(
        self,
        n_continuous: int,
        categorical_cardinalities: List[int],
        config: ModelConfiguration,
        embedding_dim: Union[str, int] = "auto",
    ):
        super().__init__()

        self.n_continuous = n_continuous
        self.categorical_cardinalities = categorical_cardinalities
        self.config = config

        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleList()
        total_embedding_dim = 0

        for cardinality in categorical_cardinalities:
            if embedding_dim == "auto":
                # Common heuristic: min(50, ceil(cardinality / 2)) or sqrt
                dim = min(50, max(4, int(np.ceil(np.sqrt(cardinality)))))
            else:
                dim = embedding_dim

            self.embeddings.append(nn.Embedding(cardinality, dim))
            total_embedding_dim += dim

        # Total input dimension to MLP
        self.n_features = n_continuous + total_embedding_dim

        # Build network layers (same as DeepSurv)
        layers: List[nn.Module] = []
        in_features = self.n_features

        for i in range(config.depth):
            layers.append(nn.Linear(in_features, config.width))
            layers.append(self._get_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_features = config.width

        # Output layer
        layers.append(nn.Linear(config.width, 1))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "selu":
            return nn.SELU()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(
        self,
        x_continuous: torch.Tensor,
        x_categorical: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_continuous: Continuous features of shape (batch_size, n_continuous).
            x_categorical: Categorical indices of shape (batch_size, n_categorical).
                           Each column contains integer indices for that feature.

        Returns:
            Risk scores of shape (batch_size, 1).
        """
        # Embed categorical features
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            cat_indices = x_categorical[:, i].long()
            embedded.append(emb_layer(cat_indices))

        # Concatenate continuous and embedded features
        if embedded:
            embedded_concat = torch.cat(embedded, dim=1)
            if self.n_continuous > 0:
                x = torch.cat([x_continuous, embedded_concat], dim=1)
            else:
                x = embedded_concat
        else:
            x = x_continuous

        return self.network(x)

    def predict_risk(
        self,
        x_continuous: Union[np.ndarray, torch.Tensor],
        x_categorical: Union[np.ndarray, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """Predict risk scores.

        Args:
            x_continuous: Continuous input features.
            x_categorical: Categorical feature indices.
            device: Device to use for computation.

        Returns:
            Risk scores as numpy array.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        if isinstance(x_continuous, np.ndarray):
            x_continuous = torch.from_numpy(x_continuous).float()
        if isinstance(x_categorical, np.ndarray):
            x_categorical = torch.from_numpy(x_categorical).long()

        x_continuous = x_continuous.to(device)
        x_categorical = x_categorical.to(device)

        with torch.no_grad():
            risk = self.forward(x_continuous, x_categorical)

        return risk.cpu().numpy().ravel()

    def n_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_weight_norm(self) -> float:
        """Compute L2 norm of all weights."""
        total_norm = 0.0
        for p in self.parameters():
            if p.requires_grad:
                total_norm += p.data.norm(2).item() ** 2
        return total_norm ** 0.5


class DeepSurvLoss(nn.Module):
    """Cox partial likelihood loss as a PyTorch module."""

    def __init__(self, use_breslow: bool = False):
        """Initialize loss module.

        Args:
            use_breslow: If True, use Breslow's approximation for ties.
        """
        super().__init__()
        self.use_breslow = use_breslow

    def forward(
        self,
        risk_scores: torch.Tensor,
        event_times: torch.Tensor,
        event_indicators: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            risk_scores: Predicted risk scores.
            event_times: Observed times.
            event_indicators: Event indicators.

        Returns:
            Loss value.
        """
        if self.use_breslow:
            return cox_ph_loss_breslow(risk_scores, event_times, event_indicators)
        return cox_ph_loss(risk_scores, event_times, event_indicators)
