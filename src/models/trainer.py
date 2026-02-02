"""Training loop for DeepSurv models."""

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ModelConfiguration
from .deepsurv import DeepSurv, cox_ph_loss
from ..data.generator import SurvivalData
from ..metrics.results import MetricResult
from ..metrics.evaluator import MetricEvaluator, EvaluationData


@dataclass
class TrainingState:
    """Current state of training."""

    epoch: int = 0
    best_val_metric: float = float("inf")
    best_epoch: int = 0
    failed: bool = False
    failure_reason: Optional[str] = None
    retry_count: int = 0


class Trainer:
    """Trainer for DeepSurv models.

    Handles the training loop with:
    - Epoch-level metric logging
    - Gradient and weight norm tracking
    - Failure detection and retry logic
    - GPU OOM handling

    Args:
        model: DeepSurv model to train.
        config: Model configuration.
        train_data: Training data.
        val_data: Validation data.
        test_data: Test data.
        device: Device to train on.
        log_callback: Optional callback for logging metrics.
    """

    def __init__(
        self,
        model: DeepSurv,
        config: ModelConfiguration,
        train_data: SurvivalData,
        val_data: SurvivalData,
        test_data: SurvivalData,
        device: torch.device,
        log_callback: Optional[Callable[[MetricResult], None]] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.log_callback = log_callback

        # Prepare data
        self.train_loader = self._create_dataloader(train_data, shuffle=True)
        self.val_loader = self._create_dataloader(val_data, shuffle=False)
        self.test_loader = self._create_dataloader(test_data, shuffle=False)

        # Store raw data for metric evaluation
        self.train_eval_data = EvaluationData(
            X=train_data.X, T=train_data.T, E=train_data.E
        )
        self.val_eval_data = EvaluationData(
            X=val_data.X, T=val_data.T, E=val_data.E
        )
        self.test_eval_data = EvaluationData(
            X=test_data.X, T=test_data.T, E=test_data.E
        )

        # Metric evaluator
        self.evaluator = MetricEvaluator(self.train_eval_data)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Training state
        self.state = TrainingState()

    def _create_dataloader(
        self, data: SurvivalData, shuffle: bool
    ) -> DataLoader:
        """Create a DataLoader from survival data."""
        X = torch.from_numpy(data.X).float()
        T = torch.from_numpy(data.T).float()
        E = torch.from_numpy(data.E).float()

        dataset = TensorDataset(X, T, E)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def train(self, run_id: str) -> Tuple[TrainingState, List[MetricResult]]:
        """Run full training loop.

        Args:
            run_id: Run identifier for logging.

        Returns:
            Tuple of (final training state, list of all metric results).
        """
        all_metrics: List[MetricResult] = []

        try:
            for epoch in range(self.config.epochs):
                self.state.epoch = epoch

                # Training step
                train_loss, grad_norm, batch_loss_var = self._train_epoch()

                # Check for training failure
                if self._check_failure(train_loss, grad_norm):
                    break

                # Evaluate metrics
                metrics = self._evaluate(run_id, epoch, grad_norm, batch_loss_var)
                all_metrics.extend(metrics.values())

                # Log metrics
                if self.log_callback:
                    for m in metrics.values():
                        self.log_callback(m)

                # Track best validation metric (using IBS - lower is better)
                val_ibs = metrics["val"].integrated_brier_score
                if math.isfinite(val_ibs) and val_ibs < self.state.best_val_metric:
                    self.state.best_val_metric = val_ibs
                    self.state.best_epoch = epoch

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.state.failed = True
                self.state.failure_reason = "GPU out of memory"
                torch.cuda.empty_cache()
            else:
                self.state.failed = True
                self.state.failure_reason = str(e)

        return self.state, all_metrics

    def _train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch.

        Returns:
            Tuple of (average loss, gradient norm, batch loss variance).
        """
        self.model.train()
        total_loss = 0.0
        batch_losses = []
        total_grad_norm = 0.0
        n_batches = 0

        for X_batch, T_batch, E_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            T_batch = T_batch.to(self.device)
            E_batch = E_batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            risk_scores = self.model(X_batch)

            # Compute loss
            loss = cox_ph_loss(risk_scores, T_batch, E_batch)

            # Backward pass
            loss.backward()

            # Compute gradient norm before clipping
            grad_norm = self._compute_gradient_norm()
            total_grad_norm += grad_norm

            # Optimizer step
            self.optimizer.step()

            # Track metrics
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_grad_norm = total_grad_norm / max(n_batches, 1)
        batch_loss_var = np.var(batch_losses) if batch_losses else 0.0

        return avg_loss, avg_grad_norm, batch_loss_var

    def _compute_gradient_norm(self) -> float:
        """Compute total gradient L2 norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _check_failure(self, loss: float, grad_norm: float) -> bool:
        """Check for training failure conditions.

        Args:
            loss: Current loss value.
            grad_norm: Current gradient norm.

        Returns:
            True if training should stop due to failure.
        """
        # Check for NaN loss
        if math.isnan(loss):
            self.state.failed = True
            self.state.failure_reason = "NaN loss detected"
            return True

        # Check for infinite loss
        if math.isinf(loss):
            self.state.failed = True
            self.state.failure_reason = "Infinite loss detected"
            return True

        # Check for gradient explosion
        if math.isinf(grad_norm) or grad_norm > 1e6:
            self.state.failed = True
            self.state.failure_reason = "Gradient explosion detected"
            return True

        return False

    def _evaluate(
        self,
        run_id: str,
        epoch: int,
        grad_norm: float,
        batch_loss_var: float,
    ) -> Dict[str, MetricResult]:
        """Evaluate model on all splits.

        Args:
            run_id: Run identifier.
            epoch: Current epoch.
            grad_norm: Gradient norm from training.
            batch_loss_var: Batch loss variance.

        Returns:
            Dictionary of metrics by split.
        """
        self.model.eval()

        # Get risk scores for all splits
        with torch.no_grad():
            train_risk = self.model.predict_risk(
                self.train_eval_data.X, self.device
            )
            val_risk = self.model.predict_risk(
                self.val_eval_data.X, self.device
            )
            test_risk = self.model.predict_risk(
                self.test_eval_data.X, self.device
            )

        # Compute metrics
        weight_norm = self.model.get_weight_norm()
        current_lr = self.optimizer.param_groups[0]["lr"]

        # Train metrics
        train_metrics = self.evaluator.evaluate(
            train_risk, self.train_eval_data, "train", run_id, epoch
        )
        train_metrics.gradient_norm = grad_norm
        train_metrics.weight_norm = weight_norm
        train_metrics.learning_rate = current_lr
        train_metrics.batch_loss_variance = batch_loss_var

        # Val metrics
        val_metrics = self.evaluator.evaluate(
            val_risk, self.val_eval_data, "val", run_id, epoch
        )

        # Test metrics
        test_metrics = self.evaluator.evaluate(
            test_risk, self.test_eval_data, "test", run_id, epoch
        )

        return {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }


def train_with_retry(
    n_features: int,
    config: ModelConfiguration,
    train_data: SurvivalData,
    val_data: SurvivalData,
    test_data: SurvivalData,
    device: torch.device,
    run_id: str,
    log_callback: Optional[Callable[[MetricResult], None]] = None,
    max_retries: int = 1,
) -> Tuple[TrainingState, List[MetricResult], Optional[DeepSurv]]:
    """Train a model with retry on failure.

    Args:
        n_features: Number of input features.
        config: Model configuration.
        train_data: Training data.
        val_data: Validation data.
        test_data: Test data.
        device: Device to train on.
        run_id: Run identifier.
        log_callback: Optional logging callback.
        max_retries: Maximum number of retry attempts.

    Returns:
        Tuple of (final state, all metrics, trained model or None).
    """
    current_config = config
    all_metrics: List[MetricResult] = []

    for attempt in range(max_retries + 1):
        # Create fresh model
        model = DeepSurv(n_features, current_config)

        # Create trainer
        trainer = Trainer(
            model=model,
            config=current_config,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
            log_callback=log_callback,
        )

        # Train
        state, metrics = trainer.train(run_id)
        all_metrics.extend(metrics)

        # Check if successful
        if not state.failed:
            return state, all_metrics, model

        # If failed and more retries available, reduce learning rate
        if attempt < max_retries:
            current_config = current_config.with_reduced_lr()
            state.retry_count = attempt + 1

    # All retries failed
    return state, all_metrics, None
