"""Model checkpointing utilities."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .config import ModelConfiguration
from .deepsurv import DeepSurv


def save_checkpoint(
    model: DeepSurv,
    optimizer: torch.optim.Optimizer,
    config: ModelConfiguration,
    epoch: int,
    best_val_metric: float,
    checkpoint_path: Union[str, Path],
    additional_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint.

    Args:
        model: Trained model.
        optimizer: Optimizer state.
        config: Model configuration.
        epoch: Current epoch number.
        best_val_metric: Best validation metric so far.
        checkpoint_path: Path to save checkpoint.
        additional_info: Additional metadata to save.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
        "n_features": model.n_features,
        "epoch": epoch,
        "best_val_metric": best_val_metric,
    }

    if additional_info:
        checkpoint["additional_info"] = additional_info

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: torch.device,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load tensors to.

    Returns:
        Dictionary containing checkpoint data.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def restore_model(
    checkpoint_path: Union[str, Path],
    device: torch.device,
) -> tuple:
    """Restore a model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model to.

    Returns:
        Tuple of (model, config, epoch, best_val_metric).
    """
    checkpoint = load_checkpoint(checkpoint_path, device)

    # Recreate config
    config = ModelConfiguration.from_dict(checkpoint["config"])

    # Recreate model
    n_features = checkpoint["n_features"]
    model = DeepSurv(n_features, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    return (
        model,
        config,
        checkpoint["epoch"],
        checkpoint["best_val_metric"],
    )


class CheckpointManager:
    """Manager for model checkpoints during training.

    Handles saving and loading checkpoints, tracking best model,
    and cleaning up old checkpoints.

    Args:
        checkpoint_dir: Directory to store checkpoints.
        keep_last_n: Number of recent checkpoints to keep.
        save_best: Whether to save best model separately.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        keep_last_n: int = 3,
        save_best: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_best = save_best

        self.best_metric = float("inf")
        self.best_epoch = 0
        self.saved_checkpoints: list = []

    def save(
        self,
        model: DeepSurv,
        optimizer: torch.optim.Optimizer,
        config: ModelConfiguration,
        epoch: int,
        val_metric: float,
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint.

        Args:
            model: Model to save.
            optimizer: Optimizer to save.
            config: Model configuration.
            epoch: Current epoch.
            val_metric: Current validation metric.
            is_best: Whether this is the best model so far.

        Returns:
            Path to saved checkpoint.
        """
        # Save regular checkpoint
        checkpoint_name = f"epoch_{epoch:06d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            best_val_metric=val_metric,
            checkpoint_path=checkpoint_path,
        )

        self.saved_checkpoints.append(checkpoint_path)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        # Save best model
        if is_best and self.save_best:
            self.best_metric = val_metric
            self.best_epoch = epoch

            best_path = self.checkpoint_dir / "best.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                best_val_metric=val_metric,
                checkpoint_path=best_path,
            )

        return checkpoint_path

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        while len(self.saved_checkpoints) > self.keep_last_n:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

    def load_best(self, device: torch.device) -> Optional[tuple]:
        """Load the best checkpoint.

        Args:
            device: Device to load model to.

        Returns:
            Tuple of (model, config, epoch, best_metric) or None.
        """
        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.exists():
            return None

        return restore_model(best_path, device)

    def load_latest(self, device: torch.device) -> Optional[tuple]:
        """Load the most recent checkpoint.

        Args:
            device: Device to load model to.

        Returns:
            Tuple of (model, config, epoch, best_metric) or None.
        """
        checkpoints = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        if not checkpoints:
            return None

        return restore_model(checkpoints[-1], device)

    def get_resume_epoch(self) -> int:
        """Get the epoch to resume from.

        Returns:
            Last completed epoch, or 0 if no checkpoints exist.
        """
        checkpoints = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        if not checkpoints:
            return 0

        # Parse epoch from filename
        last_checkpoint = checkpoints[-1].stem
        epoch = int(last_checkpoint.split("_")[1])
        return epoch
