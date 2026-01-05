"""Unit tests for DeepSurv model (T035)."""

import numpy as np
import pytest
import torch

from src.models.config import ModelConfiguration
from src.models.deepsurv import DeepSurv, cox_ph_loss, DeepSurvLoss


class TestModelConfiguration:
    """Tests for ModelConfiguration."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = ModelConfiguration(width=64)
        assert config.width == 64
        assert config.depth == 2
        assert config.activation == "relu"
        assert config.dropout == 0.0
        assert config.weight_decay == 0.0
        assert config.epochs == 50000
        assert config.batch_size == 256
        assert config.learning_rate == 0.001
        assert config.optimizer == "adam"

    def test_validation_invalid_width(self):
        """Test that invalid width raises error."""
        with pytest.raises(ValueError, match="width"):
            ModelConfiguration(width=0)

    def test_validation_invalid_dropout(self):
        """Test that invalid dropout raises error."""
        with pytest.raises(ValueError, match="dropout"):
            ModelConfiguration(width=64, dropout=1.5)

    def test_parameter_count(self):
        """Test parameter count calculation."""
        config = ModelConfiguration(width=64, depth=2)
        n_params = config.n_parameters(n_features=20)

        # Input: 20 * 64 + 64 = 1344
        # Hidden: 1 * (64 * 64 + 64) = 4160
        # Output: 64 + 1 = 65
        # Total: 1344 + 4160 + 65 = 5569
        expected = 20 * 64 + 64 + (64 * 64 + 64) + 64 + 1
        assert n_params == expected

    def test_with_width(self):
        """Test creating config with different width."""
        config = ModelConfiguration(width=64, depth=2)
        new_config = config.with_width(128)

        assert new_config.width == 128
        assert new_config.depth == 2
        assert config.width == 64  # Original unchanged

    def test_with_reduced_lr(self):
        """Test creating config with reduced learning rate."""
        config = ModelConfiguration(width=64, learning_rate=0.001, retry_lr_factor=0.1)
        new_config = config.with_reduced_lr()

        assert new_config.learning_rate == 0.0001
        assert config.learning_rate == 0.001  # Original unchanged

    def test_serialization(self):
        """Test config to/from dict."""
        config = ModelConfiguration(width=128, depth=3, activation="tanh")
        data = config.to_dict()
        restored = ModelConfiguration.from_dict(data)

        assert restored.width == 128
        assert restored.depth == 3
        assert restored.activation == "tanh"


class TestDeepSurv:
    """Tests for DeepSurv model."""

    @pytest.fixture
    def model(self):
        """Create a DeepSurv model for testing."""
        config = ModelConfiguration(width=32, depth=2)
        return DeepSurv(n_features=10, config=config)

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        batch_size = 16
        n_features = 10
        return {
            "X": torch.randn(batch_size, n_features),
            "T": torch.abs(torch.randn(batch_size)) + 0.1,
            "E": torch.bernoulli(torch.full((batch_size,), 0.7)),
        }

    def test_forward_shape(self, model, sample_batch):
        """Test that forward pass produces correct output shape."""
        output = model(sample_batch["X"])
        assert output.shape == (16, 1)

    def test_forward_deterministic(self, model, sample_batch):
        """Test that forward pass is deterministic in eval mode."""
        model.eval()
        output1 = model(sample_batch["X"])
        output2 = model(sample_batch["X"])
        torch.testing.assert_close(output1, output2)

    def test_predict_risk(self, model):
        """Test predict_risk method."""
        X = np.random.randn(20, 10).astype(np.float32)
        risk = model.predict_risk(X)

        assert isinstance(risk, np.ndarray)
        assert risk.shape == (20,)

    def test_n_parameters(self, model):
        """Test parameter count method."""
        n_params = model.n_parameters()

        # Should match expected count
        expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params == expected

    def test_get_weight_norm(self, model):
        """Test weight norm calculation."""
        norm = model.get_weight_norm()
        assert norm > 0

    def test_different_activations(self):
        """Test model with different activation functions."""
        for activation in ["relu", "tanh", "selu"]:
            config = ModelConfiguration(width=16, activation=activation)
            model = DeepSurv(n_features=5, config=config)
            X = torch.randn(4, 5)
            output = model(X)
            assert output.shape == (4, 1)
            assert not torch.isnan(output).any()

    def test_dropout(self):
        """Test that dropout is applied during training."""
        config = ModelConfiguration(width=32, dropout=0.5)
        model = DeepSurv(n_features=10, config=config)

        X = torch.randn(100, 10)
        model.train()

        # With dropout, multiple forward passes should give different results
        outputs = [model(X).sum().item() for _ in range(5)]
        # At least some should be different (high probability with dropout=0.5)
        assert len(set(outputs)) > 1


class TestCoxLoss:
    """Tests for Cox partial likelihood loss."""

    def test_loss_with_events(self):
        """Test loss computation with events."""
        risk_scores = torch.tensor([1.0, 0.5, 0.0, -0.5])
        event_times = torch.tensor([1.0, 2.0, 3.0, 4.0])
        event_indicators = torch.tensor([1.0, 1.0, 1.0, 1.0])

        loss = cox_ph_loss(risk_scores, event_times, event_indicators)

        assert not torch.isnan(loss)
        assert loss > 0

    def test_loss_with_censoring(self):
        """Test loss computation with censored observations."""
        risk_scores = torch.tensor([1.0, 0.5, 0.0, -0.5])
        event_times = torch.tensor([1.0, 2.0, 3.0, 4.0])
        event_indicators = torch.tensor([1.0, 0.0, 1.0, 0.0])  # 50% censoring

        loss = cox_ph_loss(risk_scores, event_times, event_indicators)

        assert not torch.isnan(loss)
        assert loss > 0

    def test_loss_no_events(self):
        """Test loss with no events returns zero."""
        risk_scores = torch.tensor([1.0, 0.5])
        event_times = torch.tensor([1.0, 2.0])
        event_indicators = torch.tensor([0.0, 0.0])

        loss = cox_ph_loss(risk_scores, event_times, event_indicators)

        assert loss.item() == 0.0

    def test_loss_differentiable(self):
        """Test that loss is differentiable."""
        risk_scores = torch.tensor([1.0, 0.5, 0.0], requires_grad=True)
        event_times = torch.tensor([1.0, 2.0, 3.0])
        event_indicators = torch.tensor([1.0, 1.0, 0.0])

        loss = cox_ph_loss(risk_scores, event_times, event_indicators)
        loss.backward()

        assert risk_scores.grad is not None
        assert not torch.isnan(risk_scores.grad).any()

    def test_loss_ordering(self):
        """Test that higher risk scores for earlier events give lower loss."""
        event_times = torch.tensor([1.0, 2.0, 3.0])
        event_indicators = torch.tensor([1.0, 1.0, 1.0])

        # Correct ordering: higher risk for earlier events
        correct_risk = torch.tensor([2.0, 1.0, 0.0])
        # Wrong ordering
        wrong_risk = torch.tensor([0.0, 1.0, 2.0])

        correct_loss = cox_ph_loss(correct_risk, event_times, event_indicators)
        wrong_loss = cox_ph_loss(wrong_risk, event_times, event_indicators)

        assert correct_loss < wrong_loss


class TestDeepSurvLoss:
    """Tests for DeepSurvLoss module."""

    def test_loss_module(self):
        """Test loss as module."""
        loss_fn = DeepSurvLoss()

        risk_scores = torch.tensor([1.0, 0.5, 0.0])
        event_times = torch.tensor([1.0, 2.0, 3.0])
        event_indicators = torch.tensor([1.0, 1.0, 0.0])

        loss = loss_fn(risk_scores, event_times, event_indicators)

        assert not torch.isnan(loss)

    def test_breslow_mode(self):
        """Test Breslow tie handling mode."""
        loss_fn = DeepSurvLoss(use_breslow=True)

        # Create data with tied event times
        risk_scores = torch.tensor([1.0, 0.5, 0.0, -0.5])
        event_times = torch.tensor([1.0, 1.0, 2.0, 2.0])  # Tied times
        event_indicators = torch.tensor([1.0, 1.0, 1.0, 0.0])

        loss = loss_fn(risk_scores, event_times, event_indicators)

        assert not torch.isnan(loss)
        assert loss > 0
