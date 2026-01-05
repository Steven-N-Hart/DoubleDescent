"""Shared pytest fixtures for Double Descent Experiment Framework tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def set_seeds(random_seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    return random_seed


@pytest.fixture
def sample_covariates():
    """Generate sample covariate data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    return np.random.randn(n_samples, n_features).astype(np.float32)


@pytest.fixture
def sample_survival_data(sample_covariates):
    """Generate sample survival data for testing."""
    np.random.seed(42)
    n_samples = sample_covariates.shape[0]

    # Generate survival times and events
    T = np.random.exponential(scale=1.0, size=n_samples).astype(np.float32)
    E = np.random.binomial(n=1, p=0.7, size=n_samples).astype(np.float32)

    return {
        "X": sample_covariates,
        "T": T,
        "E": E,
    }


@pytest.fixture
def device():
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tmp_experiment_dir(tmp_path):
    """Create a temporary experiment directory structure."""
    exp_dir = tmp_path / "experiment_test"
    exp_dir.mkdir()
    (exp_dir / "data").mkdir()
    (exp_dir / "runs").mkdir()
    (exp_dir / "results").mkdir()
    (exp_dir / "tensorboard").mkdir()
    return exp_dir
