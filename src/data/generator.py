"""Synthetic survival data generator."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import stats

from .copula import generate_correlated_uniform
from .scenarios import DataScenario
from .types import CovariateType


@dataclass
class SurvivalData:
    """Container for generated survival data.

    Attributes:
        X: Covariate matrix of shape (n_samples, n_features).
        T: Observed survival times of shape (n_samples,).
        E: Event indicators of shape (n_samples,).
        T_true: True event times before censoring of shape (n_samples,).
        beta: Ground truth coefficients of shape (n_features,).
    """

    X: np.ndarray
    T: np.ndarray
    E: np.ndarray
    T_true: np.ndarray
    beta: np.ndarray

    def save(self, path: Union[str, Path]) -> None:
        """Save data to npz file.

        Args:
            path: Path to save file.
        """
        np.savez(
            path,
            X=self.X,
            T=self.T,
            E=self.E,
            T_true=self.T_true,
            beta=self.beta,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SurvivalData":
        """Load data from npz file.

        Args:
            path: Path to npz file.

        Returns:
            SurvivalData instance.
        """
        data = np.load(path)
        return cls(
            X=data["X"],
            T=data["T"],
            E=data["E"],
            T_true=data["T_true"],
            beta=data["beta"],
        )


class SurvivalDataGenerator:
    """Generator for synthetic survival data.

    Generates survival data using inverse transform sampling with
    a Weibull baseline hazard. Supports multiple covariate distributions
    and censoring mechanisms.

    Args:
        scenario: Data scenario configuration.
        seed: Random seed for reproducibility.
    """

    def __init__(self, scenario: DataScenario, seed: int = 42):
        self.scenario = scenario
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self) -> SurvivalData:
        """Generate synthetic survival data.

        Returns:
            SurvivalData with covariates, times, events, and ground truth.
        """
        # Generate covariates
        X = self._generate_covariates()

        # Generate ground truth coefficients
        beta = self._generate_coefficients()

        # Compute predictor (linear or nonlinear)
        if self.scenario.nonlinear:
            predictor = self._compute_nonlinear_predictor(X, beta)
        else:
            predictor = X @ beta

        # Generate true event times using inverse transform sampling
        T_true = self._generate_event_times(predictor)

        # Generate censoring times and apply censoring
        T, E = self._apply_censoring(T_true)

        return SurvivalData(X=X, T=T, E=E, T_true=T_true, beta=beta)

    def _generate_covariates(self) -> np.ndarray:
        """Generate covariates based on scenario configuration."""
        n = self.scenario.n_samples
        p = self.scenario.n_features

        if self.scenario.covariate_type == CovariateType.GAUSSIAN:
            return self._generate_gaussian_covariates(n, p)
        elif self.scenario.covariate_type == CovariateType.LOGNORMAL:
            return self._generate_lognormal_covariates(n, p)
        elif self.scenario.covariate_type == CovariateType.CATEGORICAL:
            return self._generate_categorical_covariates(n, p)
        elif self.scenario.covariate_type == CovariateType.MIXED:
            return self._generate_mixed_covariates(n, p)
        else:
            raise ValueError(f"Unknown covariate type: {self.scenario.covariate_type}")

    def _generate_gaussian_covariates(
        self, n_samples: int, n_features: int
    ) -> np.ndarray:
        """Generate standard Gaussian covariates."""
        if self.scenario.correlation_matrix is not None:
            # Use Gaussian copula for correlated covariates
            uniform = generate_correlated_uniform(
                n_samples,
                n_features,
                self.scenario.correlation_matrix,
                self.rng,
            )
            return stats.norm.ppf(uniform)
        else:
            return self.rng.standard_normal(size=(n_samples, n_features))

    def _generate_lognormal_covariates(
        self, n_samples: int, n_features: int
    ) -> np.ndarray:
        """Generate log-normal covariates (biomarker-like)."""
        if self.scenario.correlation_matrix is not None:
            uniform = generate_correlated_uniform(
                n_samples,
                n_features,
                self.scenario.correlation_matrix,
                self.rng,
            )
            # Transform to log-normal: mean=0, sigma=1 in log space
            return stats.lognorm.ppf(uniform, s=1.0)
        else:
            return self.rng.lognormal(mean=0, sigma=1, size=(n_samples, n_features))

    def _generate_categorical_covariates(
        self, n_samples: int, n_features: int
    ) -> np.ndarray:
        """Generate one-hot encoded categorical covariates."""
        n_cat = self.scenario.n_categorical_features
        cardinality = self.scenario.cardinality

        if n_cat == 0:
            n_cat = n_features  # All features are categorical

        # Generate categorical indices
        cat_indices = self.rng.integers(0, cardinality, size=(n_samples, n_cat))

        # One-hot encode
        # Total columns = n_cat * cardinality (if we fully encode)
        # But we'll truncate/pad to n_features for consistency
        one_hot = np.zeros((n_samples, n_cat * cardinality))
        for i in range(n_cat):
            one_hot[np.arange(n_samples), i * cardinality + cat_indices[:, i]] = 1.0

        # Truncate or pad to n_features
        if one_hot.shape[1] >= n_features:
            return one_hot[:, :n_features]
        else:
            # Pad with zeros
            padding = np.zeros((n_samples, n_features - one_hot.shape[1]))
            return np.hstack([one_hot, padding])

    def _generate_mixed_covariates(
        self, n_samples: int, n_features: int
    ) -> np.ndarray:
        """Generate mixed continuous and categorical covariates."""
        n_cat = self.scenario.n_categorical_features
        n_cont = n_features - n_cat

        if n_cont <= 0:
            return self._generate_categorical_covariates(n_samples, n_features)

        # Generate continuous covariates (Gaussian)
        continuous = self.rng.standard_normal(size=(n_samples, n_cont))

        # Generate categorical covariates (simple integer encoding for mixed)
        categorical = self.rng.integers(
            0, self.scenario.cardinality, size=(n_samples, n_cat)
        ).astype(float)
        # Normalize categorical to have similar scale as continuous
        categorical = (categorical - categorical.mean()) / (categorical.std() + 1e-8)

        return np.hstack([continuous, categorical])

    def _generate_coefficients(self) -> np.ndarray:
        """Generate sparse ground truth coefficients.

        Only the first n_predictive features have non-zero coefficients.
        """
        beta = np.zeros(self.scenario.n_features)

        # Generate non-zero coefficients for predictive features
        low, high = self.scenario.coefficient_range
        beta[: self.scenario.n_predictive] = self.rng.uniform(
            low, high, size=self.scenario.n_predictive
        )

        return beta

    def _compute_nonlinear_predictor(
        self, X: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Compute nonlinear predictor with interactions and quadratic terms.

        The nonlinear predictor includes:
        - Linear terms: X @ beta (same as linear case)
        - Interaction terms: x_i * x_j for pairs of predictive features
        - Quadratic terms: x_i^2 for predictive features

        This creates a ground truth that cannot be learned by a linear model,
        justifying the use of neural networks.

        Args:
            X: Covariate matrix of shape (n_samples, n_features).
            beta: Linear coefficients of shape (n_features,).

        Returns:
            Nonlinear predictor of shape (n_samples,).
        """
        n_pred = self.scenario.n_predictive
        interaction_strength = self.scenario.interaction_strength
        quadratic_strength = self.scenario.quadratic_strength

        # Start with linear component
        predictor = X @ beta

        # Add interaction terms: x_0*x_1, x_1*x_2, x_2*x_3, ...
        # Use consecutive pairs of predictive features
        for i in range(min(n_pred - 1, 5)):  # Limit to 5 interaction terms
            interaction = X[:, i] * X[:, i + 1]
            # Scale by product of corresponding betas to maintain effect scale
            scale = abs(beta[i] * beta[i + 1]) ** 0.5
            predictor += interaction_strength * scale * interaction

        # Add quadratic terms: x_0^2, x_1^2, ...
        for i in range(min(n_pred, 5)):  # Limit to 5 quadratic terms
            quadratic = X[:, i] ** 2 - 1  # Center around 0 (E[X^2] = 1 for standard normal)
            scale = abs(beta[i])
            predictor += quadratic_strength * scale * quadratic

        return predictor

    def _generate_event_times(self, linear_predictor: np.ndarray) -> np.ndarray:
        """Generate event times using inverse transform sampling.

        Uses Weibull baseline hazard:
        h_0(t) = lambda * nu * (lambda * t)^(nu - 1)
        S_0(t) = exp(-(lambda * t)^nu)

        With proportional hazards:
        S(t|x) = S_0(t)^exp(x'beta) = exp(-(lambda * t)^nu * exp(x'beta))

        Args:
            linear_predictor: x'beta for each sample.

        Returns:
            Event times.
        """
        n = len(linear_predictor)
        lam = self.scenario.weibull_scale
        nu = self.scenario.weibull_shape

        # Generate uniform random variables
        U = self.rng.uniform(0, 1, size=n)

        # Inverse transform: T = (-log(U) / (lambda^nu * exp(linear_predictor)))^(1/nu)
        # Simplified: T = (1/lambda) * (-log(U) / exp(linear_predictor))^(1/nu)
        exp_lp = np.exp(linear_predictor)

        # Avoid numerical issues
        exp_lp = np.clip(exp_lp, 1e-10, 1e10)

        T = (1.0 / lam) * ((-np.log(U + 1e-10)) / exp_lp) ** (1.0 / nu)

        # Ensure positive times
        T = np.maximum(T, 1e-10)

        return T

    def _apply_censoring(
        self, T_true: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random censoring to achieve target censoring rate.

        Uses exponential censoring distribution with rate parameter
        calibrated to achieve the target censoring rate.

        Args:
            T_true: True event times.

        Returns:
            Tuple of (observed times, event indicators).
        """
        from .censoring import calibrate_censoring_rate

        target_rate = self.scenario.censoring_rate
        n = len(T_true)

        if target_rate <= 0:
            # No censoring
            return T_true.copy(), np.ones(n, dtype=np.float32)

        # Calibrate censoring distribution parameter
        censoring_scale = calibrate_censoring_rate(T_true, target_rate, self.rng)

        # Generate censoring times
        C = self.rng.exponential(scale=censoring_scale, size=n)

        # Apply censoring
        T = np.minimum(T_true, C)
        E = (T_true <= C).astype(np.float32)

        return T, E


class DataSplitter:
    """Utility for splitting survival data into train/val/test sets."""

    def __init__(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
    ):
        """Initialize splitter.

        Args:
            train_ratio: Fraction for training set.
            val_ratio: Fraction for validation set.
            test_ratio: Fraction for test set.
            seed: Random seed.
        """
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.rng = np.random.default_rng(seed)

    def split(
        self, data: SurvivalData
    ) -> Tuple[SurvivalData, SurvivalData, SurvivalData]:
        """Split data into train/val/test sets.

        Args:
            data: Full dataset.

        Returns:
            Tuple of (train, val, test) SurvivalData.
        """
        n = len(data.T)
        indices = self.rng.permutation(n)

        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        train_data = SurvivalData(
            X=data.X[train_idx],
            T=data.T[train_idx],
            E=data.E[train_idx],
            T_true=data.T_true[train_idx],
            beta=data.beta,
        )

        val_data = SurvivalData(
            X=data.X[val_idx],
            T=data.T[val_idx],
            E=data.E[val_idx],
            T_true=data.T_true[val_idx],
            beta=data.beta,
        )

        test_data = SurvivalData(
            X=data.X[test_idx],
            T=data.T[test_idx],
            E=data.E[test_idx],
            T_true=data.T_true[test_idx],
            beta=data.beta,
        )

        return train_data, val_data, test_data
