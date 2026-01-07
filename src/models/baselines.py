"""Classical survival analysis baselines for comparison.

Provides wrappers around scikit-survival models for comparison with
neural network approaches.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False

from ..metrics.concordance import calculate_c_index
from ..metrics.brier import calculate_integrated_brier_score
from ..metrics.likelihood import breslow_estimator, compute_survival_function


@dataclass
class BaselineResults:
    """Container for baseline model results.

    Attributes:
        model_name: Name of the baseline model.
        c_index: Concordance index on test set.
        ibs: Integrated Brier Score on test set (if available).
        nll: Negative log-likelihood (if applicable).
        risk_scores: Predicted risk scores.
    """
    model_name: str
    c_index: float
    ibs: Optional[float] = None
    nll: Optional[float] = None
    risk_scores: Optional[np.ndarray] = None


class CoxPHBaseline:
    """Wrapper for scikit-survival's CoxPHSurvivalAnalysis.

    This is a penalized Cox proportional hazards model using
    L2 regularization (Ridge regression).
    """

    def __init__(self, alpha: float = 0.01, max_iter: int = 1000):
        """Initialize Cox PH baseline.

        Args:
            alpha: L2 regularization strength.
            max_iter: Maximum iterations for optimization.
        """
        if not HAS_SKSURV:
            raise ImportError(
                "scikit-survival is required for baselines. "
                "Install with: pip install scikit-survival"
            )

        self.alpha = alpha
        self.max_iter = max_iter
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
    ) -> "CoxPHBaseline":
        """Fit the Cox PH model.

        Args:
            X: Feature matrix (n_samples, n_features).
            T: Event times (n_samples,).
            E: Event indicators (n_samples,).

        Returns:
            Self for chaining.
        """
        # Create structured array for sksurv
        y = np.array(
            [(bool(e), t) for e, t in zip(E, T)],
            dtype=[("event", bool), ("time", float)],
        )

        self.model = CoxPHSurvivalAnalysis(
            alpha=self.alpha,
            n_iter=self.max_iter,
        )
        self.model.fit(X, y)

        # Store training data for survival function computation
        self._train_T = T
        self._train_E = E

        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores (higher = higher risk).

        Args:
            X: Feature matrix.

        Returns:
            Risk scores (n_samples,).
        """
        return self.model.predict(X)

    def predict_survival_function(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict survival functions.

        Args:
            X: Feature matrix.
            times: Time points for evaluation.

        Returns:
            Tuple of (times, survival_functions) where survival_functions
            has shape (n_samples, n_times).
        """
        surv_funcs = self.model.predict_survival_function(X)

        if times is None:
            # Use the times from the first survival function
            times = surv_funcs[0].x

        # Evaluate at specified times
        survival_functions = np.zeros((len(surv_funcs), len(times)))
        for i, sf in enumerate(surv_funcs):
            survival_functions[i, :] = sf(times)

        return times, survival_functions

    def evaluate(
        self,
        X_train: np.ndarray,
        T_train: np.ndarray,
        E_train: np.ndarray,
        X_test: np.ndarray,
        T_test: np.ndarray,
        E_test: np.ndarray,
    ) -> BaselineResults:
        """Evaluate model on test data.

        Args:
            X_train: Training features (for IBS calculation).
            T_train: Training event times.
            E_train: Training event indicators.
            X_test: Test features.
            T_test: Test event times.
            E_test: Test event indicators.

        Returns:
            BaselineResults with evaluation metrics.
        """
        risk_scores = self.predict_risk(X_test)

        # C-index
        c_index = calculate_c_index(risk_scores, T_test, E_test)

        # IBS
        try:
            times, surv_funcs = self.predict_survival_function(X_test)
            ibs = calculate_integrated_brier_score(
                surv_funcs,
                times,
                T_train,
                E_train,
                T_test,
                E_test,
            )
        except Exception:
            ibs = None

        return BaselineResults(
            model_name="CoxPH",
            c_index=c_index,
            ibs=ibs,
            risk_scores=risk_scores,
        )


class RandomSurvivalForestBaseline:
    """Wrapper for scikit-survival's RandomSurvivalForest."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
    ):
        """Initialize Random Survival Forest baseline.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees (None = unlimited).
            min_samples_split: Minimum samples to split a node.
            min_samples_leaf: Minimum samples in a leaf.
            n_jobs: Number of parallel jobs (-1 = all cores).
            random_state: Random seed for reproducibility.
        """
        if not HAS_SKSURV:
            raise ImportError(
                "scikit-survival is required for baselines. "
                "Install with: pip install scikit-survival"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
    ) -> "RandomSurvivalForestBaseline":
        """Fit the Random Survival Forest.

        Args:
            X: Feature matrix (n_samples, n_features).
            T: Event times (n_samples,).
            E: Event indicators (n_samples,).

        Returns:
            Self for chaining.
        """
        # Create structured array for sksurv
        y = np.array(
            [(bool(e), t) for e, t in zip(E, T)],
            dtype=[("event", bool), ("time", float)],
        )

        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.model.fit(X, y)

        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores (higher = higher risk).

        RSF uses the cumulative hazard at a reference time.

        Args:
            X: Feature matrix.

        Returns:
            Risk scores (n_samples,).
        """
        return self.model.predict(X)

    def predict_survival_function(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict survival functions.

        Args:
            X: Feature matrix.
            times: Time points for evaluation.

        Returns:
            Tuple of (times, survival_functions) where survival_functions
            has shape (n_samples, n_times).
        """
        surv_funcs = self.model.predict_survival_function(X)

        if times is None:
            # Use the times from the first survival function
            times = surv_funcs[0].x

        # Evaluate at specified times
        survival_functions = np.zeros((len(surv_funcs), len(times)))
        for i, sf in enumerate(surv_funcs):
            survival_functions[i, :] = sf(times)

        return times, survival_functions

    def evaluate(
        self,
        X_train: np.ndarray,
        T_train: np.ndarray,
        E_train: np.ndarray,
        X_test: np.ndarray,
        T_test: np.ndarray,
        E_test: np.ndarray,
    ) -> BaselineResults:
        """Evaluate model on test data.

        Args:
            X_train: Training features (for IBS calculation).
            T_train: Training event times.
            E_train: Training event indicators.
            X_test: Test features.
            T_test: Test event times.
            E_test: Test event indicators.

        Returns:
            BaselineResults with evaluation metrics.
        """
        risk_scores = self.predict_risk(X_test)

        # C-index
        c_index = calculate_c_index(risk_scores, T_test, E_test)

        # IBS
        try:
            times, surv_funcs = self.predict_survival_function(X_test)
            ibs = calculate_integrated_brier_score(
                surv_funcs,
                times,
                T_train,
                E_train,
                T_test,
                E_test,
            )
        except Exception:
            ibs = None

        return BaselineResults(
            model_name="RSF",
            c_index=c_index,
            ibs=ibs,
            risk_scores=risk_scores,
        )


def run_baselines(
    X_train: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    seed: Optional[int] = None,
) -> Dict[str, BaselineResults]:
    """Run all baseline models and return results.

    Args:
        X_train: Training features.
        T_train: Training event times.
        E_train: Training event indicators.
        X_test: Test features.
        T_test: Test event times.
        E_test: Test event indicators.
        seed: Random seed for RSF.

    Returns:
        Dictionary mapping model names to results.
    """
    results = {}

    # Cox PH
    try:
        cox = CoxPHBaseline(alpha=0.01)
        cox.fit(X_train, T_train, E_train)
        results["CoxPH"] = cox.evaluate(
            X_train, T_train, E_train,
            X_test, T_test, E_test,
        )
    except Exception as e:
        print(f"CoxPH failed: {e}")

    # Random Survival Forest
    try:
        rsf = RandomSurvivalForestBaseline(
            n_estimators=100,
            random_state=seed,
        )
        rsf.fit(X_train, T_train, E_train)
        results["RSF"] = rsf.evaluate(
            X_train, T_train, E_train,
            X_test, T_test, E_test,
        )
    except Exception as e:
        print(f"RSF failed: {e}")

    return results
