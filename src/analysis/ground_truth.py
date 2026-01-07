"""Ground truth verification utilities.

Provides functions to compare model predictions against known
ground truth from synthetic data generation.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.stats import spearmanr, pearsonr

from ..metrics.concordance import calculate_c_index


def load_ground_truth(experiment_dir: Union[str, Path]) -> Dict:
    """Load ground truth data from an experiment directory.

    Args:
        experiment_dir: Path to experiment output directory.

    Returns:
        Dictionary with:
            - beta: True regression coefficients
            - train/val/test: Dicts with X, T, E, T_true
    """
    experiment_dir = Path(experiment_dir)
    data_dir = experiment_dir / "data"

    # Load ground truth metadata
    with open(data_dir / "ground_truth.json") as f:
        metadata = json.load(f)

    result = {
        "beta": np.array(metadata["beta"]),
        "n_features": metadata.get("n_features", len(metadata["beta"])),
        "n_predictive": metadata.get("n_predictive", 10),
        "censoring_rate": metadata.get("censoring_rate", 0.3),
    }

    # Load data splits
    for split in ["train", "val", "test"]:
        split_path = data_dir / f"{split}.npz"
        if split_path.exists():
            data = np.load(split_path)
            result[split] = {
                "X": data["X"],
                "T": data["T"],
                "E": data["E"],
            }
            # T_true may not be in older experiments
            if "T_true" in data:
                result[split]["T_true"] = data["T_true"]

    return result


def compute_true_risk(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute true linear predictor (risk score).

    Args:
        X: Covariate matrix (n_samples, n_features).
        beta: True regression coefficients.

    Returns:
        True risk scores (n_samples,).
    """
    return X @ beta


def compute_risk_correlation(
    predicted_risk: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
    method: str = "spearman",
) -> Tuple[float, float]:
    """Compute correlation between predicted and true risk scores.

    Since Cox models only need to get rankings correct, Spearman
    correlation is more appropriate than Pearson.

    Args:
        predicted_risk: Model's predicted risk scores.
        X: Covariate matrix.
        beta: True regression coefficients.
        method: "spearman" or "pearson".

    Returns:
        Tuple of (correlation, p-value).
    """
    true_risk = compute_true_risk(X, beta)

    if method == "spearman":
        corr, pval = spearmanr(predicted_risk, true_risk)
    elif method == "pearson":
        corr, pval = pearsonr(predicted_risk, true_risk)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(corr), float(pval)


def compute_bayes_optimal_concordance(
    X: np.ndarray,
    beta: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    use_true_times: bool = False,
    T_true: Optional[np.ndarray] = None,
) -> float:
    """Compute C-index achievable with perfect knowledge of beta.

    This represents the theoretical upper bound for any model
    on this data, given the censoring pattern.

    Args:
        X: Covariate matrix.
        beta: True regression coefficients.
        T: Observed survival times.
        E: Event indicators.
        use_true_times: If True and T_true provided, use uncensored times.
        T_true: True (uncensored) event times.

    Returns:
        Bayes-optimal concordance index.
    """
    true_risk = compute_true_risk(X, beta)

    if use_true_times and T_true is not None:
        # Perfect scenario: no censoring, true times
        c_index = calculate_c_index(
            true_risk,
            T_true,
            np.ones(len(T_true)),  # All events observed
        )
    else:
        # Realistic scenario: with observed censoring
        c_index = calculate_c_index(true_risk, T, E)

    return c_index


def analyze_coefficient_recovery(
    model_coefficients: np.ndarray,
    true_beta: np.ndarray,
    n_predictive: int,
) -> Dict[str, float]:
    """Analyze how well a linear model recovers true coefficients.

    Only applicable to linear models (Cox PH) or when extracting
    feature importance from neural networks.

    Args:
        model_coefficients: Estimated coefficients from model.
        true_beta: True regression coefficients.
        n_predictive: Number of truly predictive features.

    Returns:
        Dictionary with recovery metrics.
    """
    # Correlation between estimated and true coefficients
    corr, _ = pearsonr(model_coefficients, true_beta)

    # Mean squared error
    mse = np.mean((model_coefficients - true_beta) ** 2)

    # Check if model correctly identifies predictive features
    # (features 0 to n_predictive-1 should have non-zero coefficients)
    pred_mask = np.zeros(len(true_beta), dtype=bool)
    pred_mask[:n_predictive] = True

    # Top-k feature selection accuracy
    top_k_estimated = np.argsort(np.abs(model_coefficients))[-n_predictive:]
    top_k_true = np.arange(n_predictive)
    selection_accuracy = len(set(top_k_estimated) & set(top_k_true)) / n_predictive

    return {
        "coefficient_correlation": float(corr),
        "coefficient_mse": float(mse),
        "feature_selection_accuracy": float(selection_accuracy),
    }


def summarize_ground_truth_verification(
    experiment_dir: Union[str, Path],
    predicted_risk_test: np.ndarray,
) -> Dict[str, float]:
    """Generate summary of ground truth verification.

    Args:
        experiment_dir: Path to experiment directory.
        predicted_risk_test: Model's predicted risk scores on test set.

    Returns:
        Dictionary with verification metrics.
    """
    gt = load_ground_truth(experiment_dir)

    X_test = gt["test"]["X"]
    T_test = gt["test"]["T"]
    E_test = gt["test"]["E"]
    beta = gt["beta"]

    # Risk correlation
    spearman_corr, spearman_p = compute_risk_correlation(
        predicted_risk_test, X_test, beta, method="spearman"
    )

    # Model's achieved C-index
    model_cindex = calculate_c_index(predicted_risk_test, T_test, E_test)

    # Bayes-optimal C-index
    bayes_cindex = compute_bayes_optimal_concordance(
        X_test, beta, T_test, E_test
    )

    # Bayes-optimal with true times (if available)
    T_true = gt["test"].get("T_true")
    if T_true is not None:
        bayes_cindex_no_censoring = compute_bayes_optimal_concordance(
            X_test, beta, T_test, E_test,
            use_true_times=True, T_true=T_true
        )
    else:
        bayes_cindex_no_censoring = np.nan

    return {
        "risk_spearman_correlation": spearman_corr,
        "risk_spearman_pvalue": spearman_p,
        "model_cindex": model_cindex,
        "bayes_optimal_cindex": bayes_cindex,
        "bayes_optimal_cindex_no_censoring": bayes_cindex_no_censoring,
        "cindex_gap": bayes_cindex - model_cindex,
    }
