"""Statistical significance testing utilities.

Provides functions for comparing experimental results across
conditions with proper statistical testing.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


def paired_significance_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Perform paired t-test between two conditions.

    Use this when comparing the same models/seeds across two conditions
    (e.g., regularized vs unregularized on same random seeds).

    Args:
        values_a: Values from condition A (one per seed).
        values_b: Values from condition B (same seeds as A).
        alternative: "two-sided", "less", or "greater".

    Returns:
        Dictionary with:
            - t_statistic: t-test statistic
            - p_value: p-value
            - mean_diff: Mean difference (B - A)
            - std_diff: Std of differences
            - cohens_d: Effect size
            - ci_lower, ci_upper: 95% CI for mean difference
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    if len(values_a) != len(values_b):
        raise ValueError("Arrays must have same length for paired test")

    differences = values_b - values_a
    n = len(differences)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(values_b, values_a, alternative=alternative)

    # Effect size (Cohen's d for paired samples)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    # 95% confidence interval for mean difference
    se = std_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "cohens_d": float(cohens_d),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n": int(n),
    }


def independent_significance_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alternative: str = "two-sided",
    equal_var: bool = False,
) -> Dict[str, float]:
    """Perform independent samples t-test.

    Use this when comparing different models/conditions that don't
    share the same seeds.

    Args:
        values_a: Values from condition A.
        values_b: Values from condition B.
        alternative: "two-sided", "less", or "greater".
        equal_var: Assume equal variances (Welch's t-test if False).

    Returns:
        Dictionary with test statistics and effect size.
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)
    std_a = np.std(values_a, ddof=1)
    std_b = np.std(values_b, ddof=1)

    # Independent t-test (Welch's by default)
    t_stat, p_value = stats.ttest_ind(
        values_a, values_b,
        alternative=alternative,
        equal_var=equal_var,
    )

    # Pooled standard deviation for effect size
    n_a, n_b = len(values_a), len(values_b)
    pooled_std = np.sqrt(
        ((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2)
    )
    cohens_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0.0

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_a": float(mean_a),
        "mean_b": float(mean_b),
        "mean_diff": float(mean_b - mean_a),
        "cohens_d": float(cohens_d),
        "n_a": int(n_a),
        "n_b": int(n_b),
    }


def compute_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute confidence interval for mean.

    Args:
        values: Sample values.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (mean, ci_lower, ci_upper).
    """
    values = np.asarray(values)
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)

    # t-distribution critical value
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se

    return float(mean), float(ci_lower), float(ci_upper)


def multiple_comparison_correction(
    p_values: Union[List[float], np.ndarray],
    method: str = "holm",
) -> np.ndarray:
    """Correct p-values for multiple comparisons.

    Args:
        p_values: Array of p-values.
        method: Correction method:
            - "bonferroni": Bonferroni correction (conservative)
            - "holm": Holm-Bonferroni (less conservative)
            - "fdr": Benjamini-Hochberg FDR control

    Returns:
        Corrected p-values.
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    if method == "bonferroni":
        return np.minimum(p_values * n, 1.0)

    elif method == "holm":
        # Holm-Bonferroni step-down procedure
        order = np.argsort(p_values)
        adjusted = np.zeros(n)

        for i, idx in enumerate(order):
            adjusted[idx] = p_values[idx] * (n - i)

        # Ensure monotonicity
        for i in range(1, n):
            idx = order[i]
            prev_idx = order[i - 1]
            adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])

        return np.minimum(adjusted, 1.0)

    elif method == "fdr":
        # Benjamini-Hochberg procedure
        order = np.argsort(p_values)
        adjusted = np.zeros(n)

        for i, idx in enumerate(order):
            adjusted[idx] = p_values[idx] * n / (i + 1)

        # Ensure monotonicity (from end to start)
        for i in range(n - 2, -1, -1):
            idx = order[i]
            next_idx = order[i + 1]
            adjusted[idx] = min(adjusted[idx], adjusted[next_idx])

        return np.minimum(adjusted, 1.0)

    else:
        raise ValueError(f"Unknown correction method: {method}")


def summarize_regularization_effect(
    baseline_cindex: np.ndarray,
    regularized_cindex: np.ndarray,
    width: int,
) -> Dict[str, Union[float, str]]:
    """Summarize the effect of regularization at a specific width.

    Args:
        baseline_cindex: C-index values from unregularized models (per seed).
        regularized_cindex: C-index values from regularized models (per seed).
        width: Model width for reporting.

    Returns:
        Dictionary with formatted summary for manuscript.
    """
    result = paired_significance_test(baseline_cindex, regularized_cindex)

    # Determine significance
    if result["p_value"] < 0.001:
        sig_str = "p < 0.001"
    elif result["p_value"] < 0.01:
        sig_str = f"p = {result['p_value']:.3f}"
    elif result["p_value"] < 0.05:
        sig_str = f"p = {result['p_value']:.3f}"
    else:
        sig_str = f"p = {result['p_value']:.2f}, not significant"

    # Effect size interpretation
    d = abs(result["cohens_d"])
    if d < 0.2:
        effect_str = "negligible"
    elif d < 0.5:
        effect_str = "small"
    elif d < 0.8:
        effect_str = "medium"
    else:
        effect_str = "large"

    return {
        "width": width,
        "mean_improvement": result["mean_diff"],
        "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"],
        "p_value": result["p_value"],
        "cohens_d": result["cohens_d"],
        "significance_string": sig_str,
        "effect_size_interpretation": effect_str,
        "n_seeds": result["n"],
        "manuscript_text": (
            f"an improvement of {result['mean_diff']:.3f} "
            f"(95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}], "
            f"paired t-test {sig_str})"
        ),
    }
