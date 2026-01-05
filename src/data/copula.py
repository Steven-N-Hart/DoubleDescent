"""Gaussian copula for generating correlated covariates."""

import numpy as np
from scipy import stats
from typing import Optional


def generate_correlated_uniform(
    n_samples: int,
    n_features: int,
    correlation_matrix: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate correlated uniform random variables using Gaussian copula.

    The Gaussian copula transforms independent normal variables to
    correlated uniform variables, which can then be transformed to
    any desired marginal distribution.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features (dimensions).
        correlation_matrix: Correlation matrix for the Gaussian copula.
            If None, uses identity matrix (independent features).
            Must be positive semi-definite.
        rng: NumPy random generator for reproducibility.

    Returns:
        Array of shape (n_samples, n_features) with values in [0, 1].

    Raises:
        ValueError: If correlation matrix is not positive semi-definite.
    """
    if rng is None:
        rng = np.random.default_rng()

    if correlation_matrix is None:
        # Independent features
        return rng.uniform(0, 1, size=(n_samples, n_features))

    correlation_matrix = np.asarray(correlation_matrix)

    # Validate correlation matrix
    if correlation_matrix.shape != (n_features, n_features):
        raise ValueError(
            f"Correlation matrix shape {correlation_matrix.shape} "
            f"does not match n_features={n_features}"
        )

    # Check positive semi-definiteness
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)
    if np.min(eigenvalues) < -1e-10:
        raise ValueError("Correlation matrix must be positive semi-definite")

    # Generate correlated normal samples
    try:
        # Use Cholesky decomposition for efficiency
        L = np.linalg.cholesky(correlation_matrix)
        Z = rng.standard_normal(size=(n_samples, n_features))
        correlated_normal = Z @ L.T
    except np.linalg.LinAlgError:
        # Fall back to eigendecomposition if Cholesky fails
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
        sqrt_cov = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        Z = rng.standard_normal(size=(n_samples, n_features))
        correlated_normal = Z @ sqrt_cov.T

    # Transform to uniform using normal CDF
    uniform_samples = stats.norm.cdf(correlated_normal)

    return uniform_samples


def create_ar1_correlation(n_features: int, rho: float) -> np.ndarray:
    """Create AR(1) correlation matrix.

    AR(1) structure: corr(i, j) = rho^|i-j|

    Args:
        n_features: Matrix dimension.
        rho: Correlation parameter in (-1, 1).

    Returns:
        Correlation matrix of shape (n_features, n_features).
    """
    if not -1 < rho < 1:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")

    indices = np.arange(n_features)
    return rho ** np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])


def create_block_correlation(
    n_features: int,
    n_blocks: int,
    within_block_rho: float,
    between_block_rho: float = 0.0,
) -> np.ndarray:
    """Create block-structured correlation matrix.

    Features within the same block have correlation `within_block_rho`,
    features in different blocks have correlation `between_block_rho`.

    Args:
        n_features: Total number of features.
        n_blocks: Number of blocks.
        within_block_rho: Correlation within blocks.
        between_block_rho: Correlation between blocks.

    Returns:
        Correlation matrix of shape (n_features, n_features).
    """
    if not 0 <= within_block_rho < 1:
        raise ValueError(f"within_block_rho must be in [0, 1), got {within_block_rho}")
    if not -1 < between_block_rho < 1:
        raise ValueError(
            f"between_block_rho must be in (-1, 1), got {between_block_rho}"
        )

    block_size = n_features // n_blocks
    remainder = n_features % n_blocks

    # Initialize with between-block correlation
    corr_matrix = np.full((n_features, n_features), between_block_rho)

    # Set within-block correlations
    start = 0
    for b in range(n_blocks):
        size = block_size + (1 if b < remainder else 0)
        end = start + size
        corr_matrix[start:end, start:end] = within_block_rho
        start = end

    # Set diagonal to 1
    np.fill_diagonal(corr_matrix, 1.0)

    return corr_matrix
