"""Loaders for real-world survival datasets via pycox."""

import numpy as np
import pandas as pd
from typing import List

from .generator import SurvivalData


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in-place.

    Continuous columns (float dtype) are filled with the column median;
    integer/object columns are treated as categorical and filled with the mode.

    Args:
        df: DataFrame to impute.

    Returns:
        DataFrame with missing values filled.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def _standardize(X: np.ndarray) -> np.ndarray:
    """Standardize columns to zero mean and unit variance.

    Columns with zero variance are left unchanged.

    Args:
        X: Feature matrix of shape (n_samples, n_features).

    Returns:
        Standardized feature matrix.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    return (X - mean) / std


def _dataframe_to_survival_data(df: pd.DataFrame, feature_cols: List[str]) -> SurvivalData:
    """Convert a pycox-style DataFrame to SurvivalData.

    Args:
        df: DataFrame containing feature columns plus 'duration' and 'event'.
        feature_cols: Ordered list of feature column names.

    Returns:
        SurvivalData with standardized features and no ground-truth coefficients.
    """
    df = _impute(df)

    X = df[feature_cols].values.astype(np.float64)
    X = _standardize(X)

    T = df["duration"].values.astype(np.float64)
    E = df["event"].values.astype(np.float32)

    return SurvivalData(
        X=X,
        T=T,
        E=E,
        T_true=T.copy(),
        beta=np.array([]),
    )


def load_metabric() -> SurvivalData:
    """Load the METABRIC breast cancer dataset.

    Uses ``pycox.datasets.metabric``. The dataset contains 1,904 patients
    with 9 features (4 gene expression measurements and 5 clinical variables).
    The outcome is overall survival time in months.

    Returns:
        SurvivalData with standardized features. ``T_true`` equals ``T`` and
        ``beta`` is an empty array (no ground truth coefficients for real data).

    Raises:
        ImportError: If pycox is not installed.
    """
    try:
        from pycox.datasets import metabric
    except ImportError as exc:
        raise ImportError(
            "pycox is required to load METABRIC. Install it with: pip install pycox"
        ) from exc

    df = metabric.read_df()
    feature_cols = [c for c in df.columns if c not in ("duration", "event")]
    return _dataframe_to_survival_data(df, feature_cols)


def load_support() -> SurvivalData:
    """Load the SUPPORT (Study to Understand Prognoses and Preferences) dataset.

    Uses ``pycox.datasets.support``. The dataset contains 8,873 critically ill
    patients with 14 features. The outcome is survival time in days.

    Returns:
        SurvivalData with standardized features. ``T_true`` equals ``T`` and
        ``beta`` is an empty array (no ground truth coefficients for real data).

    Raises:
        ImportError: If pycox is not installed.
    """
    try:
        from pycox.datasets import support
    except ImportError as exc:
        raise ImportError(
            "pycox is required to load SUPPORT. Install it with: pip install pycox"
        ) from exc

    df = support.read_df()
    feature_cols = [c for c in df.columns if c not in ("duration", "event")]
    return _dataframe_to_survival_data(df, feature_cols)


def load_real_dataset(name: str) -> SurvivalData:
    """Load a real-world survival dataset by name.

    Args:
        name: Dataset identifier. Supported values are ``"metabric"`` and
            ``"support"`` (case-insensitive).

    Returns:
        SurvivalData for the requested dataset.

    Raises:
        ValueError: If ``name`` is not a recognised dataset identifier.
        ImportError: If pycox is not installed.
    """
    loaders = {
        "metabric": load_metabric,
        "support": load_support,
    }
    key = name.lower()
    if key not in loaders:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported datasets: {sorted(loaders)}"
        )
    return loaders[key]()
