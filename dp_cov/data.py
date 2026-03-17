"""Data loading and preprocessing for DP covariance experiments."""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_wine,
)


def _show_progress(step, total, label):
    """Print a simple one-line progress bar for long dataset preparation steps."""
    width = 28
    filled = int(width * step / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = int(100 * step / total)
    print(f"\r[{bar}] {pct:3d}% {label}", end="", flush=True)
    if step == total:
        print()


def load_and_preprocess(path, sep=",", header="infer"):
    df = pd.read_csv(path, sep=sep, header=header)
    X = df.values.astype(float)

    # Input tables are loaded in the standard ML layout (n x d).
    # We standardize feature-columns, then transpose to the paper's d x n layout
    # where each column is one sample, and finally normalize each sample-column.
    X = X[:, X.std(axis=0) > 0]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = X.T
    col_norms = np.linalg.norm(X, axis=0, keepdims=True)
    X = X / col_norms
    return X


def _standardize_and_normalize(df):
    X = df.values.astype(float)
    X = X[:, X.std(axis=0) > 0]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = X.T                      # d x n
    col_norms = np.linalg.norm(X, axis=0, keepdims=True)
    X = X / np.maximum(col_norms, 1e-12)   # exact unit-norm normalization
    return X


def load_wine_appendix():
    """Wine dataset used in the supplement: sklearn's 13-feature wine data."""
    X, _ = load_wine(return_X_y=True, as_frame=True)
    return _standardize_and_normalize(X)


def load_airfoil_appendix():
    """Airfoil data from the local CSV, matching the 6-column setup in the supplement."""
    df = pd.read_csv("data/airfoil.csv", header=None)
    return _standardize_and_normalize(df)


def load_breast_cancer_appendix():
    """Breast Cancer Wisconsin dataset: d=30, n=569. sklearn built-in."""
    X, _ = load_breast_cancer(return_X_y=True, as_frame=True)
    return _standardize_and_normalize(X)


def load_digits_appendix():
    """Digits dataset: d=64, n=1797. sklearn built-in."""
    X, _ = load_digits(return_X_y=True, as_frame=True)
    return _standardize_and_normalize(X)


def load_california_housing_appendix():
    """California Housing dataset: d=8, n=20640. sklearn built-in."""
    data = fetch_california_housing(as_frame=True)
    return _standardize_and_normalize(data.data)


def load_volkert_appendix():
    """
    Volkert dataset from OpenML (ID 41166).

    A ChaLearn AutoML challenge dataset with 180 numerical features and 58,310
    samples. Features are already numerical and complete (no missing values),
    making it a clean high-dimensional tabular benchmark (d=180).
    """
    total_steps = 4
    cache_dir = "data/.sk_cache"
    _show_progress(0, total_steps, "Volkert: fetching OpenML dataset")
    volkert = fetch_openml(data_id=41166, as_frame=True, data_home=cache_dir)
    _show_progress(1, total_steps, "Volkert: extracting numerical features")
    X = volkert.data.copy().select_dtypes(include="number")
    _show_progress(2, total_steps, "Volkert: standardizing features")
    return_val = _standardize_and_normalize(X)
    _show_progress(3, total_steps, "Volkert: transposing and clipping samples")
    _show_progress(4, total_steps, f"Volkert: ready with shape {return_val.shape}")
    return return_val


def load_adult_appendix():
    """
    Adult dataset aligned to the supplement as closely as possible.

    We use the standard OpenML Adult dataset and one-hot encode categorical columns.
    This yields 105 features in this encoding, which is the closest standard local
    representation available to the supplement's reported 108-dimensional setup.
    """
    total_steps = 5
    cache_dir = "data/.sk_cache"
    _show_progress(0, total_steps, "Adult: fetching OpenML dataset")
    adult = fetch_openml("adult", version=2, as_frame=True, data_home=cache_dir)
    _show_progress(1, total_steps, "Adult: copying raw frame")
    X = adult.data.copy()
    _show_progress(2, total_steps, "Adult: one-hot encoding categorical columns")
    X = pd.get_dummies(X, drop_first=False)
    _show_progress(3, total_steps, "Adult: standardizing features")
    X = X.values.astype(float)
    X = X[:, X.std(axis=0) > 0]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    _show_progress(4, total_steps, "Adult: transposing and clipping samples")
    X = X.T
    col_norms = np.linalg.norm(X, axis=0, keepdims=True)
    X = X / np.maximum(col_norms, 1e-12)
    _show_progress(5, total_steps, f"Adult: ready with shape {X.shape}")
    return X


def subsample(X, n, rng):
    idx = rng.choice(X.shape[1], size=min(n, X.shape[1]), replace=False)
    return X[:, idx]
