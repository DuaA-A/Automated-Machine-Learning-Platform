# Scaling.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_numerical_columns(X: pd.DataFrame) -> list:
    return X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 1: StandardScaler
# Formula : z = (x - mean) / std
# Result  : mean = 0, std = 1
# Best for: large/unbounded ranges, outliers, SVM, LR, PCA
# ─────────────────────────────────────────────────────────────
def scale_standard(X: pd.DataFrame):
    num_cols = get_numerical_columns(X)
    if not num_cols:
        return X, None
    scaler   = StandardScaler()
    X_scaled = X.copy()
    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])
    return X_scaled, scaler


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 2: MinMaxScaler
# Formula : x_scaled = (x - min) / (max - min)
# Result  : all values in [0, 1]
# Best for: bounded ranges (0-100), KNN, Neural Networks
# ─────────────────────────────────────────────────────────────
def scale_minmax(X: pd.DataFrame):
    num_cols = get_numerical_columns(X)
    if not num_cols:
        return X, None
    scaler   = MinMaxScaler()
    X_scaled = X.copy()
    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])
    return X_scaled, scaler


# ─────────────────────────────────────────────────────────────
# SMART AUTO SELECTOR — called by preprocessing.py
# Applies StandardScaler to standard_cols list
# Applies MinMaxScaler  to minmax_cols list
# Returns (X_scaled DataFrame, std_scaler, mm_scaler)
# ─────────────────────────────────────────────────────────────
def auto_scale(X: pd.DataFrame,
               standard_cols: list = None,
               minmax_cols:   list = None):
    """
    standard_cols : columns to scale with StandardScaler
    minmax_cols   : columns to scale with MinMaxScaler
    If neither is provided, StandardScaler is applied to all numerical cols.
    Returns: (X_scaled DataFrame, std_scaler or None, mm_scaler or None)
    """
    num_cols = get_numerical_columns(X)
    if not num_cols:
        return X, None, None

    # Default: scale everything with StandardScaler
    if standard_cols is None and minmax_cols is None:
        standard_cols = num_cols
        minmax_cols   = []

    X_scaled   = X.copy()
    std_scaler = None
    mm_scaler  = None

    valid_std = [c for c in (standard_cols or []) if c in X_scaled.columns]
    valid_mm  = [c for c in (minmax_cols   or []) if c in X_scaled.columns]

    if valid_std:
        std_scaler = StandardScaler()
        X_scaled[valid_std] = std_scaler.fit_transform(X_scaled[valid_std])

    if valid_mm:
        mm_scaler = MinMaxScaler()
        X_scaled[valid_mm] = mm_scaler.fit_transform(X_scaled[valid_mm])

    return X_scaled, std_scaler, mm_scaler