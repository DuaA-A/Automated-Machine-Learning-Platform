# encoding.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


def get_categorical_columns(X: pd.DataFrame) -> list:
    return X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 1: OneHotEncoder
# Creates a new binary column for each unique category value.
# Example: city = [Cairo, Alex] → city_Cairo=1,0 | city_Alex=0,1
# Best for: columns with FEW unique values (≤ 15)
# ─────────────────────────────────────────────────────────────
def encode_onehot(X: pd.DataFrame):
    cat_cols = get_categorical_columns(X)
    if not cat_cols:
        return X.values, None
    ct = ColumnTransformer(
        transformers=[('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)],
        remainder='passthrough'
    )
    X_encoded = ct.fit_transform(X)
    return X_encoded, ct


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 2: LabelEncoder
# Replaces each category with a single integer.
# Example: city = [Cairo, Alex, Cairo] → [1, 0, 1]
# Best for: columns with MANY unique values (> 15)
# ─────────────────────────────────────────────────────────────
def encode_label(X: pd.DataFrame):
    cat_cols = get_categorical_columns(X)
    X_encoded = X.copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le
    return X_encoded.values, encoders


# ─────────────────────────────────────────────────────────────
# SMART AUTO SELECTOR
# Picks OHE for columns with ≤ threshold unique values,
# LabelEncoder for columns with > threshold unique values.
# Also accepts explicit lists so preprocessing.py can pass
# exactly which columns go where (from analyze_dataset).
# ─────────────────────────────────────────────────────────────
def auto_encode(X: pd.DataFrame,
                threshold: int = 15,
                ohe_cols: list = None,
                label_cols: list = None):
    """
    If ohe_cols / label_cols are provided (from analyze_dataset),
    use them directly. Otherwise auto-detect from threshold.
    """
    cat_cols = get_categorical_columns(X)

    # Use provided lists if given, otherwise auto-detect
    if ohe_cols is None:
        ohe_cols = [c for c in cat_cols if X[c].nunique() <= threshold]
    if label_cols is None:
        label_cols = [c for c in cat_cols if X[c].nunique() > threshold]

    X_result = X.copy()
    label_encoders = {}

    # Apply LabelEncoder first (in-place on DataFrame)
    for col in label_cols:
        if col in X_result.columns:
            le = LabelEncoder()
            X_result[col] = le.fit_transform(X_result[col].astype(str))
            label_encoders[col] = le

    # Apply OneHotEncoder via ColumnTransformer
    ohe_encoder = None
    valid_ohe = [c for c in ohe_cols if c in X_result.columns]

    if valid_ohe:
        ct = ColumnTransformer(
            transformers=[('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), valid_ohe)],
            remainder='passthrough'
        )
        X_result = ct.fit_transform(X_result)
        ohe_encoder = ct
    else:
        X_result = X_result.values

    return X_result, ohe_encoder, label_encoders