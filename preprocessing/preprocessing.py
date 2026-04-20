# preprocessing.py
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# Import your encoding and scaling files
from encoding import auto_encode
from Scaling  import auto_scale


# ══════════════════════════════════════════════════════════════
#  ANALYZE — scans the dataset and builds a decision plan
# ══════════════════════════════════════════════════════════════
def analyze_dataset(X: pd.DataFrame, y: pd.Series = None, task: str = None) -> dict:
    """
    Looks at every column and decides:
    - which columns to drop (useless)
    - which columns need imputation and what strategy
    - which columns need encoding and which technique
    - which columns need scaling and which technique
    - whether SMOTE resampling is needed
    """
    plan = {
        "columns_to_drop":  [],
        "impute_median":    [],   # numerical with missing values
        "impute_mode":      [],   # categorical with missing values
        "encode_onehot":    [],   # categorical, ≤ 15 unique values
        "encode_label":     [],   # categorical, > 15 unique values
        "scale_standard":   [],   # numerical, large/unbounded range
        "scale_minmax":     [],   # numerical, bounded range [0–100]
        "needs_resampling": False,
        "reasons":          {}
    }

    for col in X.columns:
        series    = X[col]
        n_unique  = series.nunique()
        n_missing = series.isnull().sum()
        dtype     = series.dtype

        # ── DROP useless columns ─────────────────────────────
        if n_missing == len(series):
            plan["columns_to_drop"].append(col)
            plan["reasons"][col] = "DROPPED — every value is missing"
            continue

        if n_unique <= 1:
            plan["columns_to_drop"].append(col)
            plan["reasons"][col] = "DROPPED — only one unique value, no information"
            continue

        if n_unique == len(series) and dtype in ['int64', 'object']:
            plan["columns_to_drop"].append(col)
            plan["reasons"][col] = "DROPPED — all values unique, likely an ID column"
            continue

        # ── NUMERICAL column ─────────────────────────────────
        if dtype in ['int64', 'float64', 'int32', 'float32']:

            if n_missing > 0:
                plan["impute_median"].append(col)
                plan["reasons"][col] = f"IMPUTE with median ({n_missing} missing values)"

            col_min   = series.dropna().min()
            col_max   = series.dropna().max()

            # MinMaxScaler for already-bounded ranges (percentages, scores, probabilities)
            if col_min >= 0 and col_max <= 100:
                plan["scale_minmax"].append(col)
                plan["reasons"][col] = plan["reasons"].get(col, "") + \
                    f" | MinMaxScaler (bounded range [{col_min:.1f}–{col_max:.1f}])"
            else:
                # StandardScaler for large values, negatives, salaries, etc.
                plan["scale_standard"].append(col)
                plan["reasons"][col] = plan["reasons"].get(col, "") + \
                    f" | StandardScaler (range [{col_min:.1f}–{col_max:.1f}])"

        # ── CATEGORICAL column ───────────────────────────────
        elif dtype in ['object', 'category', 'bool']:

            if n_missing > 0:
                plan["impute_mode"].append(col)
                plan["reasons"][col] = f"IMPUTE with mode ({n_missing} missing values)"

            if n_unique <= 15:
                plan["encode_onehot"].append(col)
                plan["reasons"][col] = plan["reasons"].get(col, "") + \
                    f" | OneHotEncoder ({n_unique} unique values ≤ 15)"
            else:
                plan["encode_label"].append(col)
                plan["reasons"][col] = plan["reasons"].get(col, "") + \
                    f" | LabelEncoder ({n_unique} unique values > 15)"

    # ── RESAMPLING decision ──────────────────────────────────
    if task == "Classification" and y is not None:
        counts = y.value_counts()
        ratio  = counts.min() / counts.max()
        if ratio < 0.4:
            plan["needs_resampling"] = True
            plan["reasons"]["__target__"] = \
                f"SMOTE applied — imbalance ratio {ratio:.2f} (minority < 40% of majority)"
        else:
            plan["reasons"]["__target__"] = \
                f"No resampling needed — ratio {ratio:.2f}, classes are balanced"

    return plan


# ══════════════════════════════════════════════════════════════
#  MAIN FUNCTION — runs the adaptive pipeline
# ══════════════════════════════════════════════════════════════
def preprocess(X: pd.DataFrame, y: pd.Series, task: str):
    """
    Analyzes the dataset then applies only what is needed.
    Calls encoding.py and scaling.py internally.

    Returns: X_processed (numpy array), y_processed, info_dict
    """
    info = {}

    # Record missing values before touching anything
    info["missing_before"] = {
        col: int(X[col].isnull().sum())
        for col in X.columns
        if X[col].isnull().sum() > 0
    }

    # Build the decision plan by analyzing the data
    plan          = analyze_dataset(X.copy(), y, task)
    info["plan"]    = plan
    info["reasons"] = plan["reasons"]

    # ── STEP 1: Drop useless columns ─────────────────────────
    if plan["columns_to_drop"]:
        X = X.drop(columns=plan["columns_to_drop"])
    info["columns_dropped"] = plan["columns_to_drop"]

    # ── STEP 2: Handle missing values ────────────────────────
    for col in plan["impute_median"]:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    for col in plan["impute_mode"]:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].mode()[0])

    info["missing_handled"] = True
    info["imputed_median"]  = [c for c in plan["impute_median"] if c in X.columns]
    info["imputed_mode"]    = [c for c in plan["impute_mode"]   if c in X.columns]

    # ── STEP 3: Encode categorical columns ───────────────────
    # Calls auto_encode() from encoding.py
    # Passes ohe_cols and label_cols from the decision plan
    info["shape_before_encoding"]       = list(X.shape)
    info["categorical_columns_encoded"] = plan["encode_onehot"] + plan["encode_label"]
    info["ohe_columns"]                 = plan["encode_onehot"]
    info["label_columns"]               = plan["encode_label"]

    X_encoded, ohe_encoder, label_encoders = auto_encode(
        X,
        ohe_cols   = plan["encode_onehot"],
        label_cols = plan["encode_label"]
    )

    info["shape_after_encoding"] = list(X_encoded.shape)
    info["ohe_encoder"]          = ohe_encoder
    info["label_encoders"]       = label_encoders
    info["encoding"]             = "OHE (low-cardinality) + LabelEncoder (high-cardinality)"

    # ── STEP 4: Scale numerical columns ──────────────────────
    # Calls auto_scale() from scaling.py
    # Passes which columns use StandardScaler vs MinMaxScaler
    X_df = pd.DataFrame(X_encoded)

    info["numerical_columns_scaled"] = plan["scale_standard"] + plan["scale_minmax"]
    info["scale_standard_cols"]      = plan["scale_standard"]
    info["scale_minmax_cols"]        = plan["scale_minmax"]

    # After encoding, column names are lost — apply scalers to ALL columns
    # (they are all numeric at this point)
    all_cols = list(range(X_df.shape[1]))

    # Use StandardScaler on the full encoded array
    # (MinMaxScaler cols are handled within auto_scale via original col names
    #  only when they still exist; after OHE they lose names so we standardize all)
    X_scaled_df, std_scaler, mm_scaler = auto_scale(
        X_df,
        standard_cols = list(X_df.columns),  # scale all columns after encoding
        minmax_cols   = []
    )

    info["scaling"]  = "StandardScaler (mean=0, std=1)"
    info["scaler"]   = std_scaler

    X_final = X_scaled_df.values if isinstance(X_scaled_df, pd.DataFrame) else X_scaled_df

    # ── STEP 5: Handle class imbalance ───────────────────────
    if task == "Classification" and y is not None:
        counts = y.value_counts()
        info["class_distribution_before"] = {str(k): int(v) for k, v in counts.items()}

        ratio = counts.min() / counts.max()
        if plan["needs_resampling"]:
            smote      = SMOTE(random_state=42)
            X_final, y = smote.fit_resample(X_final, y)
            info["resampling"] = f"SMOTE applied (ratio was {ratio:.2f})"
        else:
            info["resampling"] = f"Not needed (ratio={ratio:.2f}, balanced)"

        info["class_distribution_after"] = {
            str(k): int(v) for k, v in pd.Series(y).value_counts().items()
        }
    else:
        info["resampling"]               = "Not applicable"
        info["class_distribution_before"] = {}
        info["class_distribution_after"]  = {}

    return X_final, y, info


# ══════════════════════════════════════════════════════════════
#  HELPER — strips non-serializable objects for JSON response
# ══════════════════════════════════════════════════════════════
def get_serializable_info(info: dict) -> dict:
    """Call this in your backend before returning the JSON response."""
    return {
        "missing_before":               info.get("missing_before", {}),
        "columns_dropped":              info.get("columns_dropped", []),
        "imputed_median":               info.get("imputed_median", []),
        "imputed_mode":                 info.get("imputed_mode", []),
        "categorical_columns_encoded":  info.get("categorical_columns_encoded", []),
        "ohe_columns":                  info.get("ohe_columns", []),
        "label_columns":                info.get("label_columns", []),
        "shape_before_encoding":        info.get("shape_before_encoding", []),
        "shape_after_encoding":         info.get("shape_after_encoding", []),
        "numerical_columns_scaled":     info.get("numerical_columns_scaled", []),
        "scale_standard_cols":          info.get("scale_standard_cols", []),
        "scale_minmax_cols":            info.get("scale_minmax_cols", []),
        "encoding":                     info.get("encoding", ""),
        "scaling":                      info.get("scaling", ""),
        "resampling":                   info.get("resampling", ""),
        "class_distribution_before":    info.get("class_distribution_before", {}),
        "class_distribution_after":     info.get("class_distribution_after", {}),
        "reasons":                      info.get("reasons", {}),
    }