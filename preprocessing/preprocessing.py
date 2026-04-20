# preprocessing/preprocessing.py
import pandas as pd

# Relative imports - files are in the same folder
from .encoding import auto_encode
from .scaling import auto_scale


# ══════════════════════════════════════════════════════════════
#  ANALYZE DATASET — Only for Encoding and Scaling
# ══════════════════════════════════════════════════════════════
def analyze_dataset(X: pd.DataFrame, task: str = None) -> dict:
    """
    Builds plan for:
    ii. Encode categorical variables
    iii. Scale/Normalize numerical features
    """
    plan = {
        "columns_to_drop": [],
        "encode_onehot":   [],
        "encode_label":    [],
        "scale_standard":  [],
        "scale_minmax":    [],
        "reasons":         {}
    }

    for col in X.columns:
        series = X[col]
        n_unique = series.nunique()
        dtype = series.dtype

        # Drop useless columns
        if n_unique <= 1:
            plan["columns_to_drop"].append(col)
            plan["reasons"][col] = "DROPPED — zero variance"
            continue

        is_string = str(dtype) in ['object', 'string'] or 'str' in str(dtype).lower()
        col_lower = col.lower()
        is_id = any(k in col_lower for k in ['id', '_id', 'index', 'uuid', 'key'])

        if (is_string and n_unique == len(series)) or (is_id and n_unique == len(series)):
            plan["columns_to_drop"].append(col)
            plan["reasons"][col] = "DROPPED — likely ID column"
            continue

        # ii. Encode categorical variables
        if is_string or str(dtype) in ['category', 'bool']:
            if n_unique <= 15:
                plan["encode_onehot"].append(col)
                plan["reasons"][col] = f"ENCODE with OneHotEncoder ({n_unique} unique values)"
            else:
                plan["encode_label"].append(col)
                plan["reasons"][col] = f"ENCODE with LabelEncoder ({n_unique} unique values)"

        # iii. Scale/Normalize numerical features
        elif str(dtype) in ['int64', 'float64', 'int32', 'float32']:
            col_min = series.min()
            col_max = series.max()
            if col_min >= 0 and col_max <= 100:
                plan["scale_minmax"].append(col)
                plan["reasons"][col] = f"SCALE with MinMaxScaler (range {col_min:.0f}–{col_max:.0f})"
            else:
                plan["scale_standard"].append(col)
                plan["reasons"][col] = f"SCALE with StandardScaler (range {col_min:.0f}–{col_max:.0f})"

    return plan


# ══════════════════════════════════════════════════════════════
#  MAIN PREPROCESS — Only Encoding + Scaling
# ══════════════════════════════════════════════════════════════
def preprocess(X: pd.DataFrame, y=None, task: str = None):
    info = {}

    plan = analyze_dataset(X.copy(), task)
    info["plan"] = plan
    info["reasons"] = plan["reasons"]

    # Drop useless columns
    if plan["columns_to_drop"]:
        X = X.drop(columns=plan["columns_to_drop"])

    info["columns_dropped"] = plan["columns_to_drop"]
    info["columns_kept"] = list(X.columns)
    info["shape_after_drop"] = list(X.shape)

    # ii. Encode categorical variables
    info["shape_before_encoding"] = list(X.shape)
    info["ohe_columns"] = plan["encode_onehot"]
    info["label_columns"] = plan["encode_label"]
    info["categorical_columns_encoded"] = plan["encode_onehot"] + plan["encode_label"]

    encoding_details = {}
    for col in plan["encode_onehot"]:
        if col in X.columns:
            unique_vals = sorted(X[col].dropna().astype(str).unique().tolist())
            encoding_details[col] = {
                "technique": "OneHotEncoder",
                "unique_count": len(unique_vals),
                "unique_values": unique_vals,
                "new_cols": len(unique_vals)
            }
    for col in plan["encode_label"]:
        if col in X.columns:
            unique_vals = sorted(X[col].dropna().astype(str).unique().tolist())
            encoding_details[col] = {
                "technique": "LabelEncoder",
                "unique_count": len(unique_vals),
                "unique_values": unique_vals[:10],
                "new_cols": 1
            }

    info["encoding_details"] = encoding_details

    X_encoded, ohe_encoder, label_encoders = auto_encode(
        X, ohe_cols=plan["encode_onehot"], label_cols=plan["encode_label"]
    )

    info["shape_after_encoding"] = list(getattr(X_encoded, 'shape', (len(X_encoded), 0)))
    info["encoding"] = "OneHotEncoder (≤15) + LabelEncoder (>15)"

    # iii. Scale/Normalize numerical features
    X_df = pd.DataFrame(X_encoded) if not isinstance(X_encoded, pd.DataFrame) else X_encoded

    info["scale_standard_cols"] = plan["scale_standard"]
    info["scale_minmax_cols"] = plan["scale_minmax"]
    info["numerical_columns_scaled"] = plan["scale_standard"] + plan["scale_minmax"]

    X_scaled_df, std_scaler, mm_scaler = auto_scale(
        X_df, standard_cols=plan["scale_standard"], minmax_cols=plan["scale_minmax"]
    )

    info["scaling"] = "StandardScaler + MinMaxScaler (auto selected)"
    info["scaler"] = std_scaler or mm_scaler

    X_final = X_scaled_df.values if isinstance(X_scaled_df, pd.DataFrame) else X_scaled_df

    info["missing_handled"] = False
    info["resampling"] = "Not applied"

    return X_final, y, info


def get_serializable_info(info: dict) -> dict:
    return {
        "columns_dropped": info.get("columns_dropped", []),
        "columns_kept": info.get("columns_kept", []),
        "ohe_columns": info.get("ohe_columns", []),
        "label_columns": info.get("label_columns", []),
        "categorical_columns_encoded": info.get("categorical_columns_encoded", []),
        "encoding_details": info.get("encoding_details", {}),
        "shape_before_encoding": info.get("shape_before_encoding", []),
        "shape_after_encoding": info.get("shape_after_encoding", []),
        "scale_standard_cols": info.get("scale_standard_cols", []),
        "scale_minmax_cols": info.get("scale_minmax_cols", []),
        "numerical_columns_scaled": info.get("numerical_columns_scaled", []),
        "encoding": info.get("encoding", ""),
        "scaling": info.get("scaling", ""),
        "resampling": info.get("resampling", "Not applied"),
        "reasons": info.get("reasons", {}),
    }