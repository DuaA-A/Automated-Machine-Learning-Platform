# test_preprocessing.py
import pandas as pd
import numpy as np
from preprocessing import preprocess

print("=" * 60)
print("  PREPROCESSING TEST — IS424 AutoML Project")
print("=" * 60)

# ── Build a messy test dataset ─────────────────────────────────
data = pd.DataFrame({
    'age':    [25, np.nan, 35,  45,    22, np.nan, 60],
    'salary': [50000, 60000, np.nan, 80000, 45000, 70000, 90000],
    'gender': ['male', 'female', np.nan, 'male', 'female', 'female', 'male'],
    'city':   ['Cairo', 'Alex', 'Cairo', np.nan, 'Alex', 'Cairo', 'Alex'],
    'label':  [1, 0, 0, 1, 0, 0, 1]
})

X = data.drop('label', axis=1)
y = data['label']

print("\n[STEP 0] ORIGINAL DATA (messy — has NaN values):")
print(X.to_string())
print(f"\nShape: {X.shape}  ({X.shape[0]} rows, {X.shape[1]} columns)")
print(f"Missing values per column:\n{X.isnull().sum().to_string()}")

# ── Run preprocessing ──────────────────────────────────────────
X_out, y_out, info = preprocess(X.copy(), y.copy(), task='Classification')

# ── Decision plan ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("[ANALYZE] WHAT THE PIPELINE DECIDED FOR THIS DATASET")
print("=" * 60)
for col, reason in info["reasons"].items():
    label = "TARGET" if col == "__target__" else col
    print(f"  {label:<12} : {reason}")

# ── STEP 1 result ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 1] AFTER MISSING VALUE HANDLING")
print("=" * 60)
print(f"  Imputed with median : {info['imputed_median']}")
print(f"  Imputed with mode   : {info['imputed_mode']}")
print(f"  Status : {'PASS — No NaN remaining' if not np.isnan(X_out).any() else 'FAIL'}")

# ── STEP 2 result ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 2] AFTER CATEGORICAL ENCODING")
print("=" * 60)
print(f"  OneHotEncoder on  : {info['ohe_columns']}")
print(f"  LabelEncoder on   : {info['label_columns']}")
print(f"  Shape before encoding : {info['shape_before_encoding']}")
print(f"  Shape after encoding  : {info['shape_after_encoding']}")

# ── STEP 3 result ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 3] AFTER NUMERICAL SCALING")
print("=" * 60)
print(f"  StandardScaler on : {info['scale_standard_cols']}")
print(f"  MinMaxScaler on   : {info['scale_minmax_cols']}")
print(f"\n  Final data (first 3 rows):")
print(np.round(X_out[:3], 3))
col_means = X_out.mean(axis=0).round(4)
print(f"\n  Column means (should all be ~0.0): {col_means}")
all_near_zero = all(abs(m) < 0.01 for m in col_means)
print(f"  Scaling check: {'PASS — all means ≈ 0' if all_near_zero else 'FAIL'}")

# ── STEP 4 result ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 4] CLASS IMBALANCE CHECK (Classification only)")
print("=" * 60)
original_counts = y.value_counts()
new_counts      = pd.Series(y_out).value_counts()
ratio = original_counts.min() / original_counts.max()
print(f"  Original class distribution : {original_counts.to_dict()}")
print(f"  Imbalance ratio             : {ratio:.2f}")
print(f"  Resampling applied          : {info['resampling']}")
print(f"  Class distribution after    : {new_counts.to_dict()}")

# ── FINAL SUMMARY ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Input  shape  : {X.shape}")
print(f"  Output shape  : {X_out.shape}")
print(f"  NaN remaining : {np.isnan(X_out).any()}")
print(f"  Encoding      : {info['encoding']}")
print(f"  Scaling       : {info['scaling']}")
print(f"  Resampling    : {info['resampling']}")

all_pass = not np.isnan(X_out).any() and all_near_zero
print(f"\n  OVERALL RESULT: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
print("=" * 60)

# ── BONUS: Imbalanced dataset test ────────────────────────────
print("\n\n" + "=" * 60)
print("  BONUS TEST — Imbalanced Dataset (SMOTE check)")
print("=" * 60)

from sklearn.datasets import make_classification
X_imb, y_imb = make_classification(
    n_samples=100, n_features=5,
    weights=[0.9, 0.1], random_state=42
)
X_imb_df = pd.DataFrame(X_imb, columns=[f'feature_{i}' for i in range(5)])
y_imb_s  = pd.Series(y_imb)

print(f"  Before SMOTE — class counts: {y_imb_s.value_counts().to_dict()}")
X_imb_out, y_imb_out, info2 = preprocess(X_imb_df.copy(), y_imb_s.copy(), task='Classification')
print(f"  After  SMOTE — class counts: {pd.Series(y_imb_out).value_counts().to_dict()}")
print(f"  Resampling info : {info2['resampling']}")
smote_pass = pd.Series(y_imb_out).value_counts().min() > 10
print(f"  SMOTE check : {'PASS — minority class increased' if smote_pass else 'FAIL'}")
print("=" * 60)