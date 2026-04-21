# test_preprocessing.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from preprocessing.preprocessing import build_preprocessing_pipeline

print("=" * 60)
print("  PREPROCESSING TEST — IS424 AutoML Project")
print("=" * 60)


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

print("\n" + "=" * 60)
print(" BUILDING AND RUNNING PIPELINE...")
print("=" * 60)


pipeline = build_preprocessing_pipeline(X, task='Classification')


X_out, y_out = pipeline.fit_resample(X, y)


if getattr(X_out, "to_string", None) is None:
    X_out = pd.DataFrame(X_out)



print("\n" + "=" * 60)
print("[STEP 1 & 2] IMPUTATION AND ENCODING CHECKS")
print("=" * 60)

has_nans = pd.isna(X_out).any().any()
print(f"  Missing values handled : {'PASS — No NaN remaining' if not has_nans else 'FAIL'}")


print(f"  Shape before encoding  : {X.shape}")
print(f"  Shape after processing : {X_out.shape}")
print(f"  Final columns          : {list(X_out.columns)}")



print("\n" + "=" * 60)
print("[STEP 3] NUMERICAL SCALING CHECK")
print("=" * 60)
print(f"  Final data (first 3 rows):")
print(np.round(X_out.head(3), 3))


numeric_cols = X.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:

    out_numeric_cols = [c for c in X_out.columns if any(orig in c for orig in numeric_cols)]
    col_means = X_out[out_numeric_cols].mean(axis=0).round(2)
    print(f"\n  Scaled Column means (approx): \n{col_means.to_string()}")
else:
    print("  No numeric columns to check.")



print("\n" + "=" * 60)
print("[STEP 4] CLASS IMBALANCE CHECK (SMOTENC)")
print("=" * 60)
original_counts = y.value_counts()
new_counts      = pd.Series(y_out).value_counts()
ratio_before = original_counts.min() / original_counts.max()
ratio_after  = new_counts.min() / new_counts.max()

print(f"  Original class distribution : {original_counts.to_dict()}")
print(f"  Imbalance ratio before      : {ratio_before:.2f}")
print(f"  Class distribution after    : {new_counts.to_dict()}")
print(f"  Imbalance ratio after       : {ratio_after:.2f}")
print(f"  SMOTENC check               : {'PASS — Classes are balanced' if ratio_after == 1.0 else 'FAIL'}")



print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Input  shape  : {X.shape}")
print(f"  Output shape  : {X_out.shape}")
print(f"  NaN remaining : {has_nans}")

all_pass = not has_nans and (ratio_after == 1.0)
print(f"\n  OVERALL RESULT: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
print("=" * 60)



print("\n\n" + "=" * 60)
print("  BONUS TEST — Imbalanced Dataset (Pure Numeric SMOTE)")
print("=" * 60)

from sklearn.datasets import make_classification
X_imb, y_imb = make_classification(
    n_samples=100, n_features=5,
    weights=[0.9, 0.1], random_state=42
)
X_imb_df = pd.DataFrame(X_imb, columns=[f'feature_{i}' for i in range(5)])
y_imb_s  = pd.Series(y_imb)

print(f"  Before SMOTE — class counts: {y_imb_s.value_counts().to_dict()}")


pipeline_imb = build_preprocessing_pipeline(X_imb_df, task='Classification')
X_imb_out, y_imb_out = pipeline_imb.fit_resample(X_imb_df, y_imb_s)

print(f"  After  SMOTE — class counts: {pd.Series(y_imb_out).value_counts().to_dict()}")

smote_pass = pd.Series(y_imb_out).value_counts().min() == pd.Series(y_imb_out).value_counts().max()
print(f"  SMOTE check : {'PASS — dataset is perfectly balanced' if smote_pass else 'FAIL'}")
print("=" * 60)