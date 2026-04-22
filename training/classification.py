import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_predict,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics

from preprocessing.preprocessing import fit_preprocess, transform_test

# ── Threshold: rows × columns above this → use RandomizedSearchCV ──────────
_LARGE_DATA_THRESHOLD = 10_000


def _pick_search(algo, param_grid, cv, is_large: bool):
    """
    Return a fitted-ready search object.
    - Small dataset  → GridSearchCV   (exhaustive, precise)
    - Large dataset  → RandomizedSearchCV (n_iter=20, fast)
    """
    common = dict(estimator=algo, scoring="f1_weighted", cv=cv,
                  n_jobs=-1, refit=True, random_state=42)
    if is_large:
        return RandomizedSearchCV(param_distributions=param_grid,
                                  n_iter=20, **common)
    else:
        return GridSearchCV(param_grid=param_grid, **common)


def train_classification(X, y):
    """
    Train 2 classification models (Random Forest & Gradient Boosting) with
    automatic hyperparameter search strategy based on dataset size:

      • Small data  (rows × cols ≤ 10 000) → GridSearchCV   (exhaustive)
      • Large data  (rows × cols  > 10 000) → RandomizedSearchCV (n_iter=20)

    Pipeline:
    1. Split into train/test FIRST (no leakage)
    2. Fit preprocessing on X_train only
    3. Transform X_test with the fitted pipeline
    4. Tune & evaluate both models, return the best one
    """
    # ── 1. Split FIRST ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 2. Fit preprocessing on training data ONLY ────────────────
    feature_pipeline, X_train_proc, y_train_proc, prep_info = fit_preprocess(
        X_train, y_train, task="Classification"
    )

    # ── 3. Transform test data (no fitting) ───────────────────────
    X_test_proc = transform_test(feature_pipeline, X_test)

    # ── 4. Decide search strategy from dataset size ───────────────
    data_size = X_train_proc.shape[0] * X_train_proc.shape[1]
    is_large   = data_size > _LARGE_DATA_THRESHOLD
    strategy   = "RandomizedSearchCV (large data)" if is_large else "GridSearchCV (small data)"

    # ── 5. Cross-validation setup ─────────────────────────────────
    min_class_count = pd.Series(y_train_proc).value_counts().min()
    if min_class_count >= 2:
        n_splits = min(5, min_class_count)
        cv       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        use_cv   = True
    else:
        cv       = 3   # plain integer fallback
        use_cv   = False

    # ── 6. Define 2 models + their param spaces ───────────────────
    algos_and_params = {
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators":      [100, 200, 300],
                "max_depth":         [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf":  [1, 2, 4],
            }
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators":  [100, 200, 300],
                "max_depth":     [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample":     [0.8, 1.0],
            }
        ),
    }

    # ── 7. Tune, evaluate, pick best ──────────────────────────────
    best_model = None
    best_score = -np.inf
    results    = {}

    for name, (algo, param_space) in algos_and_params.items():
        search = _pick_search(algo, param_space, cv, is_large)
        search.fit(X_train_proc, y_train_proc)
        tuned_algo = search.best_estimator_

        if use_cv:
            cv_preds        = cross_val_predict(tuned_algo, X_train_proc,
                                                y_train_proc, cv=cv)
            eval_y_true     = y_train_proc
            eval_y_pred     = cv_preds
            algo_name_display = f"{name} ({n_splits}-Fold CV) [{strategy}]"
        else:
            eval_y_true     = y_test
            eval_y_pred     = tuned_algo.predict(X_test_proc)
            algo_name_display = f"{name} (Test Split) [{strategy}]"

        score = metrics.f1_score(eval_y_true, eval_y_pred, average="weighted")

        if score > best_score:
            best_score = score
            best_model = tuned_algo

            class_labels = [str(c) for c in np.unique(eval_y_true)]

            results = {
                "algorithm":        algo_name_display,
                "search_strategy":  strategy,
                "best_params":      search.best_params_,
                "accuracy":         float(metrics.accuracy_score(eval_y_true, eval_y_pred)),
                "precision":        float(metrics.precision_score(eval_y_true, eval_y_pred,
                                                                   average="weighted", zero_division=0)),
                "recall":           float(metrics.recall_score(eval_y_true, eval_y_pred,
                                                                average="weighted", zero_division=0)),
                "f1_score":         float(score),
                "class_labels":     class_labels,
                "confusion_matrix": metrics.confusion_matrix(eval_y_true, eval_y_pred).tolist(),
            }

    return best_model, results, prep_info
