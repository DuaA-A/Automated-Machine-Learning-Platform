import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics

from preprocessing.preprocessing import fit_preprocess, transform_test

_LARGE_DATA_THRESHOLD = 10_000


def _pick_search(algo, param_grid, cv, is_large: bool):
    common = dict(estimator=algo, scoring="f1_weighted", cv=cv,
                  n_jobs=-1, refit=True)

    if is_large:
        return RandomizedSearchCV(
            param_distributions=param_grid,
            n_iter=20,
            random_state=42,
            **common
        )
    else:
        return GridSearchCV(
            param_grid=param_grid,
            **common
        )


def train_classification(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    feature_pipeline, X_train_proc, y_train_proc, prep_info = fit_preprocess(
        X_train, y_train, task="Classification"
    )
    X_test_proc = transform_test(feature_pipeline, X_test)

    data_size = X_train_proc.shape[0] * X_train_proc.shape[1]
    is_large  = data_size > _LARGE_DATA_THRESHOLD
    strategy  = "RandomizedSearchCV (large data)" if is_large else "GridSearchCV (small data)"

    min_class_count = pd.Series(y_train_proc).value_counts().min()

    if min_class_count >= 2:
        n_splits = max(2, min(5, min_class_count))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        n_splits = None
        cv = 3

    algos_and_params = {
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
            }
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            }
        ),
    }

    best_model = None
    best_score = -np.inf
    best_result = None
    for name, (algo, param_space) in algos_and_params.items():

        search = _pick_search(algo, param_space, cv, is_large)
        search.fit(X_train_proc, y_train_proc)

        tuned_model = search.best_estimator_

        y_pred = tuned_model.predict(X_test_proc)

        score = metrics.f1_score(y_test, y_pred, average="weighted")

        class_labels = [str(c) for c in np.unique(y_test)]

        result = {
            "algorithm": f"{name} [{strategy}]",
            "search_strategy": strategy,
            "best_params": search.best_params_,
            "accuracy": float(metrics.accuracy_score(y_test, y_pred)),
            "precision": float(metrics.precision_score(
                y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(metrics.recall_score(
                y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(score),
            "class_labels": class_labels,
            "confusion_matrix": metrics.confusion_matrix(y_test, y_pred).tolist(),
        }

        if score > best_score:
            best_score = score
            best_model = tuned_model
            best_result = result
    return best_model, best_result, prep_info
