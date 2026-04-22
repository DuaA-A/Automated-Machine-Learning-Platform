import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from preprocessing.preprocessing import fit_preprocess, transform_test


def train_classification(X, y):
    """
    Train classification models with no data leakage:
    1. Split into train/test FIRST
    2. Fit preprocessing on X_train only
    3. Transform X_test using the already-fitted pipeline
    4. Train and evaluate models
    """
    # ── 1. Split FIRST (before any preprocessing) ─────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 2. Fit preprocessing on training data ONLY ────────────────
    feature_pipeline, X_train_proc, y_train_proc, prep_info = fit_preprocess(
        X_train, y_train, task="Classification"
    )

    # ── 3. Transform test data (no SMOTE, no fitting) ─────────────
    X_test_proc = transform_test(feature_pipeline, X_test)

    # ── 4. Train and evaluate models ──────────────────────────────
    best_model = None
    best_score = -np.inf
    results = {}

    algos = {
        "Random Forest":          RandomForestClassifier(random_state=42),
        "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(random_state=42),
        "Gradient Boosting":      GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors":    KNeighborsClassifier()
    }

    for name, algo in algos.items():
        algo.fit(X_train_proc, y_train_proc)
        y_pred = algo.predict(X_test_proc)
        score  = metrics.f1_score(y_test, y_pred, average='weighted')

        if score > best_score:
            best_score = score
            best_model = algo
            results = {
                "algorithm":        name,
                "accuracy":         float(metrics.accuracy_score(y_test, y_pred)),
                "precision":        float(metrics.precision_score(y_test, y_pred, average='weighted')),
                "recall":           float(metrics.recall_score(y_test, y_pred, average='weighted')),
                "f1_score":         float(score),
                "confusion_matrix": metrics.confusion_matrix(y_test, y_pred).tolist()
            }

    return best_model, results, prep_info
