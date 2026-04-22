import pandas as pd
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
        "Random Forest":          RandomForestClassifier(random_state=42, max_depth=10, min_samples_split=5),
        "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(random_state=42),
        "Gradient Boosting":      GradientBoostingClassifier(random_state=42, max_depth=5),
        "K-Nearest Neighbors":    KNeighborsClassifier()
    }

    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    # Determine if we can use StratifiedKFold (needs at least 2 members per class)
    # If the user passes a continuous variable to classification, this will often fail.
    min_class_count = pd.Series(y_train_proc).value_counts().min()
    
    if min_class_count >= 2:
        n_splits = min(5, min_class_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        use_cv = True
    else:
        use_cv = False

    for name, algo in algos.items():
        algo.fit(X_train_proc, y_train_proc)
        
        if use_cv:
            # Evaluate robustly using Cross-Validation
            cv_preds = cross_val_predict(algo, X_train_proc, y_train_proc, cv=cv)
            eval_y_true = y_train_proc
            eval_y_pred = cv_preds
            algo_name_display = f"{name} ({n_splits}-Fold CV)"
        else:
            # Fallback to test set evaluation if CV is impossible
            eval_y_true = y_test
            eval_y_pred = algo.predict(X_test_proc)
            algo_name_display = f"{name} (Test Split)"
            
        score  = metrics.f1_score(eval_y_true, eval_y_pred, average='weighted')

        if score > best_score:
            best_score = score
            best_model = algo
            
            class_labels = [str(c) for c in np.unique(eval_y_true)]
            
            results = {
                "algorithm":        algo_name_display,
                "accuracy":         float(metrics.accuracy_score(eval_y_true, eval_y_pred)),
                "precision":        float(metrics.precision_score(eval_y_true, eval_y_pred, average='weighted', zero_division=0)),
                "recall":           float(metrics.recall_score(eval_y_true, eval_y_pred, average='weighted', zero_division=0)),
                "f1_score":         float(score),
                "class_labels":     class_labels,
                "confusion_matrix": metrics.confusion_matrix(eval_y_true, eval_y_pred).tolist()
            }

    return best_model, results, prep_info
