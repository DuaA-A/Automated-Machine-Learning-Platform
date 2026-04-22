import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from preprocessing.preprocessing import fit_preprocess, transform_test


def train_regression(X, y, algorithm_choice="AutoML (Find Best Model)"):
    """
    Train regression models with no data leakage:
    1. Split into train/test FIRST
    2. Fit preprocessing on X_train only
    3. Transform X_test using the already-fitted pipeline
    4. Train and evaluate models
    """
    # ── 1. Split FIRST (before any preprocessing) ─────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 2. Fit preprocessing on training data ONLY ────────────────
    feature_pipeline, X_train_proc, y_train_proc, prep_info = fit_preprocess(
        X_train, y_train, task="Regression"
    )

    # ── 3. Transform test data (no fitting) ───────────────────────
    X_test_proc = transform_test(feature_pipeline, X_test)

    # ── 4. Train and evaluate models ──────────────────────────────
    best_model = None
    best_score = np.inf
    results = {}

    algos = {
        "Random Forest":     RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression()
    }

    if algorithm_choice != "AutoML (Find Best Model)" and algorithm_choice in algos:
        algos = {algorithm_choice: algos[algorithm_choice]}

    for name, algo in algos.items():
        algo.fit(X_train_proc, y_train_proc)
        y_pred = algo.predict(X_test_proc)
        score  = metrics.mean_squared_error(y_test, y_pred)

        if score < best_score:
            best_score = score
            best_model = algo
            results = {
                "algorithm": name,
                "mae":       float(metrics.mean_absolute_error(y_test, y_pred)),
                "mse":       float(score),
                "r2_score":  float(metrics.r2_score(y_test, y_pred))
            }

    return best_model, results, prep_info
