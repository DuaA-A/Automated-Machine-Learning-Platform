import numpy as np
import logging

logger = logging.getLogger(__name__)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn import metrics

from preprocessing.preprocessing import fit_preprocess, transform_test


def train_regression(X, y):
    logger.info("Training (Regression): Starting train_regression...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Training (Regression): Data split into train {X_train.shape} and test {X_test.shape}")

    logger.info("Training (Regression): Fitting preprocessing pipeline...")
    feature_pipeline, X_train_proc, y_train_proc, prep_info = fit_preprocess(
        X_train, y_train, task="Regression"
    )

    logger.info("Training (Regression): Transforming test data...")
    X_test_proc = transform_test(feature_pipeline, X_test)

    best_model = None
    best_score = np.inf
    results = {}

    algos = {
        "Random Forest":     RandomForestRegressor(random_state=42),
        "Ridge Regression":  Ridge(random_state=42)
    }

    param_grids = {
        "Random Forest": {
            'n_estimators': [50, 100], 
            'max_depth': [None, 10],    
            'min_samples_split': [2, 5], 
            'min_samples_leaf': [1, 2]   
        },
        "Ridge Regression": {
            'alpha': [0.1, 1.0, 10.0] 

        }
    }



    for name, algo in algos.items():
        logger.info(f"Training (Regression): Starting RandomizedSearchCV for {name}...")
        search = RandomizedSearchCV(
            estimator=algo,
            param_distributions=param_grids[name],
            n_iter=10,
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train_proc, y_train_proc)
        
        logger.info(f"Training (Regression): Search for {name} completed. Best params: {search.best_params_}")
        best_estimator = search.best_estimator_
        y_pred = best_estimator.predict(X_test_proc)
        score  = metrics.mean_squared_error(y_test, y_pred)
        logger.info(f"Training (Regression): Evaluation for {name} - MSE: {score:.4f}")

        if score < best_score:
            best_score = score
            best_model = best_estimator
            results = {
                "algorithm": name,
                "best_params": search.best_params_,
                "mae":       float(metrics.mean_absolute_error(y_test, y_pred)),
                "mse":       float(score),
                "r2_score":  float(metrics.r2_score(y_test, y_pred))
            }

    return best_model, results, prep_info
