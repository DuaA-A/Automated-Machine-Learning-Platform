import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def train_regression(X_processed, y_processed):
    best_model = None
    best_score = np.inf
    results = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42
    )
    
    algos = {
        "Random Forest":     RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression()
    }
    
    for name, algo in algos.items():
        algo.fit(X_train, y_train)
        y_pred = algo.predict(X_test)
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
            
    return best_model, results
