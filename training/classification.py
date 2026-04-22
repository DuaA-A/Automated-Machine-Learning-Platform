import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def train_classification(X_processed, y_processed):
    best_model = None
    best_score = -np.inf
    results = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42
    )
    
    algos = {
        "Random Forest":       RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }
    
    for name, algo in algos.items():
        algo.fit(X_train, y_train)
        y_pred = algo.predict(X_test)
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
            
    return best_model, results
