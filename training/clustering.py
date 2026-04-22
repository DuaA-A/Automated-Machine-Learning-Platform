import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics

def train_clustering(X_processed):
    best_model = None
    best_score = -np.inf
    results = {}

    algos = {
        "K-Means":       KMeans(n_clusters=3, random_state=42),
        "Agglomerative": AgglomerativeClustering(n_clusters=3)
    }
    
    for name, algo in algos.items():
        labels = algo.fit_predict(X_processed)
        score  = metrics.silhouette_score(X_processed, labels)
        
        if score > best_score:
            best_score = score
            best_model = algo
            results = {
                "algorithm":        name,
                "silhouette_score": float(score)
            }
            
    return best_model, results
