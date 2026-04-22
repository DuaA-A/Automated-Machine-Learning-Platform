import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics

from preprocessing.preprocessing import fit_preprocess


def train_clustering(X):
    """
    Train clustering models.
    Preprocessing is fit on the full X (no y leakage concern for unsupervised tasks).
    """
    # ── Preprocess all data (no train/test split needed for unsupervised) ──
    feature_pipeline, X_processed, _, prep_info = fit_preprocess(
        X, None, task="Clustering"
    )

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

    return best_model, results, prep_info
