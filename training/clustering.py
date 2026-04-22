import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics

from preprocessing.preprocessing import fit_preprocess


def train_clustering(X, algorithm_choice="AutoML (Find Best Model)"):
    """
    Train clustering models.
    Preprocessing is fit on the full X (no y leakage concern for unsupervised tasks).
    """
    # ── Preprocess all data (no train/test split needed for unsupervised) ──
    feature_pipeline, X_processed, _, prep_info = fit_preprocess(
        X, None, task="Clustering"
    )

    # ── Professionally Find Optimal K (using K-Means & Silhouette) ──
    max_k = min(10, max(3, len(X_processed) - 1))
    best_k = 3
    best_k_score = -1

    # Only do the search if we actually have enough data to form clusters
    if len(X_processed) > 3:
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = km.fit_predict(X_processed)
            # Silhouette requires at least 2 clusters and less than n_samples
            if len(np.unique(labels)) > 1:
                score = metrics.silhouette_score(X_processed, labels)
                if score > best_k_score:
                    best_k_score = score
                    best_k = k

    # Update prep_info to tell the user what K was chosen
    prep_info["optimal_k_selected"] = best_k

    best_model = None
    best_score = -np.inf
    results = {}

    algos = {
        "K-Means":       KMeans(n_clusters=best_k, random_state=42, n_init='auto'),
        "Agglomerative": AgglomerativeClustering(n_clusters=best_k)
    }

    if algorithm_choice != "AutoML (Find Best Model)" and algorithm_choice in algos:
        algos = {algorithm_choice: algos[algorithm_choice]}

    for name, algo in algos.items():
        labels = algo.fit_predict(X_processed)
        score  = metrics.silhouette_score(X_processed, labels)

        if score > best_score:
            best_score = score
            best_model = algo
            
            # Extract cluster info
            unique, counts = np.unique(labels, return_counts=True)
            cluster_info = {str(u): int(c) for u, c in zip(unique, counts)}
            
            centroids = None
            if hasattr(algo, 'cluster_centers_'):
                # Get centroids and map back to feature names
                centroids = algo.cluster_centers_.tolist()

            results = {
                "algorithm":        name,
                "optimal_k":        best_k,
                "silhouette_score": float(score),
                "cluster_sizes":    cluster_info,
                "centroids":        centroids,
                "feature_names":    X_processed.columns.tolist()
            }

    return best_model, results, prep_info
