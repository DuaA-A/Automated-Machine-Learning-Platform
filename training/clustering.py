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

    # ── Professionally Enhance Model: Outlier Removal ──
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=0.05, random_state=42)
    outlier_labels = iso.fit_predict(X_processed)
    
    # Keep only normal data points (label 1)
    X_clean = X_processed[outlier_labels == 1].copy()
    
    # ── Professionally Find Optimal K (using K-Means & Silhouette) ──
    # Search up to 15 clusters for more granular grouping
    max_k = min(15, max(3, len(X_clean) - 1))
    best_k = 3
    best_k_score = -1

    # Only do the search if we actually have enough data to form clusters
    if len(X_clean) > 3:
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = km.fit_predict(X_clean)
            # Silhouette requires at least 2 clusters and less than n_samples
            if len(np.unique(labels)) > 1:
                score = metrics.silhouette_score(X_clean, labels)
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
        labels = algo.fit_predict(X_clean)
        score  = metrics.silhouette_score(X_clean, labels)

        if score > best_score:
            best_score = score
            best_model = algo
            
            # Extract cluster info
            unique, counts = np.unique(labels, return_counts=True)
            cluster_info = {str(u): int(c) for u, c in zip(unique, counts)}
            
            centroids = None
            if hasattr(algo, 'cluster_centers_'):
                centroids = algo.cluster_centers_.tolist()

            # ── PCA for 2D Visualization ──
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_clean)
            
            sample_size = min(500, len(X_pca))
            indices = np.random.choice(len(X_pca), sample_size, replace=False)

            # Get feature names safely
            if hasattr(X_clean, 'columns'):
                feature_names = X_clean.columns.tolist()
            elif hasattr(feature_pipeline, 'get_feature_names_out'):
                feature_names = feature_pipeline.get_feature_names_out().tolist()
            else:
                feature_names = [f"Feature {i}" for i in range(X_clean.shape[1])]

            results = {
                "algorithm":        name,
                "optimal_k":        best_k,
                "silhouette_score": float(score),
                "cluster_sizes":    cluster_info,
                "centroids":        centroids,
                "feature_names":    feature_names,
                "pca_data": {
                    "x": X_pca[indices, 0].tolist(),
                    "y": X_pca[indices, 1].tolist(),
                    "labels": labels[indices].tolist()
                }
            }

    return best_model, results, prep_info
