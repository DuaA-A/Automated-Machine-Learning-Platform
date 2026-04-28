import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn import metrics

from preprocessing.preprocessing import fit_preprocess


def train_clustering(X, algorithm_choice="AutoML (Find Best Model)"):
    """
    Train clustering models.
    Preprocessing is fit on the full X (no y leakage concern for unsupervised tasks).
    """
    # ── Preprocess all data ──
    feature_pipeline, X_processed, _, prep_info = fit_preprocess(
        X, None, task="Clustering"
    )

    # ── Professionally Enhance Model: Outlier Removal ──
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=0.05, random_state=42)
    outlier_labels = iso.fit_predict(X_processed)
    X_clean = X_processed[outlier_labels == 1].copy()

    # ── PCA for Dimensionality Reduction ──
    # If many features, use PCA to keep 95% variance (reduces noise for distances)
    n_features = X_clean.shape[1]
    if n_features > 10:
        pca_reducer = PCA(n_components=0.95, random_state=42)
        X_for_clustering = pca_reducer.fit_transform(X_clean)
    else:
        X_for_clustering = X_clean.values if hasattr(X_clean, 'values') else X_clean

    # ── Silhouette Sampling Utility ──
    def get_sampled_silhouette(data, labels):
        if len(np.unique(labels)) <= 1:
            return -1
        sample_size = min(3000, len(data))
        if len(data) > sample_size:
            indices = np.random.choice(len(data), sample_size, replace=False)
            return metrics.silhouette_score(data[indices], labels[indices])
        return metrics.silhouette_score(data, labels)

    # ── Professionally Find Optimal K (using K-Means & Silhouette) ──
    max_k = min(15, max(3, len(X_for_clustering) - 1))
    best_k = 3
    best_k_score = -1

    if len(X_for_clustering) > 3:
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10) # Higher n_init for quality
            labels = km.fit_predict(X_for_clustering)
            score = get_sampled_silhouette(X_for_clustering, labels)
            if score > best_k_score:
                best_k_score = score
                best_k = k

    prep_info["optimal_k_selected"] = best_k

    # ── Algorithm Selection ──
    algos = {
        "K-Means": KMeans(n_clusters=best_k, random_state=42, n_init=10),
        "GMM (Gaussian Mixture)": GaussianMixture(n_components=best_k, random_state=42)
    }
    
    # Only add Agglomerative if data isn't too large (O(N^3) bottleneck)
    if len(X_for_clustering) < 5000:
        algos["Agglomerative"] = AgglomerativeClustering(n_clusters=best_k)

    if algorithm_choice != "AutoML (Find Best Model)" and algorithm_choice in algos:
        algos = {algorithm_choice: algos[algorithm_choice]}

    best_model = None
    best_score = -np.inf
    results = {}

    for name, algo in algos.items():
        if hasattr(algo, 'fit_predict'):
            labels = algo.fit_predict(X_for_clustering)
        else:
            # For GMM
            algo.fit(X_for_clustering)
            labels = algo.predict(X_for_clustering)
            
        score = get_sampled_silhouette(X_for_clustering, labels)

        if score > best_score:
            best_score = score
            best_model = algo
            
            unique, counts = np.unique(labels, return_counts=True)
            cluster_info = {str(u): int(c) for u, c in zip(unique, counts)}
            
            centroids = None
            if hasattr(algo, 'cluster_centers_'):
                centroids = algo.cluster_centers_.tolist()
            elif hasattr(algo, 'means_'):
                centroids = algo.means_.tolist()

            # ── PCA for 2D Visualization ──
            pca_vis = PCA(n_components=2)
            X_pca = pca_vis.fit_transform(X_clean) # Vis on the clean data, not PCA-reduced
            
            sample_size = min(500, len(X_pca))
            indices = np.random.choice(len(X_pca), sample_size, replace=False)

            if hasattr(X_clean, 'columns'):
                feature_names = X_clean.columns.tolist()
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
