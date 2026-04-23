# preprocessing/clustering.py
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import metrics

from preprocessing.preprocessing import fit_preprocess

logger = logging.getLogger(__name__)


def train_clustering(X, algorithm_choice="AutoML (Find Best Model)"):
    """
    Train KMeans AND AgglomerativeClustering on any dataset.

    Steps:
      1. Preprocess X using fit_preprocess (impute, scale, encode)
      2. Remove outliers with IsolationForest
      3. Find optimal K by scanning K=2..15 with Silhouette Score on KMeans
      4. Train BOTH algorithms with optimal K
      5. Evaluate each with Silhouette Score
      6. Pick the best one
      7. Build PCA 2D visualization of best model's labels

    Returns: best_model, results_dict, prep_info
    """

    # ── STEP 1: Preprocess ───────────────────────────────────────
    logger.info("Clustering: Starting fit_preprocess...")
    feature_pipeline, X_processed, _, prep_info = fit_preprocess(
        X, None, task="Clustering"
    )

    # Convert to numpy array so both KMeans and Agglomerative work
    if hasattr(X_processed, 'values'):
        X_arr = X_processed.values
    else:
        X_arr = np.array(X_processed)

    logger.info(f"Clustering: Preprocessing done. Shape: {X_arr.shape}")

    # ── STEP 2: Remove outliers ──────────────────────────────────
    iso            = IsolationForest(contamination=0.05, random_state=42)
    outlier_mask   = iso.fit_predict(X_arr) == 1   # True = normal point
    X_clean        = X_arr[outlier_mask]

    n_removed = X_arr.shape[0] - X_clean.shape[0]
    logger.info(f"Clustering: Removed {n_removed} outliers. Clean shape: {X_clean.shape}")
    prep_info["outliers_removed"] = int(n_removed)

    # ── STEP 3: Find optimal K ───────────────────────────────────
    max_k         = min(15, max(3, len(X_clean) - 1))
    best_k        = 3
    best_k_score  = -1
    k_search      = {}   # {k: silhouette_score}

    if len(X_clean) > 4:
        logger.info(f"Clustering: Searching for optimal K from 2 to {max_k}...")
        for k in range(2, max_k + 1):
            km     = KMeans(n_clusters=k, random_state=42, n_init='auto')
            lbs    = km.fit_predict(X_clean)
            if len(np.unique(lbs)) > 1:
                sil = float(metrics.silhouette_score(X_clean, lbs))
                k_search[k] = round(sil, 4)
                logger.info(f"  K={k}: silhouette={sil:.4f}")
                if sil > best_k_score:
                    best_k_score = sil
                    best_k = k

    logger.info(f"Clustering: Optimal K selected = {best_k} (score={best_k_score:.4f})")
    prep_info["optimal_k_selected"] = int(best_k)
    prep_info["k_search_scores"]    = k_search

    # ── STEP 4 & 5: Train both algorithms, evaluate each ─────────
    algos = {
        "K-Means": KMeans(
            n_clusters  = best_k,
            random_state = 42,
            n_init      = 'auto'
        ),
        "Agglomerative Clustering": AgglomerativeClustering(
            n_clusters = best_k,
            linkage    = 'ward'   # ward: minimises within-cluster variance
                                  # alternatives: 'complete', 'average', 'single'
        )
    }

    # If user picked a specific algorithm, run only that one
    if algorithm_choice not in ("AutoML (Find Best Model)", None):
        # Match partial names e.g. "Agglomerative" matches "Agglomerative Clustering"
        matched = {k: v for k, v in algos.items()
                   if algorithm_choice.lower() in k.lower()}
        if matched:
            algos = matched

    comparison  = []   # side-by-side scores for both algorithms
    best_model  = None
    best_score  = -np.inf
    best_result = {}

    for name, algo in algos.items():
        logger.info(f"Clustering: Training {name}...")

        labels = algo.fit_predict(X_clean)

        # Silhouette Score: measures how well-separated clusters are
        # Range: [-1, 1] — higher is better
        sil_score = float(metrics.silhouette_score(X_clean, labels))
        logger.info(f"Clustering: {name} — Silhouette Score = {sil_score:.4f}")

        # Cluster size distribution
        unique, counts  = np.unique(labels, return_counts=True)
        cluster_sizes   = {f"Cluster {int(u)}": int(c)
                           for u, c in zip(unique, counts)}

        comparison.append({
            "algorithm":        name,
            "n_clusters":       int(best_k),
            "silhouette_score": round(sil_score, 4),
            "cluster_sizes":    cluster_sizes,
            "linkage":          "ward" if "Agglomerative" in name else "N/A"
        })

        # ── STEP 6: Track best ───────────────────────────────
        if sil_score > best_score:
            best_score = sil_score
            best_model = algo

            # Centroids — only KMeans stores cluster_centers_
            # Agglomerative does not have centroids, so we compute them manually
            if hasattr(algo, 'cluster_centers_'):
                centroids = algo.cluster_centers_.tolist()
            else:
                # Compute centroid of each cluster manually for Agglomerative
                centroids = []
                for cluster_id in sorted(unique):
                    mask      = labels == cluster_id
                    centroid  = X_clean[mask].mean(axis=0).tolist()
                    centroids.append(centroid)

            # ── STEP 7: PCA 2D for visualization ─────────────
            pca   = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_clean)

            sample_size = min(500, len(X_pca))
            indices     = np.random.choice(len(X_pca), sample_size, replace=False)

            # Feature names from pipeline output
            if hasattr(X_processed, 'columns'):
                feature_names = X_processed.columns.tolist()
            elif hasattr(feature_pipeline, 'get_feature_names_out'):
                try:
                    feature_names = feature_pipeline.get_feature_names_out().tolist()
                except Exception:
                    feature_names = [f"Feature {i}" for i in range(X_clean.shape[1])]
            else:
                feature_names = [f"Feature {i}" for i in range(X_clean.shape[1])]

            best_result = {
                "algorithm":        name,
                "optimal_k":        int(best_k),
                "silhouette_score": sil_score,
                "cluster_sizes":    cluster_sizes,
                "centroids":        centroids,
                "feature_names":    feature_names,
                "k_search_scores":  k_search,
                "comparison":       comparison,
                "pca_data": {
                    "x":      X_pca[indices, 0].tolist(),
                    "y":      X_pca[indices, 1].tolist(),
                    "labels": labels[indices].tolist()
                }
            }

    logger.info(f"Clustering: Best model = {best_result.get('algorithm')} "
                f"(Silhouette = {best_score:.4f})")

    return best_model, best_result, prep_info