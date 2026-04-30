import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from app.services.plot_service import fig_to_base64, PALETTE


def _preprocess(df, feature_cols):
    data = df[feature_cols].dropna()
    if len(data) < 4:
        raise ValueError("Données insuffisantes.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.values)
    return X_scaled, data


def _reduce_for_plot(X_scaled):
    """Réduit à 2D pour la visualisation si nécessaire."""
    if X_scaled.shape[1] == 2:
        return X_scaled
    pca = PCA(n_components=2)
    return pca.fit_transform(X_scaled)


def kmeans_clustering(df, feature_cols, n_clusters=3, max_k=10):
    """K-Means clustering avec méthode du coude et score de silhouette."""
    X_scaled, data = _preprocess(df, feature_cols)
    max_k = min(max_k, len(X_scaled) - 1, 15)

    # Méthode du coude
    inertias, sil_scores, k_range = [], [], range(2, max_k + 1)
    for k in k_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = km_temp.fit_predict(X_scaled)
        inertias.append(km_temp.inertia_)
        if len(np.unique(labels_temp)) > 1:
            sil_scores.append(silhouette_score(X_scaled, labels_temp))
        else:
            sil_scores.append(-1)

    # Clustering final
    n_clusters = max(2, min(n_clusters, len(X_scaled) - 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
    db_score = davies_bouldin_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)

    X_2d = _reduce_for_plot(X_scaled)
    centers_2d = _reduce_for_plot(kmeans.cluster_centers_) if X_scaled.shape[1] > 2 else kmeans.cluster_centers_

    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Méthode du coude
    axes[0].plot(list(k_range), inertias, "o-", color=PALETTE["blue"], linewidth=2, label="Inertie (SSE)")
    axes[0].axvline(n_clusters, color=PALETTE["orange"], linestyle="--",
                    linewidth=1.5, label=f"k={n_clusters} sélectionné")
    axes[0].set_xlabel("Nombre de clusters k", fontsize=10)
    axes[0].set_ylabel("Inertie", fontsize=10)
    axes[0].set_title("Méthode du coude", fontsize=11)
    axes[0].legend(fontsize=9)

    # 2. Score silhouette par k
    axes[1].plot(list(k_range), sil_scores, "s-", color=PALETTE["green"], linewidth=2)
    axes[1].axvline(n_clusters, color=PALETTE["orange"], linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Nombre de clusters k", fontsize=10)
    axes[1].set_ylabel("Score de silhouette", fontsize=10)
    axes[1].set_title("Score de silhouette par k", fontsize=11)

    # 3. Visualisation clusters
    for c in range(n_clusters):
        mask = labels == c
        axes[2].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=[colors[c]], label=f"Cluster {c+1}",
                        alpha=0.7, s=50, zorder=2)
    axes[2].scatter(centers_2d[:, 0], centers_2d[:, 1],
                    marker="X", s=250, c="black", label="Centroïdes", zorder=5)
    axes[2].set_xlabel("Axe 1 (PCA)" if X_scaled.shape[1] > 2 else feature_cols[0], fontsize=10)
    axes[2].set_ylabel("Axe 2 (PCA)" if X_scaled.shape[1] > 2 else feature_cols[1], fontsize=10)
    axes[2].set_title(f"K-Means (k={n_clusters})\nSilhouette = {sil:.4f}", fontsize=11)
    axes[2].legend(fontsize=9)

    plt.suptitle("Clustering K-Means", fontsize=12, y=1.01)
    plt.tight_layout()
    plot_b64 = fig_to_base64(fig)

    cluster_sizes = {f"Cluster {i+1}": int(np.sum(labels == i))
                     for i in range(n_clusters)}
    cluster_pcts = {k: round(v / len(labels) * 100, 1)
                    for k, v in cluster_sizes.items()}

    return {
        "method": "K-Means",
        "n_clusters": n_clusters,
        "n_features": len(feature_cols),
        "n_samples": len(X_scaled),
        "inertia": round(float(kmeans.inertia_), 2),
        "silhouette_score": round(float(sil), 4),
        "davies_bouldin": round(float(db_score), 4),
        "calinski_harabasz": round(float(ch_score), 2),
        "cluster_sizes": cluster_sizes,
        "cluster_percentages": cluster_pcts,
        "optimal_k_silhouette": int(list(k_range)[np.argmax(sil_scores)]),
        "plot": plot_b64,
    }


def dbscan_clustering(df, feature_cols, eps=0.5, min_samples=5):
    """DBSCAN : clustering basé sur la densité."""
    X_scaled, data = _preprocess(df, feature_cols)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    sil = silhouette_score(X_scaled, labels) if n_clusters > 1 and len(np.unique(labels)) > 1 else None

    X_2d = _reduce_for_plot(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 6))

    if n_clusters > 0:
        unique_labels = sorted(set(labels) - {-1})
        colors = plt.cm.Set2(np.linspace(0, 1, max(n_clusters, 1)))
        for c_idx, c in enumerate(unique_labels):
            mask = labels == c
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[colors[c_idx]], label=f"Cluster {c+1}",
                       alpha=0.7, s=50, zorder=2)

    noise_mask = labels == -1
    if np.any(noise_mask):
        ax.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1],
                   c=PALETTE["gray"], label=f"Bruit ({n_noise})", alpha=0.4,
                   s=20, marker="x", zorder=1)

    ax.set_xlabel("Axe 1 (PCA)" if X_scaled.shape[1] > 2 else feature_cols[0], fontsize=10)
    ax.set_ylabel("Axe 2 (PCA)" if X_scaled.shape[1] > 2 else feature_cols[1], fontsize=10)
    sil_txt = f"Silhouette = {sil:.4f}" if sil else "1 cluster ou bruit uniquement"
    ax.set_title(f"DBSCAN — eps={eps}, min_samples={min_samples}\n"
                 f"{n_clusters} cluster(s), {n_noise} bruit(s), {sil_txt}", fontsize=11)
    if n_clusters > 0:
        ax.legend(fontsize=9)
    plt.tight_layout()
    plot_b64 = fig_to_base64(fig)

    cluster_sizes = {}
    for c in sorted(set(labels) - {-1}):
        cluster_sizes[f"Cluster {c+1}"] = int(np.sum(labels == c))

    return {
        "method": "DBSCAN",
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_percent": round(n_noise / len(labels) * 100, 1),
        "n_samples": len(X_scaled),
        "silhouette_score": round(float(sil), 4) if sil else "N/A",
        "cluster_sizes": cluster_sizes,
        "note": "DBSCAN détecte les clusters de forme arbitraire et identifie les points aberrants.",
        "plot": plot_b64,
    }


def hierarchical_clustering(df, feature_cols, n_clusters=3, linkage_method="ward"):
    """Classification Ascendante Hiérarchique (CAH) avec dendrogramme."""
    X_scaled, data = _preprocess(df, feature_cols)

    # Limite pour le dendrogramme
    X_dendro = X_scaled[:min(200, len(X_scaled))]

    # CAH
    n_clusters = max(2, min(n_clusters, len(X_scaled) - 1))
    cah = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = cah.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0

    X_2d = _reduce_for_plot(X_scaled)
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Dendrogramme
    Z = linkage(X_dendro, method=linkage_method)
    dendrogram(Z, ax=axes[0], no_labels=True,
               color_threshold=Z[-n_clusters+1, 2] if n_clusters > 1 else 0,
               above_threshold_color=PALETTE["gray"])
    axes[0].axhline(Z[-n_clusters+1, 2] if n_clusters > 1 else 0,
                    color=PALETTE["orange"], linestyle="--", linewidth=1.5,
                    label=f"Coupe pour {n_clusters} clusters")
    axes[0].set_xlabel("Observations", fontsize=10)
    axes[0].set_ylabel("Distance", fontsize=10)
    axes[0].set_title(f"Dendrogramme (méthode : {linkage_method})", fontsize=11)
    axes[0].legend(fontsize=9)

    # 2. Visualisation
    for c in range(n_clusters):
        mask = labels == c
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=[colors[c]], label=f"Cluster {c+1}",
                        alpha=0.7, s=50)
    axes[1].set_xlabel("Axe 1 (PCA)" if X_scaled.shape[1] > 2 else feature_cols[0], fontsize=10)
    axes[1].set_ylabel("Axe 2 (PCA)" if X_scaled.shape[1] > 2 else feature_cols[1], fontsize=10)
    axes[1].set_title(f"CAH — {n_clusters} clusters\nSilhouette = {sil:.4f}", fontsize=11)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plot_b64 = fig_to_base64(fig)

    cluster_sizes = {f"Cluster {i+1}": int(np.sum(labels == i))
                     for i in range(n_clusters)}

    return {
        "method": "CAH",
        "linkage": linkage_method,
        "n_clusters": n_clusters,
        "n_samples": len(X_scaled),
        "silhouette_score": round(float(sil), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X_scaled, labels)), 4),
        "cluster_sizes": cluster_sizes,
        "plot": plot_b64,
    }
