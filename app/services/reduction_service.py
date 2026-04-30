import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from app.services.plot_service import fig_to_base64, PALETTE


def _preprocess(df, feature_cols):
    data = df[feature_cols].dropna()
    if len(data) < 5:
        raise ValueError("Données insuffisantes (min. 5 lignes).")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.values)
    return X_scaled, data


def apply_pca(df, feature_cols, n_components=None):
    """Analyse en Composantes Principales."""
    X_scaled, data = _preprocess(df, feature_cols)
    max_comp = min(len(feature_cols), len(data))
    if n_components is None or n_components > max_comp:
        n_components = max_comp

    pca_full = PCA()
    pca_full.fit(X_scaled)

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)

    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Projection 2D
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100 if n_components >= 2 else 0
    axes[0].scatter(X_reduced[:, 0],
                    X_reduced[:, 1] if n_components >= 2 else np.zeros(len(X_reduced)),
                    color=PALETTE["blue"], alpha=0.6, s=40)
    axes[0].set_xlabel(f"PC1 ({pc1_var:.1f}% variance)", fontsize=10)
    axes[0].set_ylabel(f"PC2 ({pc2_var:.1f}% variance)", fontsize=10)
    axes[0].set_title("Projection PCA (2D)", fontsize=11)

    # 2. Scree plot
    k = min(len(explained), 10)
    x_range = range(1, k + 1)
    axes[1].bar(x_range, explained[:k] * 100, color=PALETTE["blue"], alpha=0.7, label="Par composante")
    ax2 = axes[1].twinx()
    ax2.plot(list(x_range), cumulative[:k] * 100, "o-",
             color=PALETTE["orange"], linewidth=2, label="Cumulée")
    ax2.axhline(80, color=PALETTE["green"], linestyle="--", alpha=0.7, label="Seuil 80%")
    ax2.set_ylabel("Variance cumulée (%)", fontsize=9)
    ax2.set_ylim(0, 105)
    axes[1].set_xlabel("Composante principale", fontsize=10)
    axes[1].set_ylabel("Variance expliquée (%)", fontsize=10)
    axes[1].set_title("Éboulis des valeurs propres", fontsize=11)
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    # 3. Biplot des chargements (loadings)
    loadings = pca.components_.T
    n_vars = min(len(feature_cols), 8)
    for i, (feat, load) in enumerate(zip(feature_cols[:n_vars], loadings[:n_vars])):
        axes[2].arrow(0, 0, load[0], load[1] if n_components >= 2 else 0,
                      head_width=0.02, head_length=0.02,
                      fc=PALETTE["orange"], ec=PALETTE["orange"], linewidth=1.5)
        axes[2].text(load[0] * 1.12, (load[1] if n_components >= 2 else 0) * 1.12,
                     feat, fontsize=8, ha="center", va="center",
                     color=PALETTE["purple"])
    circle = plt.Circle((0, 0), 1, color=PALETTE["gray"], fill=False, linestyle="--", alpha=0.5)
    axes[2].add_patch(circle)
    axes[2].axhline(0, color=PALETTE["gray"], alpha=0.4, linewidth=0.8)
    axes[2].axvline(0, color=PALETTE["gray"], alpha=0.4, linewidth=0.8)
    axes[2].set_xlim(-1.3, 1.3)
    axes[2].set_ylim(-1.3, 1.3)
    axes[2].set_xlabel("PC1", fontsize=10)
    axes[2].set_ylabel("PC2", fontsize=10)
    axes[2].set_title("Cercle des corrélations", fontsize=11)
    axes[2].set_aspect("equal")

    plt.suptitle("Analyse en Composantes Principales (PCA)", fontsize=12, y=1.01)
    plt.tight_layout()
    plot_b64 = fig_to_base64(fig)

    # Composantes pour le tableau
    comp_table = []
    for i, comp in enumerate(pca.components_):
        comp_table.append({
            "name": f"PC{i+1}",
            "variance": round(float(pca.explained_variance_ratio_[i]) * 100, 2),
            "cumulative": round(float(np.cumsum(pca.explained_variance_ratio_)[i]) * 100, 2),
            "loadings": {f: round(float(l), 4) for f, l in zip(feature_cols, comp)}
        })

    # Composantes pour atteindre 80% et 95%
    threshold_80 = int(np.argmax(cumulative >= 0.80)) + 1 if any(cumulative >= 0.80) else len(cumulative)
    threshold_95 = int(np.argmax(cumulative >= 0.95)) + 1 if any(cumulative >= 0.95) else len(cumulative)

    return {
        "method": "PCA",
        "n_components": n_components,
        "n_features": len(feature_cols),
        "n_samples": len(data),
        "variance_ratio": [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_],
        "cumulative_variance": [round(float(v) * 100, 2) for v in cumulative[:n_components]],
        "total_variance_2d": round(float(sum(pca.explained_variance_ratio_[:2])) * 100, 2),
        "components_table": comp_table,
        "threshold_80": threshold_80,
        "threshold_95": threshold_95,
        "plot": plot_b64,
    }


def apply_tsne(df, feature_cols, n_components=2, perplexity=30, n_iter=1000):
    """t-SNE : réduction non-linéaire pour la visualisation."""
    X_scaled, data = _preprocess(df, feature_cols)
    if len(data) > 3000:
        X_scaled = X_scaled[:3000]

    perp = min(perplexity, len(X_scaled) - 1)

    tsne = TSNE(n_components=2, perplexity=perp, n_iter=n_iter,
                random_state=42, init="pca")
    X_embedded = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1],
                         c=np.arange(len(X_embedded)), cmap="viridis",
                         alpha=0.7, s=30)
    plt.colorbar(scatter, ax=ax, label="Index des observations")
    ax.set_xlabel("t-SNE Dim 1", fontsize=10)
    ax.set_ylabel("t-SNE Dim 2", fontsize=10)
    ax.set_title(f"Visualisation t-SNE 2D\n(perplexité={perp}, itérations={n_iter})", fontsize=11)
    plt.tight_layout()
    plot_b64 = fig_to_base64(fig)

    return {
        "method": "t-SNE",
        "n_features": len(feature_cols),
        "n_samples": len(X_scaled),
        "perplexity": perp,
        "n_iter": n_iter,
        "kl_divergence": round(float(tsne.kl_divergence_), 4),
        "note": "t-SNE est une méthode de visualisation. Les distances absolues n'ont pas de sens physique, seule la structure des clusters est interprétable.",
        "plot": plot_b64,
    }


def apply_lda(df, feature_cols, target_col, n_components=None):
    """Analyse Discriminante Linéaire (supervisée)."""
    cols = feature_cols + [target_col]
    data = df[cols].dropna()
    if len(data) < 10:
        raise ValueError("Données insuffisantes.")

    X = data[feature_cols].values
    y = data[target_col].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    n_classes = len(le.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_comp = min(n_classes - 1, len(feature_cols))
    if max_comp < 1:
        raise ValueError("LDA nécessite au moins 2 classes et 1 feature.")
    if n_components is None or n_components > max_comp:
        n_components = max_comp

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X_scaled, y_enc)

    # Couleurs par classe
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    if n_components >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for cls_idx, (cls_name, color) in enumerate(zip(le.classes_, colors)):
            mask = y_enc == cls_idx
            axes[0].scatter(X_lda[mask, 0], X_lda[mask, 1],
                            c=[color], label=str(cls_name), alpha=0.7, s=50)
        axes[0].set_xlabel(f"LD1 ({lda.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
        axes[0].set_ylabel(f"LD2 ({lda.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
        axes[0].set_title("Projection LDA 2D", fontsize=11)
        axes[0].legend(title=target_col, fontsize=9)

        # Variance expliquée
        ev = lda.explained_variance_ratio_
        axes[1].bar(range(1, len(ev)+1), ev * 100, color=PALETTE["purple"], alpha=0.8)
        axes[1].set_xlabel("Axe discriminant", fontsize=10)
        axes[1].set_ylabel("Variance expliquée (%)", fontsize=10)
        axes[1].set_title("Variance expliquée par axe LDA", fontsize=11)
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        for cls_idx, (cls_name, color) in enumerate(zip(le.classes_, colors)):
            mask = y_enc == cls_idx
            ax.hist(X_lda[mask, 0], bins=20, color=color,
                    alpha=0.6, label=str(cls_name))
        ax.set_xlabel("LD1", fontsize=10)
        ax.set_ylabel("Fréquence", fontsize=10)
        ax.set_title("Distribution LDA (1 axe)", fontsize=11)
        ax.legend(title=target_col)

    plt.tight_layout()
    plot_b64 = fig_to_base64(fig)

    return {
        "method": "LDA",
        "n_features": len(feature_cols),
        "n_classes": n_classes,
        "classes": list(le.classes_),
        "n_components": n_components,
        "n_samples": len(data),
        "explained_variance_ratio": [round(float(v)*100, 2)
                                      for v in lda.explained_variance_ratio_],
        "total_variance": round(float(sum(lda.explained_variance_ratio_)) * 100, 2),
        "plot": plot_b64,
    }
