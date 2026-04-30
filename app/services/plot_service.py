import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Thème global cohérent
PALETTE = {
    "blue": "#185FA5",
    "orange": "#D85A30",
    "green": "#0F6E56",
    "purple": "#534AB7",
    "gray": "#5F5E5A",
    "amber": "#BA7517",
}

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "DejaVu Sans",
})


def fig_to_base64(fig):
    """Convertit une figure Matplotlib en chaîne base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def plot_histogram(df, column, bins=20):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    data = df[column].dropna()

    axes[0].hist(data, bins=bins, color=PALETTE["blue"], alpha=0.8, edgecolor="white")
    axes[0].axvline(data.mean(), color=PALETTE["orange"], linestyle="--",
                    linewidth=1.5, label=f"Moyenne: {data.mean():.2f}")
    axes[0].axvline(data.median(), color=PALETTE["green"], linestyle="--",
                    linewidth=1.5, label=f"Médiane: {data.median():.2f}")
    axes[0].set_title(f"Histogramme — {column}")
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("Fréquence")
    axes[0].legend(fontsize=9)

    axes[1].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=PALETTE["blue"], alpha=0.6),
                    medianprops=dict(color=PALETTE["orange"], linewidth=2))
    axes[1].set_title(f"Boîte à moustaches — {column}")
    axes[1].set_ylabel(column)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr()
    n = len(corr)
    size = max(6, n * 0.8)
    fig, ax = plt.subplots(figsize=(min(size, 14), min(size, 12)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 9})
    ax.set_title("Matrice de corrélation", fontsize=13, pad=12)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_scatter(df, x_col, y_col, color_col=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    if color_col and color_col in df.columns:
        categories = df[color_col].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
        for cat, c in zip(categories, colors):
            mask = df[color_col] == cat
            ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                       c=[c], label=str(cat), alpha=0.7, s=40)
        ax.legend(title=color_col, fontsize=9)
    else:
        ax.scatter(df[x_col], df[y_col], color=PALETTE["blue"], alpha=0.6, s=40)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Nuage de points : {x_col} vs {y_col}")
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_distribution_grid(df, max_cols=6):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    if not numeric_cols:
        return None
    n = len(numeric_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(numeric_cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=15, color=PALETTE["blue"], alpha=0.75, edgecolor="white")
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
        axes[i].tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribution des variables numériques", fontsize=12, y=1.01)
    plt.tight_layout()
    return fig_to_base64(fig)
