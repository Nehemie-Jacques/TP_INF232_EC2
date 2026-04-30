import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from app.services.plot_service import fig_to_base64, PALETTE


def simple_regression(df, x_col, y_col, test_size=0.2):
    """Régression linéaire simple Y = aX + b."""
    data = df[[x_col, y_col]].dropna()
    if len(data) < 10:
        raise ValueError("Pas assez de données (minimum 10 lignes sans valeurs manquantes).")

    X = data[[x_col]].values
    y = data[y_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_all = model.predict(X)

    r2 = r2_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)

    coef = float(model.coef_[0])
    intercept = float(model.intercept_)

    # Graphique
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter + droite de régression
    axes[0].scatter(X, y, color=PALETTE["blue"], alpha=0.6, s=40, label="Données")
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    axes[0].plot(x_line, model.predict(x_line), color=PALETTE["orange"],
                 linewidth=2.5, label=f"ŷ = {coef:.4f}x + {intercept:.4f}")
    axes[0].scatter(X_test, y_test, color=PALETTE["green"], alpha=0.8, s=60,
                    label="Ensemble test", marker="^", zorder=5)
    axes[0].set_xlabel(x_col, fontsize=10)
    axes[0].set_ylabel(y_col, fontsize=10)
    axes[0].set_title(f"Régression simple\nR² = {r2:.4f}", fontsize=11)
    axes[0].legend(fontsize=9)

    # Résidus
    residuals = y_test - y_pred_test
    axes[1].scatter(y_pred_test, residuals, color=PALETTE["purple"], alpha=0.7, s=50)
    axes[1].axhline(0, color=PALETTE["orange"], linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Valeurs prédites", fontsize=10)
    axes[1].set_ylabel("Résidus", fontsize=10)
    axes[1].set_title("Graphique des résidus", fontsize=11)

    plt.tight_layout()
    plot_b64 = fig_to_base64(fig)

    return {
        "type": "simple",
        "x_col": x_col,
        "y_col": y_col,
        "coef": round(coef, 6),
        "intercept": round(intercept, 6),
        "equation": f"ŷ = {coef:.4f} × {x_col} + {intercept:.4f}",
        "r2": round(r2, 4),
        "r2_percent": round(r2 * 100, 2),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "plot": plot_b64,
    }


def multiple_regression(df, feature_cols, target_col, test_size=0.2):
    """Régression linéaire multiple Y = a1X1 + a2X2 + ... + b."""
    cols = feature_cols + [target_col]
    data = df[cols].dropna()

    if len(data) < 10:
        raise ValueError("Pas assez de données.")
    if len(feature_cols) < 2:
        raise ValueError("Sélectionnez au moins 2 variables prédicteurs.")

    X = data[feature_cols].values
    y = data[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # R² ajusté
    n, p = len(y_test), len(feature_cols)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    coefficients = {col: round(float(c), 6)
                    for col, c in zip(feature_cols, model.coef_)}

    # Graphiques
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Prédits vs réels
    axes[0].scatter(y_test, y_pred, color=PALETTE["blue"], alpha=0.6, s=50)
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lim, lim, color=PALETTE["orange"], linestyle="--", linewidth=2, label="Idéal")
    axes[0].set_xlabel("Valeurs réelles", fontsize=10)
    axes[0].set_ylabel("Valeurs prédites", fontsize=10)
    axes[0].set_title(f"Prédits vs Réels\nR² = {r2:.4f}", fontsize=11)
    axes[0].legend(fontsize=9)

    # Résidus
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, color=PALETTE["purple"], alpha=0.6, s=50)
    axes[1].axhline(0, color=PALETTE["orange"], linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Valeurs prédites", fontsize=10)
    axes[1].set_ylabel("Résidus", fontsize=10)
    axes[1].set_title("Graphique des résidus", fontsize=11)

    # Coefficients
    colors = [PALETTE["green"] if c > 0 else PALETTE["orange"] for c in model.coef_]
    bars = axes[2].barh(feature_cols, model.coef_, color=colors, alpha=0.8)
    axes[2].axvline(0, color=PALETTE["gray"], linewidth=1)
    axes[2].set_xlabel("Coefficient", fontsize=10)
    axes[2].set_title("Coefficients de régression", fontsize=11)
    for bar, val in zip(bars, model.coef_):
        axes[2].text(bar.get_width() + (0.02 * max(abs(model.coef_))),
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plot_b64 = fig_to_base64(fig)

    return {
        "type": "multiple",
        "feature_cols": feature_cols,
        "target_col": target_col,
        "coefficients": coefficients,
        "intercept": round(float(model.intercept_), 6),
        "r2": round(r2, 4),
        "r2_adj": round(r2_adj, 4),
        "r2_percent": round(r2 * 100, 2),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "plot": plot_b64,
    }
