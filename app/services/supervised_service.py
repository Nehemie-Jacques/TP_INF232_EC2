import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from app.services.plot_service import fig_to_base64, PALETTE


def _get_model(model_name, params):
    models = {
        "knn": KNeighborsClassifier(
            n_neighbors=int(params.get("k", 5)),
            weights=params.get("weights", "uniform"),
            metric=params.get("metric", "minkowski")
        ),
        "svm": SVC(
            kernel=params.get("kernel", "rbf"),
            C=float(params.get("C", 1.0)),
            gamma=params.get("gamma", "scale"),
            probability=True,
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=params.get("max_depth") or None,
            random_state=42,
            n_jobs=-1
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=int(params.get("max_depth", 5)),
            criterion=params.get("criterion", "gini"),
            random_state=42
        ),
    }
    return models.get(model_name)


def classify(df, feature_cols, target_col, model_name="random_forest",
             test_size=0.2, params=None):
    if params is None:
        params = {}

    cols = feature_cols + [target_col]
    data = df[cols].dropna()
    if len(data) < 10:
        raise ValueError("Données insuffisantes (min. 10 lignes).")

    X = data[feature_cols].values
    y_raw = data[target_col].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    n_classes = len(le.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42,
        stratify=y if n_classes > 1 else None
    )

    model = _get_model(model_name, params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    report = classification_report(y_test, y_pred,
                                   target_names=le.classes_, zero_division=0)

    # Graphiques
    n_plots = 3 if n_classes == 2 and hasattr(model, "predict_proba") else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots + 2, 5))

    # 1. Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                ax=axes[0], linewidths=0.5)
    axes[0].set_xlabel("Prédits", fontsize=10)
    axes[0].set_ylabel("Réels", fontsize=10)
    axes[0].set_title(f"Matrice de confusion\nAccuracy = {accuracy:.4f}", fontsize=11)

    # 2. Importance des features (RF/DT)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = min(12, len(feature_cols))
        feats = [feature_cols[i] for i in indices[:top_n]]
        vals = importances[indices[:top_n]]
        axes[1].barh(feats[::-1], vals[::-1], color=PALETTE["green"], alpha=0.8)
        axes[1].set_title("Importance des variables", fontsize=11)
        axes[1].set_xlabel("Importance (Gini)", fontsize=10)
    else:
        # Graphique barres métriques
        metrics_vals = [accuracy, precision, recall, f1]
        metrics_names = ["Accuracy", "Précision", "Rappel", "F1-Score"]
        colors_list = [PALETTE["blue"], PALETTE["green"], PALETTE["purple"], PALETTE["orange"]]
        bars = axes[1].bar(metrics_names, metrics_vals, color=colors_list, alpha=0.8)
        axes[1].set_ylim(0, 1.1)
        axes[1].set_title(f"Métriques — {model_name.upper()}", fontsize=11)
        for bar, val in zip(bars, metrics_vals):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 3. Courbe ROC (binaire uniquement)
    if n_plots == 3:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        axes[2].plot(fpr, tpr, color=PALETTE["blue"], linewidth=2,
                     label=f"AUC = {roc_auc:.4f}")
        axes[2].plot([0, 1], [0, 1], "--", color=PALETTE["gray"], linewidth=1)
        axes[2].set_xlabel("Taux de faux positifs", fontsize=10)
        axes[2].set_ylabel("Taux de vrais positifs", fontsize=10)
        axes[2].set_title("Courbe ROC", fontsize=11)
        axes[2].legend(fontsize=10)

    plt.suptitle(f"Classification supervisée — {model_name.replace('_', ' ').title()}",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plot_b64 = fig_to_base64(fig)

    # Feature importances si disponibles
    feature_importances = {}
    if hasattr(model, "feature_importances_"):
        feature_importances = {f: round(float(v), 6)
                               for f, v in zip(feature_cols, model.feature_importances_)}
        feature_importances = dict(sorted(feature_importances.items(),
                                           key=lambda x: x[1], reverse=True))

    return {
        "model_name": model_name,
        "model_display": model_name.replace("_", " ").title(),
        "feature_cols": feature_cols,
        "target_col": target_col,
        "classes": list(le.classes_),
        "n_classes": n_classes,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": round(accuracy, 4),
        "accuracy_percent": round(accuracy * 100, 2),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "cv_scores": [round(float(s), 4) for s in cv_scores],
        "classification_report": report,
        "feature_importances": feature_importances,
        "params": params,
        "plot": plot_b64,
    }
