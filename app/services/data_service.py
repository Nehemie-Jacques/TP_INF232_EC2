import os
import json
import uuid
import pandas as pd
import numpy as np
from flask import session, current_app
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file):
    """Sauvegarde un fichier uploadé, retourne le nom de fichier unique."""
    original_name = secure_filename(file.filename)
    ext = original_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_name)
    file.save(path)
    return unique_name, original_name


def load_dataframe(filename):
    """Charge un DataFrame depuis le dossier uploads."""
    path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(path):
        return None
    ext = filename.rsplit(".", 1)[1].lower()
    try:
        if ext == "csv":
            # Détection automatique du séparateur
            df = pd.read_csv(path, sep=None, engine="python")
        else:
            df = pd.read_excel(path)
        return df
    except Exception:
        return None


def get_dataset_info(df):
    """Retourne les informations descriptives d'un DataFrame."""
    if df is None:
        return {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    info = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": int(df.isnull().sum().sum()),
        "missing_by_col": df.isnull().sum().to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
    }

    # Statistiques descriptives pour colonnes numériques
    if numeric_cols:
        desc = df[numeric_cols].describe().round(4)
        info["stats"] = desc.to_dict()

    return info


def get_dataframe_preview(df, n=10):
    """Retourne les premières lignes sous forme de liste de dicts."""
    if df is None:
        return [], []
    preview = df.head(n).fillna("NaN")
    return df.columns.tolist(), preview.values.tolist()


def preprocess_dataframe(df, drop_na=True, normalize=False, feature_cols=None):
    """Prétraitement de base : suppression NaN, normalisation optionnelle."""
    if feature_cols:
        df = df[feature_cols].copy()
    else:
        df = df.copy()

    if drop_na:
        df = df.dropna()

    if normalize:
        from sklearn.preprocessing import StandardScaler
        numeric = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df[numeric] = scaler.fit_transform(df[numeric])

    return df


def save_manual_data(rows_data, columns):
    """Sauvegarde des données saisies manuellement en CSV."""
    df = pd.DataFrame(rows_data, columns=columns)
    unique_name = f"{uuid.uuid4().hex}.csv"
    path = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_name)
    df.to_csv(path, index=False)
    return unique_name, "saisie_manuelle.csv", df


def get_current_dataset_filename():
    """Récupère le nom de fichier du dataset courant depuis la session."""
    return session.get("current_dataset")


def set_current_dataset(filename):
    """Enregistre le dataset courant dans la session."""
    session["current_dataset"] = filename


def load_current_dataframe():
    """Charge le DataFrame du dataset courant."""
    filename = get_current_dataset_filename()
    if not filename:
        return None
    return load_dataframe(filename)


def get_descriptive_stats(df):
    """Calcule les statistiques descriptives complètes."""
    if df is None:
        return {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {}

    result = {}
    for col in numeric_cols:
        serie = df[col].dropna()
        result[col] = {
            "count": int(serie.count()),
            "mean": round(float(serie.mean()), 4),
            "std": round(float(serie.std()), 4),
            "min": round(float(serie.min()), 4),
            "q25": round(float(serie.quantile(0.25)), 4),
            "median": round(float(serie.median()), 4),
            "q75": round(float(serie.quantile(0.75)), 4),
            "max": round(float(serie.max()), 4),
            "skewness": round(float(serie.skew()), 4),
            "kurtosis": round(float(serie.kurtosis()), 4),
            "missing": int(df[col].isnull().sum()),
        }
    return result
