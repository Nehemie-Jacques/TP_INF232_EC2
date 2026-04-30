"""Fonctions de validation des données."""
import re
import pandas as pd
import numpy as np


def validate_column_selection(df, cols, min_count=1, numeric_only=False):
    """Valide une sélection de colonnes."""
    errors = []
    if not cols:
        errors.append(f"Sélectionnez au moins {min_count} colonne(s).")
        return errors
    if len(cols) < min_count:
        errors.append(f"Sélectionnez au moins {min_count} colonne(s).")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        errors.append(f"Colonnes introuvables : {', '.join(missing)}")
    if numeric_only:
        non_numeric = [c for c in cols if c in df.columns
                       and not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            errors.append(f"Colonnes non-numériques : {', '.join(non_numeric)}")
    return errors


def validate_sufficient_data(df, feature_cols, min_rows=10):
    """Vérifie qu'il y a assez de données non-nulles."""
    if df is None:
        return ["Aucun dataset disponible."]
    clean = df[feature_cols].dropna()
    if len(clean) < min_rows:
        return [f"Données insuffisantes après suppression des NaN : "
                f"{len(clean)} lignes (minimum {min_rows})."]
    return []


def validate_email(email):
    """Validation basique d'email."""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))


def validate_password(password):
    """Valide un mot de passe."""
    errors = []
    if len(password) < 6:
        errors.append("Le mot de passe doit avoir au moins 6 caractères.")
    return errors


def sanitize_column_name(name):
    """Nettoie un nom de colonne."""
    name = re.sub(r"[^\w\s]", "", str(name))
    name = name.strip().replace(" ", "_")
    return name or "col"
