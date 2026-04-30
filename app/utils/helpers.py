"""Fonctions utilitaires générales."""
import os
import re
from datetime import datetime


def format_file_size(size_bytes):
    """Formate une taille en octets en chaîne lisible."""
    for unit in ["o", "Ko", "Mo", "Go"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} To"


def slugify(text):
    """Convertit un texte en slug URL-safe."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text


def truncate(text, max_length=50):
    """Tronque un texte à une longueur maximale."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_number(value, decimals=4):
    """Formate un nombre avec un nombre de décimales donné."""
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return value


def get_badge_color(score, thresholds=None):
    """Retourne une classe Bootstrap selon un score."""
    if thresholds is None:
        thresholds = {"success": 0.85, "warning": 0.70}
    if score >= thresholds["success"]:
        return "success"
    elif score >= thresholds["warning"]:
        return "warning"
    return "danger"


def timestamp_to_str(dt):
    """Convertit un datetime en chaîne."""
    if dt is None:
        return "—"
    return dt.strftime("%d/%m/%Y %H:%M")
