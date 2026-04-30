#!/usr/bin/env python3
"""
Script d'initialisation de la base de données.
À lancer une seule fois après installation.
Usage: python init_db.py
"""
import os
import sys

# S'assurer que le dossier instance existe
os.makedirs("instance", exist_ok=True)
os.makedirs(os.path.join("app", "static", "uploads"), exist_ok=True)
os.makedirs(os.path.join("app", "static", "plots"), exist_ok=True)

from app import create_app, db
from app.models.user import User
from app.models.dataset import Dataset
from app.models.analysis import Analysis

app = create_app()

with app.app_context():
    db.create_all()
    print("✅ Base de données initialisée.")
    print("   Tables créées :", ["users", "datasets", "analyses"])

    # Créer un utilisateur démo (optionnel)
    if not User.query.filter_by(username="demo").first():
        demo = User(username="demo", email="demo@inf232.local")
        demo.set_password("demo123")
        db.session.add(demo)
        db.session.commit()
        print("✅ Utilisateur démo créé.")
        print("   Login : demo / demo123")
    else:
        print("ℹ  Utilisateur démo déjà existant.")

print("\n🚀 Vous pouvez maintenant lancer l'application :")
print("   flask run --debug")
print("   → http://127.0.0.1:5000")
