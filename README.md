# INF232 EC2 — Application de Collecte et Analyse de Données

> **TP INF232** — Développement d'une application web de collecte des données en ligne avec analyse descriptive et Machine Learning.

---

## Présentation

Cette application web Python/Flask permet de :

1. **Collecter des données** — via import de fichiers CSV/Excel ou saisie manuelle
2. **Explorer les données** — statistiques descriptives, histogrammes, corrélations, nuages de points
3. **Analyser les données** avec 5 modules ML complets :
   - Régression linéaire simple et multiple
   - Réduction de dimensionnalité (PCA, t-SNE, LDA)
   - Classification supervisée (KNN, SVM, Random Forest, Arbre de Décision)
   - Classification non-supervisée (K-Means, DBSCAN, CAH)
4. **Exporter les résultats** — CSV, JSON, Rapport PDF

---

## Stack technique

| Couche | Technologie |
|--------|-------------|
| Backend | Python 3.10+, Flask 3.x |
| ORM | SQLAlchemy + Flask-Migrate |
| Auth | Flask-Login |
| ML | Scikit-learn, Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Frontend | Jinja2, Bootstrap 5, Bootstrap Icons |
| Base de données | SQLite (dev) / PostgreSQL (prod) |
| Export | ReportLab (PDF), OpenPyXL (Excel) |

---

## Installation

### Prérequis

- Python 3.10 ou supérieur
- pip

### Étapes

```bash
# 1. Cloner ou dézipper le projet
cd inf232_ec2

# 2. Créer l'environnement virtuel
python -m venv venv

# 3. Activer l'environnement
# Linux / macOS :
source venv/bin/activate
# Windows :
venv\Scripts\activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Initialiser la base de données
flask db init
flask db migrate -m "Initial"
flask db upgrade

# 6. Lancer l'application
flask run --debug
```

Ouvrez votre navigateur sur **http://127.0.0.1:5000**

---

## Structure du projet

```
inf232_ec2/
├── app.py                        # Point d'entrée
├── config.py                     # Configuration (dev/prod)
├── .env                          # Variables d'environnement
├── requirements.txt
│
├── app/
│   ├── __init__.py               # Factory Flask (create_app)
│   ├── models/
│   │   ├── user.py               # Modèle utilisateur + auth
│   │   ├── dataset.py            # Modèle dataset importé
│   │   └── analysis.py           # Modèle résultat d'analyse
│   │
│   ├── routes/                   # Blueprints Flask
│   │   ├── auth.py               # /auth/login, register, logout
│   │   ├── data.py               # /data/upload, form, preview
│   │   ├── regression.py         # /analysis/regression/simple|multiple
│   │   ├── reduction.py          # /analysis/reduction/pca|tsne|lda
│   │   ├── supervised.py         # /analysis/supervised/<model>
│   │   ├── unsupervised.py       # /analysis/unsupervised/kmeans|dbscan|hierarchical
│   │   └── export.py             # /export/csv|json|pdf
│   │
│   ├── services/                 # Logique métier (ML + data)
│   │   ├── data_service.py       # Lecture, nettoyage, stats
│   │   ├── plot_service.py       # Génération centralisée des graphiques
│   │   ├── regression_service.py # Régression simple + multiple
│   │   ├── reduction_service.py  # PCA, t-SNE, LDA
│   │   ├── supervised_service.py # KNN, SVM, RF, DT
│   │   └── unsupervised_service.py # K-Means, DBSCAN, CAH
│   │
│   ├── templates/                # Templates Jinja2
│   │   ├── base.html             # Layout + sidebar + navbar
│   │   ├── dashboard.html
│   │   ├── auth/login.html, register.html
│   │   ├── data/upload.html, form.html, preview.html
│   │   ├── analysis/regression.html, reduction.html,
│   │   │           supervised.html, unsupervised.html
│   │   └── errors/404.html, 500.html
│   │
│   ├── static/
│   │   ├── css/style.css         # Styles personnalisés
│   │   ├── js/charts.js          # Interactions frontend
│   │   ├── uploads/              # Fichiers importés
│   │   └── plots/                # Graphiques générés
│   │
│   └── utils/
│       ├── helpers.py            # Fonctions utilitaires
│       └── validators.py         # Validations de données
│
├── migrations/                   # Fichiers de migration DB
├── instance/app.db               # Base SQLite (générée automatiquement)
└── tests/
    ├── test_routes.py            # Tests des routes HTTP
    └── test_services.py          # Tests unitaires ML
```

---

## Utilisation

### 1. Créer un compte

Accédez à `/auth/register` et créez votre compte utilisateur.

### 2. Importer des données

- **Fichier CSV/Excel** : `/data/upload` — glissez-déposez votre fichier
- **Saisie manuelle** : `/data/form` — entrez les colonnes et données ligne par ligne

### 3. Explorer les données

Accédez à `/data/preview` pour voir :
- Tableau d'aperçu des 15 premières lignes
- Statistiques descriptives (moyenne, écart-type, min/max, quartiles…)
- Histogrammes interactifs et boîtes à moustaches
- Matrice de corrélation
- Nuage de points interactif

### 4. Lancer les analyses

| Module | URL | Description |
|--------|-----|-------------|
| Régression simple | `/analysis/regression/simple` | Y = aX + b |
| Régression multiple | `/analysis/regression/multiple` | Y = a₁X₁ + a₂X₂ + … |
| PCA | `/analysis/reduction/pca` | Réduction + cercle des corrélations |
| t-SNE | `/analysis/reduction/tsne` | Visualisation non-linéaire |
| LDA | `/analysis/reduction/lda` | Discrimination supervisée |
| KNN | `/analysis/supervised/knn` | K plus proches voisins |
| SVM | `/analysis/supervised/svm` | Séparateur à vaste marge |
| Random Forest | `/analysis/supervised/random_forest` | Forêts aléatoires |
| Arbre de décision | `/analysis/supervised/decision_tree` | Arbre CART |
| K-Means | `/analysis/unsupervised/kmeans` | Clustering centroïde |
| DBSCAN | `/analysis/unsupervised/dbscan` | Clustering par densité |
| CAH | `/analysis/unsupervised/hierarchical` | Clustering hiérarchique + dendrogramme |

### 5. Exporter les résultats

| Format | URL | Contenu |
|--------|-----|---------|
| CSV | `/export/csv` | Dataset complet |
| JSON | `/export/stats/json` | Statistiques descriptives |
| PDF | `/export/stats/pdf` | Rapport complet avec tableaux |

---

## Lancer les tests

```bash
# Installer pytest
pip install pytest

# Lancer tous les tests
pytest tests/ -v

# Tests par module
pytest tests/test_services.py -v
pytest tests/test_routes.py -v
```

---

## Modules ML — Description

### Régression linéaire simple
Modélise la relation linéaire entre une variable X et une variable Y.
- Métriques : R², MSE, RMSE, équation de la droite
- Graphiques : droite de régression + graphique des résidus

### Régression linéaire multiple
Modélise Y à partir de plusieurs variables prédicteurs.
- Métriques : R², R² ajusté, MSE, coefficients par variable
- Graphiques : prédits vs réels, résidus, importance des coefficients

### PCA (Analyse en Composantes Principales)
Réduit la dimensionnalité en créant des composantes non-corrélées.
- Visualisations : projection 2D, éboulis des valeurs propres, cercle des corrélations
- Indique combien de composantes expliquent 80% et 95% de la variance

### t-SNE
Algorithme de visualisation non-linéaire pour explorer la structure des données.
- Paramètres : perplexité (5-50), nombre d'itérations

### LDA (Analyse Discriminante Linéaire)
Réduction supervisée qui maximise la séparation entre classes connues.

### KNN (K-Nearest Neighbors)
Classe chaque observation selon les k voisins les plus proches.

### SVM (Support Vector Machine)
Trouve le séparateur à marge maximale entre les classes.
- Noyaux : RBF, linéaire, polynomial, sigmoïde

### Random Forest
Ensemble de 100+ arbres de décision. Fournit l'importance des variables.

### Arbre de décision (CART)
Arbre de classification interprétable avec règles de décision.

### K-Means
Partitionne les données en k clusters en minimisant l'inertie.
- Aide au choix de k : méthode du coude + score de silhouette

### DBSCAN
Clustering basé sur la densité, détecte automatiquement les outliers.

### CAH (Classification Ascendante Hiérarchique)
Construit une hiérarchie de clusters visualisée par dendrogramme.

---

## Concepts théoriques couverts

| Concept | Module |
|---------|--------|
| Corrélation de Pearson | Aperçu des données |
| Droite des moindres carrés | Régression simple |
| R², MSE, RMSE | Régression |
| Valeurs propres / vecteurs propres | PCA |
| Variance expliquée | PCA |
| Entropie, indice de Gini | Arbre de décision |
| Matrice de confusion, F1, Précision, Rappel | Classification supervisée |
| Validation croisée k-fold | Classification supervisée |
| Courbe ROC, AUC | Classification binaire |
| Score de silhouette | Clustering |
| Indice de Davies-Bouldin | Clustering |
| Dendrogramme | CAH |

---

## Variables d'environnement

Fichier `.env` à la racine du projet :

```env
SECRET_KEY=votre-clé-secrète-très-longue
DATABASE_URL=sqlite:///instance/app.db
FLASK_ENV=development
FLASK_APP=app.py
```

---

## Auteur & Contexte

**Projet** : INF232 EC2 — TP de développement d'application web de collecte et analyse de données  
**Technologies** : Python, Flask, Scikit-learn, Pandas, Matplotlib, Bootstrap 5  
**Objectif pédagogique** : Couvrir les 5 grands axes de l'analyse de données ML (régression, réduction, classification supervisée et non-supervisée, analyse descriptive)
