#  NYC Taxi Fare Prediction

Projet de Machine Learning pour l'analyse et la prédiction des tarifs de taxi à New York basé sur les données de 2018.

##  Description

Ce projet contient deux notebooks Jupyter qui explorent et modélisent les données des taxis de NYC :

1. **`nyc_taxi_analysis.ipynb`** - Analyse exploratoire des données (EDA)
2. **`fare_prediction_model.ipynb`** - Modèle de prédiction des tarifs

##  Objectifs

- Analyser les patterns de trajets en taxi à New York
- Identifier les facteurs qui influencent le tarif
- Construire un modèle de régression linéaire pour prédire le prix d'une course

## Dataset

- **Source** : Données des taxis NYC 2018
- **Fichier** : `datasets/original_cleaned_nyc_taxi_data_2018.csv`
- **Taille** : 8M+ lignes de courses de taxi
- **URL** : https://www.kaggle.com/datasets/neilclack/nyc-taxi-trip-data-google-public-data?utm_source=chatgpt.com
### Variables principales

| Variable | Description |
|----------|-------------|
| `fare_amount` | Tarif de la course (variable cible) |
| `trip_distance` | Distance parcourue |
| `trip_duration` | Durée du trajet |
| `rate_code` | Type de tarification |
| `payment_type` | Mode de paiement |
| `tip_amount` | Pourboire |

## 🛠️ Installation

### Prérequis

- Python 3.8+
- pip

### Étapes

1. **Cloner le repository**
   ```bash
   git clone https://github.com/yassinefri/Campaign-Analytics-Platform.git
   cd Campaign-Analytics-Platform
   ```

2. **Créer un environnement virtuel** (recommandé)
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # ou
   venv\Scripts\activate     # Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Lancer Jupyter**
   ```bash
   jupyter notebook
   ```

## Dépendances

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

##  Contenu des Notebooks

### 1. Analyse Exploratoire (`nyc_taxi_analysis.ipynb`)

- Chargement et exploration des données
- Analyse des valeurs manquantes
- Statistiques descriptives
- Analyse des variables catégorielles
- Visualisations des distributions
- Analyse des corrélations

### 2. Modèle de Prédiction (`fare_prediction_model.ipynb`)

- Préparation et nettoyage des données
- Échantillonnage (100k lignes pour l'entraînement)
- Sélection des features
- Analyse des corrélations
- Entraînement du modèle de régression linéaire
- Évaluation des performances (R², MSE, MAE)
- Visualisation des prédictions vs valeurs réelles

##  Méthodologie

1. **Nettoyage des données**
   - Suppression des valeurs aberrantes
   - Filtrage des tarifs entre 0 et 200$
   - Filtrage des distances entre 0 et 100 miles
   - Filtrage des durées entre 1 min et 2h

2. **Feature Engineering**
   - Sélection des variables les plus corrélées
   - Encodage des variables catégorielles

3. **Modélisation**
   - Régression linéaire (scikit-learn)
   - Split train/test
   - Validation croisée

##  Résultats

Le modèle de régression linéaire permet de prédire le tarif d'une course en fonction de :
- La distance du trajet
- Le type de tarification
- Le mode de paiement
- Les taxes et suppléments

##  Structure du Projet

```
Campaign-Analytics-Platform/
├── README.md
├── requirements.txt
├── fare_prediction_model.ipynb
├── nyc_taxi_analysis.ipynb
└── datasets/
    └── original_cleaned_nyc_taxi_data_2018.csv