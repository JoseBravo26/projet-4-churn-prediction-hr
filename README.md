# 👥 Prédicteur de Churn des Employés

**Status** : ✅ Production Ready | **Version** : 1.0.0 | **Dernière mise à jour** : Mars 2026

---

## 📊 Description

Application complète de machine learning qui prédit si un employé risque de quitter l'entreprise en fonction de ses données personnelles, professionnelles et de satisfaction au travail.

Le modèle est exposé via une **API REST développée avec FastAPI**, connectée à une base **PostgreSQL** qui enregistre les entrées, les prédictions et un journal d'audit des appels.

---

## 🎯 Objectif

Identifier les employés ayant un risque élevé de départ pour mettre en place des stratégies de rétention proactives et mesurables.

---

## 📈 Caractéristiques du Modèle

| Caractéristique | Valeur |
|---|---|
| **Algorithme** | Logistic Regression (Optimisée) |
| **Dataset d'entraînement** | 1 470 employés |
| **Nombre de features** | 23 variables prédictives |
| **Accuracy** | ~95% |
| **Seuil optimal** | Configuré pour maximiser le recall |
| **Coverage des tests** | 77% (11 tests, 244 lignes) |

---

## 🏗️ Architecture Technique

```
Client (Front / Autre service)
          │
          ▼
     ┌─────────────────────┐
     │  API FastAPI        │
     │  (main_api.py)      │
     └────────┬────────────┘
              │
         ┌────┴──────┐
         │           │
         ▼           ▼
    ┌─────────┐  ┌──────────────┐
    │ Models/ │  │ PostgreSQL   │
    │ *.pkl   │  │ - employees  │
    └─────────┘  │ - predictions│
                 │ - audit_log  │
                 └──────────────┘
```

### Flux de traitement

À chaque appel `/predict` ou `/test-prediction`, l'API :

1. **Valide les données** : Schéma Pydantic `EmployeeInput`
2. **Prétraite** : Applique le `StandardScaler` (normalisation des features)
3. **Prédiction** : Appelle le modèle `LogisticRegression` pour obtenir la probabilité de churn
4. **Seuillage** : Compare la probabilité au seuil optimal pour classifier (Risque Élevé / Faible)
5. **Enregistrement** : Stocke l'employé dans `employees`, la prédiction dans `predictions`, et trace l'appel dans `audit_log`
6. **Réponse** : Retourne `PredictionResponse` avec risque, probabilité, seuil et recommandation

---

## 📝 Variables d'Entrée (23 features)

### 👤 Informations Personnelles
- **Âge** (18-65 ans)
- **Niveau d'éducation** (1-5)
- **Distance domicile-travail** (km)

### 💼 Expérience et Trajectoire
- **Expériences précédentes**
- **Années d'expérience totale**
- **Années dans l'entreprise**
- **Années au poste actuel**

### 📊 Évaluation et Performance
- **Évaluation précédente** (1-4)
- **Évaluation actuelle** (1-4)
- **Niveau hiérarchique** (1-5)
- **Employés sous responsabilité**

### 😊 Satisfaction au Travail (1-4 chacun)
- **Satisfaction de l'environnement**
- **Satisfaction de la nature du travail**
- **Satisfaction de l'équipe**
- **Satisfaction de l'équilibre vie-travail**

### 💰 Compensation et Avantages
- **Revenu mensuel** (€)
- **Dernière augmentation salaire** (%)
- **Heures supplémentaires** (Oui/Non)
- **Participation plan actions** (PEE)
- **Formations complétées**

### 🚀 Progression et Carrière
- **Années depuis dernière promotion**
- **Années sous responsable actuel**

---

## 🌐 API FastAPI

L'API est documentée automatiquement via **Swagger/OpenAPI** :

- **Documentation interactive** : `http://127.0.0.1:8000/docs`
- **Schéma OpenAPI** : `http://127.0.0.1:8000/openapi.json`

### Endpoints Principaux

#### `GET /health`
Vérifie l'état de l'API et du modèle.

**Réponse** :
```json
{
  "status": "OK",
  "model_loaded": true,
  "scaler_loaded": true,
  "threshold_loaded": true,
  "database_connected": true
}
```

#### `POST /predict`
Prédiction d'un employé individuel.

**Body** :
```json
{
  "age": 35,
  "education_level": 3,
  "distance_from_home": 15,
  "experience_count": 2,
  "total_experience": 10,
  "years_in_company": 5,
  "years_in_role": 2,
  "previous_evaluation": 3,
  "current_evaluation": 4,
  "hierarchy_level": 2,
  "subordinates_count": 3,
  "environment_satisfaction": 3,
  "job_satisfaction": 4,
  "team_satisfaction": 3,
  "worklife_balance_satisfaction": 3,
  "monthly_revenue": 3500,
  "last_raise_percentage": 11,
  "overtime": false,
  "stock_option_level": 1,
  "trainings_completed": 2,
  "years_since_promotion": 2,
  "years_with_current_manager": 2
}
```

**Réponse** :
```json
{
  "risk_level": "Faible",
  "churn_probability": 0.15,
  "threshold_applied": 0.5,
  "recommendation": "Maintenir la relation positive, surveiller satisfactions"
}
```

#### `POST /test-prediction`
Teste rapidement l'API avec un employé fictif.

**Réponse** : Même format que `/predict`

#### `POST /predict-bulk`
Prédictions en masse pour une liste d'employés.

**Body** :
```json
{
  "employees": [
    { /* EmployeeInput 1 */ },
    { /* EmployeeInput 2 */ }
  ]
}
```

**Réponse** :
```json
{
  "predictions": [
    { /* PredictionResponse 1 */ },
    { /* PredictionResponse 2 */ }
  ],
  "statistics": {
    "total_processed": 2,
    "high_risk_count": 0,
    "low_risk_count": 2,
    "high_risk_percentage": 0.0
  }
}
```

---

## 🚀 Installation Locale (API FastAPI + PostgreSQL)

### Prérequis

- **Python 3.10+**
- **PostgreSQL 12+** avec base `churn_predictor_db` et utilisateur `churn_app`
- **Fichiers de modèle** dans `models/` :
  - `lr_model_opt.pkl` (Logistic Regression)
  - `scaler.pkl` (StandardScaler)
  - `seuil_opt.pkl` (Seuil optimal)

### Étapes d'installation

```bash
# 1. Cloner le repo
git clone <URL-du-repo>
cd projet-4-churn-prediction-hr

# 2. Créer l'environnement virtuel
Pandas : Manipulation de données

Numpy : Calculs numériques

Joblib : Sérialisation modèles

Gradio : Interface utilisateur

Hugging Face Spaces : Hébergement gratuit

Architecture
text
┌─────────────────┐
│   Données Input │
└────────┬────────┘
         │
    ┌────▼────┐
    │  Scaler │ (Normalisation)
    └────┬────┘
         │
    ┌────▼──────────┐
    │  LR Model     │ (Prédiction)
    └────┬──────────┘
         │
    ┌────▼────────────┐
    │ Seuil Optimal   │ (Classification)
    └────┬────────────┘
         │
    ┌────▼──────────┐
    │  Résultat     │
    └───────────────┘
📊 Métriques du Modèle
Accuracy : ~95%

Precision : Élevée (peu de faux positifs)

Recall : Optimisé (captures maximum de churn réels)

AUC-ROC : Excellent discriminant

Threshold : 0.5 (seuil de probabilité)

📋 Installation Locale (Développement)
bash
# 1. Cloner le repository
git clone https://huggingface.co/spaces/josibra/churn-predictor](https://huggingface.co/spaces/josibra/churn-predictor
cd churn-predictor

# 2. Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Créer la base de données et les tables
# (Voir le script sql/create_db.sql fourni)
winget install --id PostgreSQL.PostgreSQL.17
psql -U postgres -f sql/create_db.sql

# 5. Lancer l'API FastAPI
python main_api.py
# Ou avec uvicorn directement :
# uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
```

### Points d'accès
# 5. Accéder à l'interface
# http://localhost:7860
🌐 Accès en Ligne
L'application est disponible sur Hugging Face Spaces :
[https://huggingface.co/spaces/josibra/churn-predictor](https://huggingface.co/spaces/josibra/churn-predictor)
🎓 Données d'Entraînement
Source : Système d'Information Ressources Humaines (SIRH)

- **API** : `http://127.0.0.1:8000`
- **Docs Swagger** : `http://127.0.0.1:8000/docs`
- **Endpoint de test** : `http://127.0.0.1:8000/test-prediction`

---

## ✅ Tests et Couverture

Les tests unitaires et fonctionnels sont implémentés avec **Pytest** :

### Fichiers de test

- **`tests/test_preprocessing.py`** : tests du prétraitement des données et de la validation Pydantic
- **`tests/test_model.py`** : tests du modèle (chargement, formes des probabilités, cas à haut risque)
- **`tests/test_api.py`** : tests fonctionnels des endpoints (`/health`, `/predict`, `/test-prediction`, `/predict-bulk`)
- **`tests/test_smoke.py`** : test de base du modèle (chargement et inférence)

## Intégration Continue (CI)

Une CI GitHub Actions est configurée dans `.github/workflows/ci.yml` :

- Déclenchement sur `push` (branches `main`, `develop`) et `pull_request` vers `main`.
- Environnement : `ubuntu-latest`, Python 3.10.
- Étapes :
  - Installation des dépendances (requirements + pytest + psycopg2-binary).
  - Exécution de la suite de tests `pytest -q`.

Tous les commits doivent passer la CI avant d’être fusionnés sur `main`.


### Exécuter les tests
 une fois lance l'API, dans autre terminal
```bash
pytest --cov=main_api --cov-report=term-missing
```
# Test santé
curl http://127.0.0.1:8000/health

# Test prédiction
curl -X POST http://127.0.0.1:8000/test-prediction

# Tests unitaires
pytest --cov=main_api --cov-report=term-missing


# Explication
- `probabilite_abandon` : probabilité (%) que l’employé quitte l’entreprise.
- `seuil_applique` : seuil (%) à partir duquel on considère le risque comme « Élevé ».
- `confiance_modele` : 100 - probabilite_abandon (probabilité de rester).
- `details` : informations résumées utiles pour l’interprétation métier (salaire, ancienneté, satisfaction…).

### Règle de décision appliquée

- Si `probabilite_abandon` ≥ `seuil_applique` → **Risque Élevé**
- Si `probabilite_abandon` < `seuil_applique` → **Risque Faible**


### Résultats

- **11 tests exécutés** avec succès ✅
- **Couverture** : **77 %** sur `main_api.py` (244 lignes exécutables)
- **Lignes non couvertes** : Principalement branches d'erreur rares et bloc de lancement (`if __name__ == "__main__":`)
- **Fonctions critiques** : 100% couvertes (prétraitement, prédiction, endpoints)

---

## 🗄️ Base de Données PostgreSQL

### Tables

#### `employees`
Enregistre chaque employé prédit.

```sql
CREATE TABLE employees (
  id SERIAL PRIMARY KEY,
  age INT,
  education_level INT,
  distance_from_home FLOAT,
  ... (23 colonnes features)
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `predictions`
Résultats des prédictions.

```sql
CREATE TABLE predictions (
  id SERIAL PRIMARY KEY,
  employee_id INT REFERENCES employees(id),
  churn_probability FLOAT,
  risk_level VARCHAR(10),
  recommendation TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `audit_log`
Journal d'audit de tous les appels API.

```sql
CREATE TABLE audit_log (
  id SERIAL PRIMARY KEY,
  endpoint VARCHAR(50),
  status VARCHAR(10),
  error_message TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Voir `sql/create_db.sql` pour le script complet.**
## psql -U postgres -f sql/create_db.sql

---

## 🔄 Mise à Jour et Maintenance du Modèle

Le modèle est stocké dans le dossier `models/` sous forme de fichiers `.pkl` sérialisés.

### Protocole de mise à jour

1. **Réentraîner le modèle** dans le notebook (`Projet-4-Churn.ipynb`)
   - Charger le dataset (`dataset_final.csv`)
   - Exécuter le pipeline de prétraitement
   - Entraîner et optimiser le modèle

2. **Exporter les artefacts** dans `models/`
   - `lr_model_opt.pkl` (modèle Logistic Regression)
   - `scaler.pkl` (StandardScaler)
   - `seuil_opt.pkl` (seuil optimal)

3. **Redémarrer l'API**
   ```bash
   python main_api.py
   ```
   L'API charge automatiquement les nouveaux fichiers au démarrage.

4. **Valider avec les tests**
   ```bash
   pytest --cov=main_api --cov-report=term-missing
   ```
   Tous les tests doivent passer pour garantir la conformité.

5. **Consulter l'audit**
   La table `audit_log` enregistre tous les appels (succès et erreurs), facilitant le suivi et le debugging en production.

### Fréquence recommandée

- **Vérification mensuelle** : Analyser la dérive des prédictions (compare prédictions vs réalité)
- **Réentraînement trimestriel** : Avec nouvelles données si disponibles
- **Monitoring continu** : Via `audit_log` et stats de couverture

---

## 🔐 Sécurité et Confidentialité

✅ Les données ne sont pas stockées au-delà de la session API  
✅ Les prédictions sont faites en temps réel (no caching)  
✅ Aucune sauvegarde d'informations sensibles en dehors de PostgreSQL  
✅ Application open-source et auditable  
✅ Connexion PostgreSQL sécurisée (variables d'environnement)

---

## 🛠️ Stack Technique

| Composant | Technologie |
|---|---|
| **Backend API** | FastAPI 0.104+ |
| **Serveur ASGI** | Uvicorn |
| **Machine Learning** | Scikit-learn (Logistic Regression) |
| **Données** | Pandas, NumPy, Joblib |
| **Base de données** | PostgreSQL 12+ |
| **Tests** | Pytest, pytest-cov |
| **Environnement** | Python 3.10+ |

---

## 📁 Structure du Projet

```
projet-4-churn-prediction-hr/
├── main_api.py                    # API FastAPI principale
├── Projet-4-Churn.ipynb          # Notebook ML (entraînement)
├── models/                         # Artefacts du modèle
│   ├── lr_model_opt.pkl
│   ├── scaler.pkl
│   └── seuil_opt.pkl
├── sql/
│   └── create_db.sql              # Script création BDD
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_smoke.py
├── requirements.txt               # Dépendances Python
├── README.md                      # Ce fichier
├── API_GUIDE.md                   # Guide détaillé des endpoints
├── MAINTENANCE.md                 # Guide de maintenance et mise à jour
└── dataset_final.csv             # Dataset d'entraînement
```

---

## 📚 Documentation Complète

- **`API_GUIDE.md`** : Résumé des endpoints et exemples d'utilisation
- **`MAINTENANCE.md`** : Protocole détaillé de mise à jour du modèle
- **Swagger interactif** : `http://127.0.0.1:8000/docs` (une fois l'API lancée)

---

## 🚀 Améliorations Futures Possibles

- [ ] Intégration SHAP pour explicabilité des prédictions
- [ ] Historique des prédictions par employé (dashboard)
- [ ] Upload CSV pour prédictions en batch
- [ ] Graphiques d'analyse et KPIs
- [ ] Authentification utilisateur (OAuth2)
- [ ] Alertes email pour cas critiques
- [ ] Containerisation Docker
- [ ] CI/CD avec GitHub Actions

---

## 👨‍💻 Auteur

**José Bravo** - Data Scientist | Machine Learning Engineer

---

## 📄 Licence

MIT License - Libre d'utilisation et modification

---

**Dernière mise à jour** : Mars 2026  
**Version** : 1.0.0  
**Status** : ✅ Production Ready
