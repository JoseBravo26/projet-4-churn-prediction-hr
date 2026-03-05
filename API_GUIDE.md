# 📘 Guide API - Prédicteur de Churn

**Version** : 1.0.0  
**Base URL** : `http://127.0.0.1:8000`  
**Documentation interactive** : `http://127.0.0.1:8000/docs`

---

## 🎯 Vue d'ensemble

Cette API FastAPI prédiet le risque de départ d'un employé en fonction de ses données personnelles, professionnelles et de satisfaction. L'API enregistre toutes les prédictions et les appels dans une base PostgreSQL.

---

## 🔐 Authentification

Actuellement, aucune authentification n'est requise. À implémenter selon besoin.

---

## 📋 Format de réponse standard

Toutes les réponses sont au format JSON.

### Réponse de succès (HTTP 200)

```json
{
  "risk_level": "Élevé" | "Faible",
  "churn_probability": 0.75,
  "threshold_applied": 0.5,
  "recommendation": "string"
}
```

### Réponse d'erreur (HTTP 4xx/5xx)

```json
{
  "detail": "Message d'erreur descriptif"
}
```

---

## 🔍 Endpoints détaillés

---

### 1. GET `/health`

**Description** : Vérifie l'état de l'API, du modèle et de la base de données.

**Paramètres** : Aucun

**Réponse** (HTTP 200) :

```json
{
  "status": "OK",
  "model_loaded": true,
  "scaler_loaded": true,
  "threshold_loaded": true,
  "database_connected": true
}
```

**Cas d'erreur** (HTTP 500) :

```json
{
  "status": "ERROR",
  "detail": "Modèle non chargé"
}
```

**Exemple cURL** :

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

---

### 2. POST `/predict`

**Description** : Prédit le risque de churn pour un employé individuel.

**Content-Type** : `application/json`

**Body** (EmployeeInput) :

```json
{
  "age": 35,
  "education_level": 3,
  "distance_from_home": 15.5,
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
  "monthly_revenue": 3500.0,
  "last_raise_percentage": 11.0,
  "overtime": false,
  "stock_option_level": 1,
  "trainings_completed": 2,
  "years_since_promotion": 2,
  "years_with_current_manager": 2
}
```

**Réponse** (HTTP 200 - Prédiction réussie) :

```json
{
  "risk_level": "Faible",
  "churn_probability": 0.15,
  "threshold_applied": 0.5,
  "recommendation": "Maintenir la relation positive, surveiller satisfactions"
}
```

**Autre réponse possible** :

```json
{
  "risk_level": "Élevé",
  "churn_probability": 0.75,
  "threshold_applied": 0.5,
  "recommendation": "Intervention immédiate (entretien RH, augmentation, formation, promotion)"
}
```

**Cas d'erreur** (HTTP 422 - Validation) :

```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "ensure this value is greater than or equal to 18",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

**Exemple cURL** :

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Exemple Python (requests)** :

```python
import requests

url = "http://127.0.0.1:8000/predict"
data = {
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
    "overtime": False,
    "stock_option_level": 1,
    "trainings_completed": 2,
    "years_since_promotion": 2,
    "years_with_current_manager": 2
}

response = requests.post(url, json=data)
print(response.json())
```

---

### 3. POST `/test-prediction`

**Description** : Teste rapidement l'API avec un employé fictif par défaut.

**Paramètres** : Aucun  
**Body** : Aucun

**Réponse** (HTTP 200) :

```json
{
  "risk_level": "Faible",
  "churn_probability": 0.22,
  "threshold_applied": 0.5,
  "recommendation": "Maintenir la relation positive, surveiller satisfactions"
}
```

**Cas d'erreur** (HTTP 500) :

```json
{
  "detail": "Erreur lors de la prédiction de test"
}
```

**Exemple cURL** :

```bash
curl -X POST "http://127.0.0.1:8000/test-prediction"
```

**Exemple Python** :

```python
import requests

response = requests.post("http://127.0.0.1:8000/test-prediction")
print(response.json())
```

---

### 4. POST `/predict-bulk`

**Description** : Prédit le risque pour une liste d'employés et retourne des statistiques globales.

**Content-Type** : `application/json`

**Body** (BulkPredictionRequest) :

```json
{
  "employees": [
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
    },
    {
      "age": 45,
      "education_level": 4,
      "distance_from_home": 25,
      "experience_count": 3,
      "total_experience": 20,
      "years_in_company": 8,
      "years_in_role": 4,
      "previous_evaluation": 4,
      "current_evaluation": 4,
      "hierarchy_level": 3,
      "subordinates_count": 5,
      "environment_satisfaction": 2,
      "job_satisfaction": 2,
      "team_satisfaction": 3,
      "worklife_balance_satisfaction": 1,
      "monthly_revenue": 5000,
      "last_raise_percentage": 8,
      "overtime": true,
      "stock_option_level": 2,
      "trainings_completed": 5,
      "years_since_promotion": 3,
      "years_with_current_manager": 4
    }
  ]
}
```

**Réponse** (HTTP 200) :

```json
{
  "predictions": [
    {
      "risk_level": "Faible",
      "churn_probability": 0.15,
      "threshold_applied": 0.5,
      "recommendation": "Maintenir la relation positive, surveiller satisfactions"
    },
    {
      "risk_level": "Élevé",
      "churn_probability": 0.78,
      "threshold_applied": 0.5,
      "recommendation": "Intervention immédiate (entretien RH, augmentation, formation, promotion)"
    }
  ],
  "statistics": {
    "total_processed": 2,
    "high_risk_count": 1,
    "low_risk_count": 1,
    "high_risk_percentage": 50.0
  }
}
```

**Cas d'erreur** (HTTP 422) :

```json
{
  "detail": "Au moins 1 employé est requis"
}
```

**Exemple cURL** :

```bash
curl -X POST "http://127.0.0.1:8000/predict-bulk" \
  -H "Content-Type: application/json" \
  -d '{
    "employees": [
      {
        "age": 35,
        "education_level": 3,
        ...
      }
    ]
  }'
```

---

## 📊 Codes de réponse HTTP

| Code | Signification | Exemple |
|---|---|---|
| **200** | Succès | Prédiction réussie |
| **422** | Erreur de validation | Champ manquant ou invalide |
| **500** | Erreur serveur | Modèle non chargé, BDD inaccessible |

---

## 📈 Interprétation des résultats

### `risk_level`
- **"Élevé"** : Probabilité de départ > seuil optimal (50%)
  - **Action** : Intervention immédiate requise (entretien RH, augmentation, formation, promotion)
  
- **"Faible"** : Probabilité de départ ≤ seuil optimal
  - **Action** : Maintenir la relation positive, surveiller les satisfactions

### `churn_probability`
Probabilité exprimée entre 0.0 (aucun risque) et 1.0 (risque maximal).

### `threshold_applied`
Le seuil utilisé pour la classification (par défaut 0.5). Les prédictions respectent ce seuil.

### `recommendation`
Recommandation d'action basée sur le risque identifié.

---

## 🔄 Exemples d'intégration

### Python - Prédiction unique

```python
import requests
import json

url = "http://127.0.0.1:8000/predict"

employee = {
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
    "overtime": False,
    "stock_option_level": 1,
    "trainings_completed": 2,
    "years_since_promotion": 2,
    "years_with_current_manager": 2
}

response = requests.post(url, json=employee)
result = response.json()

if result["risk_level"] == "Élevé":
    print(f"⚠️ ALERTE : Cet employé a {result['churn_probability']*100:.1f}% de risque de départ")
    print(f"Action : {result['recommendation']}")
else:
    print(f"✅ Employé stable : {result['churn_probability']*100:.1f}% de risque")
```

### Python - Prédictions en masse

```python
import requests
import pandas as pd

url = "http://127.0.0.1:8000/predict-bulk"

# Charger des données depuis CSV
df = pd.read_csv("employees.csv")
employees_list = df.to_dict('records')

payload = {"employees": employees_list}

response = requests.post(url, json=payload)
result = response.json()

# Afficher les statistiques
stats = result["statistics"]
print(f"Total traité : {stats['total_processed']}")
print(f"Risque élevé : {stats['high_risk_count']} ({stats['high_risk_percentage']:.1f}%)")
print(f"Risque faible : {stats['low_risk_count']}")

# Enregistrer les résultats
with open("predictions_result.json", "w") as f:
    json.dump(result, f, indent=2)
```

---

## ⚠️ Gestion des erreurs courantes

### Erreur : "age : ensure this value is greater than or equal to 18"
**Cause** : L'âge est inférieur à 18 ans  
**Solution** : Vérifier que l'âge est entre 18 et 65

### Erreur : "Cannot connect to database"
**Cause** : PostgreSQL n'est pas accessible  
**Solution** : Vérifier que PostgreSQL est lancé et la base existe (`churn_predictor_db`)

### Erreur : "Model not loaded"
**Cause** : Les fichiers `.pkl` ne sont pas dans le dossier `models/`  
**Solution** : Vérifier que `lr_model_opt.pkl`, `scaler.pkl` et `seuil_opt.pkl` existent

---

## 📞 Support

Pour des questions ou des rapports de bug :
- Consulter la documentation interactive : `http://127.0.0.1:8000/docs`
- Vérifier les logs : `http://127.0.0.1:8000/audit_log` (via BDD)
- Relancer l'API : `python main_api.py`

---

**Dernière mise à jour** : Mars 2026
