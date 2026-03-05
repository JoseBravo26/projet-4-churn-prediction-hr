# 🔧 Guide de Maintenance et Mise à Jour du Modèle

**Version** : 1.0.0  
**Audience** : Data Scientists, DevOps, Administrateurs  
**Fréquence de révision** : Trimestrielle

---

## 📋 Table des matières

1. [Protocole de mise à jour du modèle](#protocole-de-mise-à-jour)
2. [Monitoring et dérive du modèle](#monitoring-et-dérive)
3. [Versioning et traçabilité](#versioning-et-traçabilité)
4. [Rollback en cas de problème](#rollback)
5. [Checklist de déploiement](#checklist)
6. [FAQ et dépannage](#faq)

---

## 🔄 Protocole de mise à jour du modèle

### Contexte

Le modèle est stocké dans le dossier `models/` sous forme de fichiers `.pkl` sérialisés :

- **`lr_model_opt.pkl`** : Modèle Logistic Regression entraîné
- **`scaler.pkl`** : StandardScaler pour normalisation des features
- **`seuil_opt.pkl`** : Seuil optimal pour classification (par défaut 0.5)

L'API FastAPI charge ces fichiers au démarrage via `joblib.load()`.

### Étape 1 : Collecte et Préparation des Données

**Quand ?** Avant chaque réentraînement (trimestriel recommandé)

**Fichier** : `Projet-4-Churn.ipynb`

**Actions** :

1. Charger le dataset d'entraînement
   ```python
   import pandas as pd
   df = pd.read_csv("dataset_final.csv")
   print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
   ```

2. Vérifier la qualité des données
   ```python
   # Chercher les valeurs manquantes
   missing_values = df.isnull().sum()
   print(f"Valeurs manquantes : \n{missing_values[missing_values > 0]}")
   
   # Chercher les doublons
   duplicates = df.duplicated().sum()
   print(f"Doublons : {duplicates}")
   ```

3. Appliquer le même prétraitement qu'à l'entraînement initial
   - Encodage des variables catégoriques
   - Normalisation des features numériques
   - Gestion des outliers (si applicable)

### Étape 2 : Entraînement du Nouveau Modèle

**Quand ?** Une fois les données validées

**Actions** :

1. Diviser les données en train/test
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   ```

2. Entraîner la Logistic Regression
   ```python
   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
   model.fit(X_train_scaled, y_train)
   ```

3. Évaluer le modèle sur l'ensemble de test
   ```python
   from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
   
   y_pred = model.predict(X_test_scaled)
   y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
   
   print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
   print(f"Precision : {precision_score(y_test, y_pred):.4f}")
   print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
   print(f"AUC-ROC   : {roc_auc_score(y_test, y_pred_proba):.4f}")
   ```

4. Calculer/optimiser le seuil optimal
   ```python
   from sklearn.metrics import confusion_matrix
   
   # Chercher le seuil qui maximise recall (recall = VP / (VP + FN))
   thresholds = np.linspace(0, 1, 101)
   best_threshold = 0.5
   best_recall = 0
   
   for threshold in thresholds:
       y_pred_thresh = (y_pred_proba >= threshold).astype(int)
       recall = recall_score(y_test, y_pred_thresh)
       if recall > best_recall:
           best_recall = recall
           best_threshold = threshold
   
   print(f"Seuil optimal : {best_threshold}")
   print(f"Recall maximum : {best_recall}")
   ```

### Étape 3 : Validation du Nouveau Modèle

**Actions** :

1. Comparer avec l'ancien modèle
   ```python
   # Charger l'ancien modèle
   import joblib
   old_model = joblib.load("models/lr_model_opt.pkl")
   
   # Prédictions de l'ancien modèle
   old_pred_proba = old_model.predict_proba(X_test_scaled)[:, 1]
   old_acc = accuracy_score(y_test, (old_pred_proba >= 0.5).astype(int))
   
   # Comparer
   new_acc = accuracy_score(y_test, y_pred)
   improvement = (new_acc - old_acc) * 100
   print(f"Ancien modèle : {old_acc:.4f}")
   print(f"Nouveau modèle : {new_acc:.4f}")
   print(f"Amélioration : {improvement:.2f}%")
   ```

2. Documenter les métriques
   - Accuracy, Precision, Recall, AUC-ROC
   - Seuil optimal
   - Matrice de confusion
   - Temps d'entraînement

3. **Décision**
   - Si nouvelle accuracy > ancienne accuracy **+2%** → Valider et passer à l'étape 4
   - Sinon → Revérifier les données ou la préparation, redémarrer l'étape 2

### Étape 4 : Export des Fichiers du Modèle

**Actions** :

1. Sauvegarder le nouveau modèle
   ```python
   import joblib
   joblib.dump(model, "models/lr_model_opt_new.pkl")
   ```

2. Sauvegarder le scaler (utilisé à l'entraînement)
   ```python
   joblib.dump(scaler, "models/scaler_new.pkl")
   ```

3. Sauvegarder le seuil optimal
   ```python
   joblib.dump(best_threshold, "models/seuil_opt_new.pkl")
   ```

4. Vérifier que les fichiers ont bien été créés
   ```bash
   ls -lh models/
   ```

### Étape 5 : Test en Environnement de Staging

**Actions** :

1. Créer une copie de sauvegarde des fichiers actuels
   ```bash
   mkdir -p models/backup_$(date +%Y%m%d_%H%M%S)
   cp models/lr_model_opt.pkl models/backup_*/
   cp models/scaler.pkl models/backup_*/
   cp models/seuil_opt.pkl models/backup_*/
   ```

2. Copier les nouveaux fichiers
   ```bash
   cp models/lr_model_opt_new.pkl models/lr_model_opt.pkl
   cp models/scaler_new.pkl models/scaler.pkl
   cp models/seuil_opt_new.pkl models/seuil_opt.pkl
   ```

3. Redémarrer l'API en mode staging (si disponible)
   ```bash
   python main_api.py
   ```

4. Lancer les tests automatisés
   ```bash
   pytest --cov=main_api --cov-report=term-missing
   ```

   **Tous les tests doivent passer** ✅

5. Tester manuellement les endpoints clés
   ```bash
   # Test /health
   curl http://127.0.0.1:8000/health
   
   # Test /test-prediction
   curl -X POST http://127.0.0.1:8000/test-prediction
   
   # Test /predict avec un cas normal
   curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{ ... }'
   ```

### Étape 6 : Déploiement en Production

**Pré-déploiement** :

- ✅ Tous les tests passent
- ✅ API répond correctement en staging
- ✅ Prédictions sont cohérentes
- ✅ Backup de l'ancien modèle effectué

**Actions de déploiement** :

1. Si en production actuellement, enregistrer la version
   ```python
   import json
   from datetime import datetime
   
   version_info = {
       "version": "1.1.0",
       "deployed_at": datetime.now().isoformat(),
       "accuracy": 0.95,
       "recall": 0.88,
       "threshold": 0.5,
       "backup_path": "models/backup_20260303_143000"
   }
   
   with open("models/version.json", "w") as f:
       json.dump(version_info, f, indent=2)
   ```

2. Redémarrer l'API en production
   ```bash
   # Si utilisant systemd
   sudo systemctl restart churn-api
   
   # Ou manuellement
   python main_api.py &
   ```

3. Vérifier que l'API est accessible
   ```bash
   curl http://127.0.0.1:8000/health
   ```

4. Enregistrer dans la table `model_versions` (optionnel)
   ```sql
   INSERT INTO model_versions (
       version, algorithm, features_count, training_samples,
       accuracy, precision, recall, auc_roc, threshold_applied,
       model_path, scaler_path, threshold_path, deployed_at, is_active
   ) VALUES (
       '1.1.0', 'LogisticRegression', 23, 1470,
       0.95, 0.92, 0.88, 0.96, 0.5,
       'models/lr_model_opt.pkl', 'models/scaler.pkl', 'models/seuil_opt.pkl',
       NOW(), TRUE
   );
   ```

---

## 📊 Monitoring et Dérive du Modèle

### Indicateurs à suivre

1. **Performance des prédictions**
   - Comparer prédictions vs résultats réels (si données réelles disponibles)
   - Chercher une dérive du recall / precision

2. **Distribution des probabilités**
   ```sql
   SELECT 
       COUNT(*) as total,
       COUNT(CASE WHEN churn_probability > 0.5 THEN 1 END) as high_risk,
       AVG(churn_probability) as avg_probability
   FROM predictions
   WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE)
   ```

3. **Données d'entrée**
   - Vérifier que les features restent en plages normales
   - Chercher des valeurs extrêmes

### Seuils d'alerte

| Indicateur | Seuil d'alerte | Action |
|---|---|---|
| **Accuracy** | Baisse > 5% | Réentraîner le modèle |
| **Recall** | < 0.75 | Ajuster le seuil ou réentraîner |
| **Taux de high-risk** | Dépasse 70% | Analyser les features d'entrée |
| **Erreurs API** | > 5% de 500 errors | Vérifier BDD et fichiers modèles |

### Requêtes de monitoring (PostgreSQL)

```sql
-- Statistiques globales
SELECT * FROM v_prediction_statistics;

-- Dernières prédictions
SELECT * FROM v_latest_predictions
ORDER BY prediction_date DESC
LIMIT 20;

-- Résumé des appels API
SELECT * FROM v_api_summary;

-- Erreurs récentes
SELECT endpoint, status_code, error_message, created_at
FROM audit_log
WHERE status = 'ERROR'
ORDER BY created_at DESC
LIMIT 10;
```

---

## 🏷️ Versioning et Traçabilité

### Naming Convention

```
models/
├── lr_model_opt.pkl          # Modèle courant en production
├── scaler.pkl                # Scaler courant
├── seuil_opt.pkl             # Seuil courant
├── version.json              # Métadonnées de la version active
└── backup_20260303_143000/   # Backups horodatées
    ├── lr_model_opt.pkl
    ├── scaler.pkl
    └── seuil_opt.pkl
```

### Historique des versions

Utiliser la table `model_versions` pour tracer chaque déploiement :

```sql
SELECT * FROM model_versions ORDER BY deployed_at DESC;
```

---

## ↩️ Rollback en Cas de Problème

### Scénario 1 : Le nouveau modèle ne charge pas

```bash
# Restaurer depuis le backup
cp models/backup_20260303_143000/lr_model_opt.pkl models/lr_model_opt.pkl
cp models/backup_20260303_143000/scaler.pkl models/scaler.pkl
cp models/backup_20260303_143000/seuil_opt.pkl models/seuil_opt.pkl

# Redémarrer l'API
python main_api.py
```

### Scénario 2 : Les prédictions sont mauvaises

```sql
-- Revérifier les métriques dans audit_log
SELECT * FROM audit_log
WHERE created_at > '2026-03-03 14:30:00'
ORDER BY created_at DESC;

-- Vérifier si les erreurs augmentent
SELECT endpoint, status, COUNT(*) as count
FROM audit_log
WHERE created_at > '2026-03-03 14:30:00'
GROUP BY endpoint, status;
```

Si problème confirmé :

```bash
# Restaurer l'ancien modèle
cp models/backup_20260303_143000/* models/

# Redémarrer
python main_api.py

# Marquer l'ancienne version comme active à nouveau
UPDATE model_versions SET is_active = FALSE WHERE version = '1.1.0';
UPDATE model_versions SET is_active = TRUE WHERE version = '1.0.0';
```

---

## ✅ Checklist de Déploiement

Avant tout déploiement en production :

- [ ] Données collectées et validées
- [ ] Nouveau modèle entraîné et évalué
- [ ] Metrics améliorées ou équivalentes (accuracy, recall, AUC)
- [ ] Seuil optimal calculé
- [ ] Fichiers .pkl générés et vérifiés
- [ ] Backup de l'ancien modèle créé
- [ ] Tests unitaires passent (`pytest --cov`)
- [ ] Tests API manuels réussis (health, predict, predict-bulk)
- [ ] Versioning enregistré
- [ ] Équipe RH/métier informée du changement

---

## ❓ FAQ et Dépannage

### Q1 : Combien de temps faut-il pour réentraîner le modèle ?

**R** : Environ 5-10 minutes pour :
- Charger et prétraiter les données (1-2 min)
- Entraîner le modèle (2-3 min)
- Évaluer et valider (1-2 min)
- Exporter les fichiers (< 1 min)

### Q2 : Comment sais-je que le modèle a besoin d'être réentraîné ?

**R** : Réentraîner si :
- Accuracy baisse de > 5%
- Nouvelles données disponibles (mensuelles ou trimestrielles)
- Analyse des erreurs montre une tendance
- Dérive observée dans les distributions de features

### Q3 : Peut-on réentraîner le modèle sans redémarrer l'API ?

**R** : Non. L'API charge les fichiers `.pkl` au démarrage. Il faut redémarrer pour charger les nouveaux fichiers.

### Q4 : Que faire si un employé conteste une prédiction ?

**R** : 
1. Récupérer la prédiction exacte dans `predictions` table
2. Vérifier les features utilisées dans `employees` table
3. Reproduire la prédiction manuellement en notebook
4. Expliquer les factors en haute probabilité (via SHAP, optionnel)

### Q5 : Comment monitorer le modèle en production ?

**R** :
- Consulter `audit_log` pour les erreurs
- Vérifier `v_prediction_statistics` pour les tendances
- Établir une alerte si accuracy chute ou erreurs augmentent
- Revoir mensuellement les résultats vs résultats réels

---

## 📞 Contacts et Escalade

| Rôle | Responsabilité |
|---|---|
| **Data Scientist** | Entraîner et valider le modèle |
| **DevOps** | Déployer et monitorer l'API |
| **RH/Métier** | Fournir feedback sur les prédictions |

---

**Dernière mise à jour** : Mars 2026  
**Version du guide** : 1.0.0  
**Fréquence de révision** : Trimestrielle
