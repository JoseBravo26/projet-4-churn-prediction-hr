---
title: Prédicteur de Churn des Employés
emoji: 👥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.6.0"
python_version: "3.10"
app_file: app.py
pinned: false
---
👥 Prédicteur de Churn des Employés
📊 Description
Application de machine learning qui prédit si un employé risque de quitter l'entreprise en fonction de ses données personnelles, professionnelles et de satisfaction au travail.

🎯 Objectif
Identifier les employés ayant un risque élevé de départ pour mettre en place des stratégies de rétention proactives.

📈 Caractéristiques du Modèle
Algorithme : Logistic Regression (Optimisé)

Dataset d'entraînement : 1 470 employés

Nombre de features : 23 variables prédictives

Précision : ~95%

Seuil optimal : Configuré pour maximiser le recall

📝 Variables d'Entrée
👤 Informations Personnelles
Âge (18-65 ans)

Niveau d'éducation (1-5)

Distance domicile-travail (km)

💼 Expérience et Trajectoire
Expériences précédentes

Années d'expérience totale

Années dans l'entreprise

Années au poste actuel

📊 Évaluation et Performance
Évaluation précédente (1-4)

Évaluation actuelle (1-4)

Niveau hiérarchique (1-5)

Employés sous responsabilité

😊 Satisfaction au Travail (1-4)
Satisfaction de l'environnement

Satisfaction de la nature du travail

Satisfaction de l'équipe

Satisfaction de l'équilibre vie-travail

💰 Compensation et Avantages
Revenu mensuel (€)

Dernier augmentation salaire (%)

Heures supplémentaires (Oui/Non)

Participation plan actions (PEE)

Formations complétées

🚀 Progression et Carrière
Années depuis dernière promotion

Années sous responsable actuel

🚀 Comment Utiliser
Étapes Simples
Remplis tous les champs avec les informations de l'employé

Clique sur "Prédire le Risque de Churn"

Consulte le résultat :

Niveau de risque (Élevé/Faible)

Probabilité de churn (%)

Recommandations d'action

📈 Interprétation des Résultats
🔴 Risque Élevé : Probabilité de départ > seuil optimal

📌 Action recommandée : Intervention immédiate (entretien RH, augmentation, formation, promotion)

🟢 Faible Risque : Probabilité de départ < seuil optimal

📌 Action recommandée : Maintenir la relation positive, surveiller satisfactions

🛠️ Technologie
Stack Technique
Python 3.8+

Scikit-learn : Machine Learning (Logistic Regression)

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
git clone https://huggingface.co/spaces/TON_USERNAME/churn-predictor](https://huggingface.co/spaces/josibra/churn-predictor
cd churn-predictor

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
python app.py

# 5. Accéder à l'interface
# http://localhost:7860
🌐 Accès en Ligne
L'application est disponible sur Hugging Face Spaces :
[https://huggingface.co/spaces/TON_USERNAME/churn-predictor](https://huggingface.co/spaces/josibra/churn-predictor)
🎓 Données d'Entraînement
Source : Système d'Information Ressources Humaines (SIRH)

Taille : 1 470 employés

Caractéristiques : 23 variables

Target : a_quitte_l_entreprise (Oui/Non)

Déséquilibre : Classes relativement équilibrées

🔐 Sécurité et Confidentialité
✅ Les données ne sont pas stockées sur le serveur

✅ Les prédictions sont faites en temps réel

✅ Aucune sauvegarde d'informations sensibles

✅ Application publique et entièrement open-source

📞 Support et Contribution
Documentation
Gradio Docs

HF Spaces Docs

Scikit-learn Docs

Fichiers Modèle
Localisation : models/ folder

lr_model_opt.pkl - Modèle Logistic Regression optimisé

scaler.pkl - StandardScaler pour normalisation

seuil_opt.pkl - Seuil optimal pour prédictions

🚀 Améliorations Futures Possibles
 Intégration SHAP pour explicabilité des prédictions

 Historique des prédictions par employé

 Upload CSV pour prédictions en batch

 Graphiques d'analyse et dashboards

 Authentification utilisateur

 Alertes email pour cas critiques

 API REST pour intégration externe

👨‍💻 Auteur
José Bravo - Data Scientist | Machine Learning Engineer

📄 Licence
MIT License - Libre d'utilisation et modification

Dernière mise à jour : Mars 2026
Version : 1.0.0
Status : ✅ Production Ready
