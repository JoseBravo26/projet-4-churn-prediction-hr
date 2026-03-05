"""
API FastAPI pour prédiction de Churn
Expose le modèle ML avec validation Pydantic robuste
"""

from unittest import result

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import joblib
import pandas as pd
import numpy as np
from enum import Enum
import uvicorn
import warnings
try:
    import psycopg2
except ImportError:
    psycopg2 = None

from psycopg2.extras import RealDictCursor

warnings.filterwarnings('ignore')

# ========================================
# 🔌 CONNEXION À LA BASE POSTGRES
# ========================================

def get_db_connection():
    if psycopg2 is None:
        raise RuntimeError("psycopg2 n'est pas disponible (ex. en CI), la connexion BDD est désactivée.")
    conn = psycopg2.connect(
        dbname="churn_predictor_db",
        user="churn_app",
        password="secure_password_123",
        host="localhost",
        port=5432,
    )
    conn.set_client_encoding("LATIN1")
    return conn



# ========================================
# 📦 CHARGER MODÈLE, SCALER ET SEUIL
# ========================================

try:
    modele = joblib.load('models/lr_model_opt.pkl')
    scaler = joblib.load('models/scaler.pkl')
    dict_seuil = joblib.load('models/seuil_opt.pkl')
    meilleur_seuil = dict_seuil['meilleur_seuil_lr']
    
    # Obtenir les features attendus
    if hasattr(scaler, 'feature_names_in_'):
        noms_features = list(scaler.feature_names_in_)
    else:
        noms_features = None
    
    print(f"✅ Modèle chargé avec succès")
    print(f"✅ Features détectés: {len(noms_features) if noms_features else 'Unknown'}")
    
except Exception as e:
    print(f"❌ Erreur au chargement: {str(e)}")
    modele = None
    scaler = None
    meilleur_seuil = None
    noms_features = None

# ========================================
# 📋 DÉFINIR LES ENUMS POUR VALIDATION
# ========================================

class GenreEnum(str, Enum):
    masculin = "Masculin"
    feminin = "Feminin"

class EtatCivilEnum(str, Enum):
    celibataire = "Celibataire"
    marie = "Marie(e)"
    divorce = "Divorce(e)"

class DepartementEnum(str, Enum):
    consulting = "Consulting"
    rh = "Ressources Humaines"
    it = "IT"
    finance = "Finance"
    marketing = "Marketing"

class DomaineEtudeEnum(str, Enum):
    entrepreneuriat = "Entrepreunariat"
    infra_cloud = "Infra & Cloud"
    marketing = "Marketing"
    ressources_humaines = "Ressources Humaines"
    transformation_digitale = "Transformation Digitale"
    autres = "Autres"

class FrequenceDeplacementEnum(str, Enum):
    rare = "Rare"
    modere = "Modere"
    frequent = "Fréquent"

# ========================================
# 🔧 MODÈLES PYDANTIC POUR VALIDATION
# ========================================

class EmployeeInput(BaseModel):
    """
    Modèle de validation pour les données d'entrée d'un employé
    """
    # Information personnelle
    age: int = Field(..., ge=18, le=70, description="Âge de l'employé (18-70)")
    genre: GenreEnum = Field(..., description="Genre de l'employé")
    etat_civil: EtatCivilEnum = Field(..., description="État civil")
    salaire: float = Field(..., gt=0, description="Salaire mensuel en euros")
    distance: float = Field(..., ge=0, description="Distance domicile-travail en km")
    
    # Entreprise et poste
    departement: DepartementEnum = Field(..., description="Département")
    domaine_etude: DomaineEtudeEnum = Field(..., description="Domaine d'étude")
    niveau_hierarchique: int = Field(..., ge=1, le=5, description="Niveau hiérarchique (1-5)")
    poste_freq_deplacement: FrequenceDeplacementEnum = Field(..., description="Fréquence de déplacement")
    
    # Expérience
    emplois_precedents: int = Field(..., ge=0, description="Nombre d'emplois antérieurs")
    experience_totale: float = Field(..., ge=0, description="Années d'expérience totale")
    annees_entreprise: float = Field(..., ge=0, description="Années dans l'entreprise")
    annees_poste: float = Field(..., ge=0, description="Années au poste actuel")
    annees_derniere_promotion: float = Field(..., ge=0, description="Années depuis dernière promotion")
    annees_responsable_actuel: float = Field(..., ge=0, description="Années sous responsable actuel")
    
    # Travail
    heures_semaine: float = Field(..., ge=1, le=70, description="Heures travaillées par semaine")
    heures_supplementaires: bool = Field(False, description="Travaille heures supplémentaires?")
    employes_supervision: int = Field(..., ge=0, description="Nombre d'employés supervisés")
    
    # Évaluations
    evaluation_precedente: int = Field(..., ge=1, le=4, description="Évaluation précédente (1-4)")
    evaluation_actuelle: int = Field(..., ge=1, le=4, description="Évaluation actuelle (1-4)")
    
    # Satisfaction
    satisfaction_environnement: int = Field(..., ge=1, le=4, description="Satisfaction environnement (1-4)")
    satisfaction_travail: int = Field(..., ge=1, le=4, description="Satisfaction type de travail (1-4)")
    satisfaction_equipe: int = Field(..., ge=1, le=4, description="Satisfaction équipe (1-4)")
    satisfaction_balance: int = Field(..., ge=1, le=4, description="Satisfaction équilibre vie-travail (1-4)")
    
    # Compensation
    augmentation_salaire: float = Field(..., ge=0, description="Dernière augmentation en %")
    participation_pee: int = Field(..., ge=0, description="Participation plan actions")
    formations_completees: int = Field(..., ge=0, description="Formations complétées")
    
    @validator('experience_totale')
    def valider_experience_totale(cls, v, values):
        if 'annees_entreprise' in values and v < values['annees_entreprise']:
            raise ValueError('Experience totale ne peut pas être inférieure aux années dans l\'entreprise')
        return v
    
    @validator('annees_poste')
    def valider_annees_poste(cls, v, values):
        if 'annees_entreprise' in values and v > values['annees_entreprise']:
            raise ValueError('Années au poste ne peut pas dépasser années dans l\'entreprise')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "genre": "Masculin",
                "etat_civil": "Marié(e)",
                "salaire": 5000,
                "distance": 5,
                "departement": "Consulting",
                "domaine_etude": "Transformation Digitale",
                "niveau_hierarchique": 2,
                "poste_freq_deplacement": "Modéré",
                "emplois_precedents": 3,
                "experience_totale": 8,
                "annees_entreprise": 5,
                "annees_poste": 2,
                "annees_derniere_promotion": 1,
                "annees_responsable_actuel": 3,
                "heures_semaine": 40,
                "heures_supplementaires": False,
                "employes_supervision": 0,
                "evaluation_precedente": 3,
                "evaluation_actuelle": 3,
                "satisfaction_environnement": 3,
                "satisfaction_travail": 3,
                "satisfaction_equipe": 3,
                "satisfaction_balance": 3,
                "augmentation_salaire": 15,
                "participation_pee": 1,
                "formations_completees": 2
            }
        }

class PredictionResponse(BaseModel):
    """
    Modèle de réponse pour la prédiction
    """
    prediction: str = Field(..., description="Prédiction: 'Risque Élevé' ou 'Risque Faible'")
    probabilite_abandon: float = Field(..., ge=0, le=100, description="Probabilité d'abandon en %")
    seuil_applique: float = Field(..., ge=0, le=100, description="Seuil appliqué en %")
    confiance_modele: float = Field(..., ge=0, le=100, description="Confiance du modèle en %")
    recommandation: str = Field(..., description="Recommandation d'action")
    details: Dict = Field(..., description="Détails additionnels")

class BulkPredictionRequest(BaseModel):
    """
    Modèle pour prédictions en masse
    """
    employes: List[EmployeeInput] = Field(..., description="Liste d'employés à prédire")

class BulkPredictionResponse(BaseModel):
    """
    Modèle de réponse pour prédictions en masse
    """
    total: int = Field(..., description="Nombre total de prédictions")
    predictions: List[PredictionResponse] = Field(..., description="Liste des prédictions")
    risque_eleve_count: int = Field(..., description="Nombre d'employés à risque élevé")
    taux_risque_eleve: float = Field(..., ge=0, le=100, description="Pourcentage d'employés à risque élevé")

# ========================================
# 🔧 FONCTION DE PRÉTRAITEMENT
# ========================================

def pretraiter_donnees(employee: EmployeeInput) -> pd.DataFrame:
    """
    Prétraite les données d'un employé selon le modèle
    """
    # Créer DataFrame avec les données brutes
    donnees = {
        'age': [employee.age],
        'revenu_mensuel': [employee.salaire],
        'nombre_experiences_precedentes': [employee.emplois_precedents],
        'nombre_heures_travailless': [employee.heures_semaine],
        'annee_experience_totale': [employee.experience_totale],
        'annees_dans_l_entreprise': [employee.annees_entreprise],
        'annees_dans_le_poste_actuel': [employee.annees_poste],
        'satisfaction_employee_environnement': [employee.satisfaction_environnement],
        'note_evaluation_precedente': [employee.evaluation_precedente],
        'niveau_hierarchique_poste': [employee.niveau_hierarchique],
        'satisfaction_employee_nature_travail': [employee.satisfaction_travail],
        'satisfaction_employee_equipe': [employee.satisfaction_equipe],
        'satisfaction_employee_equilibre_pro_perso': [employee.satisfaction_balance],
        'note_evaluation_actuelle': [employee.evaluation_actuelle],
        'heure_supplementaires': [1 if employee.heures_supplementaires else 0],
        'augementation_salaire_precedente': [employee.augmentation_salaire],
        'nombre_participation_pee': [employee.participation_pee],
        'nb_formations_suivies': [employee.formations_completees],
        'nombre_employee_sous_responsabilite': [employee.employes_supervision],
        'distance_domicile_travail': [employee.distance],
        'annees_depuis_la_derniere_promotion': [employee.annees_derniere_promotion],
        'annes_sous_responsable_actuel': [employee.annees_responsable_actuel],
        'genre': [1 if employee.genre == GenreEnum.feminin else 0],
        'est_marie': [1 if employee.etat_civil == EtatCivilEnum.marie else 0],
    }
    
    df = pd.DataFrame(donnees)
    
    # Feature Engineering
    df['revenu_par_age'] = df['revenu_mensuel'] / (df['age'] + 1)
    df['ratio_exp_entreprise'] = df['annee_experience_totale'] / (df['annees_dans_l_entreprise'] + 1)
    
    # Age groups (deleted)
    df['poste_level'] = df['niveau_hierarchique_poste']
    
    # Fréquence de déplacement
    deplacement_map = {'Rare': 1, 'Modéré': 2, 'Fréquent': 3}
    df['freq_deplacement_level'] = deplacement_map.get(employee.poste_freq_deplacement.value, 1)
    
    # Satisfaction moyenne
    satisfactions = [employee.satisfaction_environnement, employee.satisfaction_travail, 
                     employee.satisfaction_equipe, employee.satisfaction_balance]
    df['satisfaccion_media'] = np.mean(satisfactions)
    
    # One-hot encoding
    departements_possibles = ['Consulting', 'Ressources Humaines', 'IT', 'Finance', 'Marketing']
    for dept in departements_possibles:
        col_name = f'departement_{dept}'
        df[col_name] = 1 if employee.departement.value == dept else 0
    
    domaines_possibles = ['Entrepreunariat', 'Infra & Cloud', 'Marketing', 'Ressources Humaines', 'Transformation Digitale', 'Autres']
    for domaine in domaines_possibles:
        col_name = f'domaine_etude_{domaine}'
        df[col_name] = 1 if employee.domaine_etude.value == domaine else 0
    
    # Colonnes supplémentaires
    df['% augementation_salaire_precedente'] = df['augementation_salaire_precedente']
    df['niveau_education'] = 3
    
    # Sélectionner les colonnes attendues
    if noms_features is not None:
        colonnes_attendues = noms_features
    else:
        colonnes_attendues = [
            'genre', '% augementation_salaire_precedente', 'niveau_education', 'est_marie',
            'departement_Consulting', 'departement_Ressources Humaines', 'departement_IT',
            'departement_Finance', 'departement_Marketing',
            'domaine_etude_Entrepreunariat', 'domaine_etude_Infra & Cloud', 
            'domaine_etude_Marketing', 'domaine_etude_Ressources Humaines', 
            'domaine_etude_Transformation Digitale', 'domaine_etude_Autres',
            'poste_level', 'freq_deplacement_level', 
            'ratio_exp_entreprise', 'revenu_par_age', 'satisfaccion_media'
        ]
    
    df_final = pd.DataFrame()
    for col in colonnes_attendues:
        if col in df.columns:
            df_final[col] = df[col]
        else:
            df_final[col] = 0
    
    return df_final

# ========================================
# 🔮 FONCTION DE PRÉDICTION
# ========================================

def faire_prediction(employee: EmployeeInput) -> PredictionResponse:
    """
    Réalise une prédiction pour un employé ET enregistre le résultat en base.
    """
    if modele is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé correctement. Vérifiez les fichiers dans 'models'."
        )

    # 1️⃣ Prétraiter + prédire
    donnees_pretraitees = pretraiter_donnees(employee)
    donnees_normalisees = scaler.transform(donnees_pretraitees)
    probabilites = modele.predict_proba(donnees_normalisees)[0]
    prob_abandon = probabilites[1]

    # Appliquer le seuil
    prediction_flag = 1 if prob_abandon >= meilleur_seuil else 0
    pourcentage_abandon = prob_abandon * 100
    pourcentage_seuil = meilleur_seuil * 100

    if prediction_flag == 1:
        prediction_texte = "Risque Élevé"
        db_prediction = "RISQUE_ELEVE"
        recommandation = (
            "Intervention immédiate recommandée (augmentation, promotion, avantages, télétravail, etc.)."
        )
    else:
        prediction_texte = "Risque Faible"
        db_prediction = "RISQUE_FAIBLE"
        recommandation = "Employé plutôt stable. Maintenir la relation positive et surveiller l’évolution."

    # 2️⃣ Construire l'objet de réponse (toujours, même si la base échoue)
    reponse = PredictionResponse(
        prediction=prediction_texte,
        probabilite_abandon=round(pourcentage_abandon, 2),
        seuil_applique=round(pourcentage_seuil, 2),
        confiance_modele=round(max(probabilites) * 100, 2),
        recommandation=recommandation,
        details={
            "prob_rester": round(probabilites[0] * 100, 2),
            "prob_partir": round(probabilites[1] * 100, 2),
            "satisfaction_moyenne": round(
                np.mean([
                    employee.satisfaction_environnement,
                    employee.satisfaction_travail,
                    employee.satisfaction_equipe,
                    employee.satisfaction_balance,
                ]), 2,
            ),
            "salaire": employee.salaire,
            "departement": employee.departement.value,
            "anciennete_ans": employee.annees_entreprise,
        },
    )

    # 3️⃣ Enregistrer en base de données
    prediction_id = None  # pour le logging en cas d'erreur

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # 3a. Employees
        satisfaction_moy = float(
            np.mean(
                [
                    employee.satisfaction_environnement,
                    employee.satisfaction_travail,
                    employee.satisfaction_equipe,
                    employee.satisfaction_balance,
                ]
            ) / 4.0
        )

        cur.execute(
            """
            INSERT INTO employees (age, genre, salaire, anciennete, satisfaction, turnover)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            RETURNING id;
            """,
            (
                employee.age,
                employee.genre.value,
                float(employee.salaire),
                float(employee.annees_entreprise),
                satisfaction_moy,
                bool(prediction_flag),
            ),
        )
        result = cur.fetchone()
        if result:
            emp_id = result[0]
        else:
            cur.execute("SELECT id FROM employees ORDER BY id DESC LIMIT 1;")
            emp_id = cur.fetchone()[0]

        # 3b. Predictions
        cur.execute(
            """
            INSERT INTO predictions (emp_id, prediction, probability)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            RETURNING id;
            """,
            (emp_id, db_prediction, float(prob_abandon)),
        )
        result = cur.fetchone()
        if result:
            prediction_id = result[0]
        else:
            cur.execute("SELECT id FROM predictions ORDER BY id DESC LIMIT 1;")
            prediction_id = cur.fetchone()[0]

        # 3c. Audit log SUCCESS
        cur.execute(
            """
            INSERT INTO audit_log (prediction_id, action, status)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
            """,
            (prediction_id, "PREDICT", "SUCCESS"),
        )

        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Données enregistrées: emp_id={emp_id}, pred_id={prediction_id}")

    except Exception as db_err:
        print(f"⚠️ Erreur enregistrement DB: {repr(db_err)}")

        # Essayer de logguer aussi l'erreur dans audit_log
        try:
            print("➡️ Tentative d'écriture du log d'erreur")

            if "conn" in locals() and conn:
                conn.rollback()
                cur = conn.cursor()
            else:
                conn = get_db_connection()
                cur = conn.cursor()
                # prediction_id reste None si on n'a pas réussi à l'obtenir

            cur.execute(
                """
                INSERT INTO audit_log (prediction_id, action, status)
                VALUES (%s, %s, %s);
                """,
                (prediction_id, "PREDICT", "ERROR"),
            )
            conn.commit()
            cur.close()
            conn.close()
            print("✅ Log d'erreur écrit dans audit_log")
        except Exception as log_err:
            print(f"⚠️ Erreur lors de l'écriture du log: {repr(log_err)}")
            try:
                if "conn" in locals() and conn:
                    conn.rollback()
            except:
                pass


    # 4️⃣ Retourner la réponse
    return reponse

# ========================================
# 🚀 CRÉER L'APPLICATION FASTAPI
# ========================================

app = FastAPI(
    title="API Prédiction Churn",
    description="API pour prédire le risque d'abandon des employés",
    version="1.0.0",
    contact={
        "name": "Support",
        "email": "support@example.com"
    }
)

# Ajouter CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# 📍 ENDPOINTS
# ========================================

@app.get("/", tags=["Info"])
async def root():
    """Endpoint racine - Bienvenue"""
    return {
        "message": "Bienvenue sur l'API de prédiction de Churn",
        "version": "1.0.0",
        "endpoints": {
            "prediction": "/predict",
            "bulk_prediction": "/predict-bulk",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Info"])
async def health_check():
    """Vérifier la santé de l'API et du modèle"""
    status_modele = "✅ Chargé" if modele is not None else "❌ Non chargé"
    status_scaler = "✅ Chargé" if scaler is not None else "❌ Non chargé"
    
    all_ok = modele is not None and scaler is not None
    
    return {
        "status": "healthy" if all_ok else "unhealthy",
        "modele": status_modele,
        "scaler": status_scaler,
        "seuil": f"{meilleur_seuil:.4f}" if meilleur_seuil else "N/A",
        "features_count": len(noms_features) if noms_features else 0
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
async def predict(employee: EmployeeInput):
    """
    Prédire le risque de churn pour UN employé
    
    - **age**: Âge de l'employé (18-70)
    - **genre**: Masculin ou Féminin
    - **salaire**: Salaire mensuel en euros
    - ... (voir le body pour tous les champs)
    
    Retourne une prédiction avec probabilité et recommandation
    """
    return faire_prediction(employee)

@app.post("/predict-bulk", response_model=BulkPredictionResponse, tags=["Prédiction"])
async def predict_bulk(request: BulkPredictionRequest):
    """
    Prédire le risque de churn pour PLUSIEURS employés à la fois
    
    Accepte une liste d'employés et retourne:
    - Liste de toutes les prédictions
    - Statistiques globales (nombre à risque élevé, taux)
    """
    if len(request.employes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La liste des employés ne peut pas être vide"
        )
    
    if len(request.employes) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 1000 employés par requête pour éviter les surcharges"
        )
    
    predictions = []
    risque_eleve_count = 0
    
    for employee in request.employes:
        pred = faire_prediction(employee)
        predictions.append(pred)
        if pred.prediction == "Risque Élevé":
            risque_eleve_count += 1
    
    taux_risque = (risque_eleve_count / len(predictions)) * 100 if predictions else 0
    
    return BulkPredictionResponse(
        total=len(predictions),
        predictions=predictions,
        risque_eleve_count=risque_eleve_count,
        taux_risque_eleve=round(taux_risque, 2)
    )

@app.get("/info-modele", tags=["Info"])
async def info_modele():
    """Obtenir les informations du modèle"""
    return {
        "features_count": len(noms_features) if noms_features else 0,
        "features": noms_features[:10] if noms_features else None,
        "seuil_optimal": round(meilleur_seuil, 4) if meilleur_seuil else None,
        "modele_type": "Logistic Regression (Optimized)",
        "status": "ready" if modele is not None else "not_loaded"
    }

# ========================================
# 🧪 ENDPOINT DE TEST
# ========================================

@app.post("/test-prediction", response_model=PredictionResponse, tags=["Test"])
async def test_prediction():
    """
    Endpoint de TEST - Prédit avec des valeurs par défaut
    Utilise pour vérifier que l'API fonctionne correctement
    """
    test_employee = EmployeeInput(
        age=35,
        genre=GenreEnum.masculin,
        etat_civil=EtatCivilEnum.marie,
        salaire=5000,
        distance=5,
        departement=DepartementEnum.consulting,
        domaine_etude=DomaineEtudeEnum.transformation_digitale,
        niveau_hierarchique=2,
        poste_freq_deplacement=FrequenceDeplacementEnum.modere,
        emplois_precedents=3,
        experience_totale=8,
        annees_entreprise=5,
        annees_poste=2,
        annees_derniere_promotion=1,
        annees_responsable_actuel=3,
        heures_semaine=40,
        heures_supplementaires=False,
        employes_supervision=0,
        evaluation_precedente=3,
        evaluation_actuelle=3,
        satisfaction_environnement=3,
        satisfaction_travail=3,
        satisfaction_equipe=3,
        satisfaction_balance=3,
        augmentation_salaire=15,
        participation_pee=1,
        formations_completees=2
    )
    
    return faire_prediction(test_employee)

# ========================================
# 🚀 LANCER L'API
# ========================================

if __name__ == "__main__":
    print("🚀 Démarrage de l'API FastAPI")
    print("📖 Documentation interactive: http://127.0.0.1:8000/docs")
    print("🧪 Tests: http://127.0.0.1:8000/test-prediction")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
