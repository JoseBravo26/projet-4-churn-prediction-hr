import os
import sys
from fastapi.testclient import TestClient

# Ajouter la racine du projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main_api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "unhealthy")
    assert "modele" in data
    assert "scaler" in data


def test_test_prediction_ok_and_has_fields():
    response = client.post("/test-prediction")
    assert response.status_code == 200
    data = response.json()

    for field in [
        "prediction",
        "probabilite_abandon",
        "seuil_applique",
        "confiance_modele",
        "recommandation",
        "details",
    ]:
        assert field in data

    for field in ["prob_rester", "prob_partir", "satisfaction_moyenne"]:
        assert field in data["details"]

from main_api import GenreEnum, EtatCivilEnum, DepartementEnum, \
    DomaineEtudeEnum, FrequenceDeplacementEnum

def build_payload(**overrides):
    base = dict(
        age=35,
        genre=GenreEnum.masculin.value,
        etat_civil=EtatCivilEnum.celibataire.value,
        salaire=5000,
        distance=5,
        departement=DepartementEnum.consulting.value,
        domaine_etude=DomaineEtudeEnum.transformation_digitale.value,
        niveau_hierarchique=2,
        poste_freq_deplacement=FrequenceDeplacementEnum.modere.value,
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
        formations_completees=2,
    )
    base.update(overrides)
    return base


def test_predict_single_valid_employee():
    payload = build_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ("Risque Élevé", "Risque Faible")
    assert 0 <= data["probabilite_abandon"] <= 100


def test_predict_missing_field_returns_422():
    payload = build_payload()
    payload.pop("age")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
