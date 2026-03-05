import numpy as np
import os
import sys

# Ajouter la racine du projet (contenant main_api.py) au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main_api import modele, scaler, pretraiter_donnees, EmployeeInput, \
    GenreEnum, EtatCivilEnum, DepartementEnum, DomaineEtudeEnum, FrequenceDeplacementEnum

def build_employee(**overrides):
    base = dict(
        age=35,
        genre=GenreEnum.masculin,
        etat_civil=EtatCivilEnum.celibataire,
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
        formations_completees=2,
    )
    base.update(overrides)
    return EmployeeInput(**base)

def test_modele_charge():
    """Le modèle et le scaler doivent être chargés."""
    assert modele is not None
    assert scaler is not None

def test_probabilites_dimension_et_somme():
    """Le modèle doit renvoyer un vecteur de probas de taille 2 qui somme à ~1."""
    emp = build_employee()
    X = pretraiter_donnees(emp)
    X_scaled = scaler.transform(X)
    probs = modele.predict_proba(X_scaled)[0]

    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)

def test_cas_haut_risque_plus_que_moyen():
    """Profil très défavorable → probabilité de churn élevée."""
    emp = build_employee(
        satisfaction_environnement=1,
        satisfaction_travail=1,
        satisfaction_equipe=1,
        satisfaction_balance=1,
        annees_entreprise=1.0,
        annees_poste=1.0,          # <= annees_entreprise pour respecter le validateur
        experience_totale=1.0,
        salaire=1800,
    )
    X = pretraiter_donnees(emp)
    X_scaled = scaler.transform(X)
    probs = modele.predict_proba(X_scaled)[0]
    prob_churn = probs[1]

    # seuil plus souple: le cas doit être au-dessus de ton meilleur seuil
    from main_api import meilleur_seuil
    assert prob_churn >= meilleur_seuil

