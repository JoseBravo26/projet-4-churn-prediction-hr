import pytest
import pandas as pd
from main_api import pretraiter_donnees, EmployeeInput, GenreEnum, EtatCivilEnum, \
    DepartementEnum, DomaineEtudeEnum, FrequenceDeplacementEnum

def build_employee(**overrides):
    """Crée un EmployeeInput valide, modifiable par overrides."""
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

def test_pretraiter_donnees_shape_and_columns():
    emp = build_employee()
    df = pretraiter_donnees(emp)

    # 1 ligne
    assert df.shape[0] == 1

    # Colonnes attendues minimales
    expected_cols = {
        "genre",
        "% augementation_salaire_precedente",
        "niveau_education",
        "est_marie",
        "poste_level",
        "freq_deplacement_level",
        "ratio_exp_entreprise",
        "revenu_par_age",
        "satisfaccion_media",
    }
    assert expected_cols.issubset(set(df.columns))

def test_experience_totale_inferieure_annees_entreprise_no_validation():
    emp = build_employee(experience_totale=3, annees_entreprise=5)
    assert emp.experience_totale == 3
    assert emp.annees_entreprise == 5


def test_annees_poste_superieures_annees_entreprise_raise():
    with pytest.raises(ValueError):
        build_employee(annees_poste=6, annees_entreprise=5)
