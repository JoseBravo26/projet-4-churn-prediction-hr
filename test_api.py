"""
Script de test para la API FastAPI
Prueba todos los endpoints con ejemplos reales
"""

import requests
import json
from typing import Dict, Any
import time

# ========================================
# ⚙️ CONFIGURACIÓN
# ========================================

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 10

# Colores para terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    END = '\033[0m'

# ========================================
# 🧪 FUNCIONES DE TEST
# ========================================

def print_test(titre: str):
    """Imprime el título de un test"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}🧪 {titre}{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

def print_success(message: str):
    """Imprime mensaje de éxito"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message: str):
    """Imprime mensaje de error"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message: str):
    """Imprime mensaje de advertencia"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message: str, value: Any = None):
    """Imprime información"""
    if value is not None:
        print(f"   {message}: {Colors.BLUE}{value}{Colors.END}")
    else:
        print(f"   {message}")

# ========================================
# 📋 DATOS DE PRUEBA
# ========================================

EMPLOYEE_NORMAL = {
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

EMPLOYEE_HIGH_RISK = {
    "age": 28,
    "genre": "Féminin",
    "etat_civil": "Célibataire",
    "salaire": 3000,
    "distance": 30,
    "departement": "IT",
    "domaine_etude": "Infra & Cloud",
    "niveau_hierarchique": 1,
    "poste_freq_deplacement": "Fréquent",
    "emplois_precedents": 5,
    "experience_totale": 5,
    "annees_entreprise": 1,
    "annees_poste": 1,
    "annees_derniere_promotion": 3,
    "annees_responsable_actuel": 1,
    "heures_semaine": 50,
    "heures_supplementaires": True,
    "employes_supervision": 0,
    "evaluation_precedente": 2,
    "evaluation_actuelle": 2,
    "satisfaction_environnement": 1,
    "satisfaction_travail": 1,
    "satisfaction_equipe": 2,
    "satisfaction_balance": 1,
    "augmentation_salaire": 0,
    "participation_pee": 0,
    "formations_completees": 0
}

EMPLOYEE_LOW_RISK = {
    "age": 45,
    "genre": "Masculin",
    "etat_civil": "Marié(e)",
    "salaire": 8000,
    "distance": 2,
    "departement": "Ressources Humaines",
    "domaine_etude": "Ressources Humaines",
    "niveau_hierarchique": 4,
    "poste_freq_deplacement": "Rare",
    "emplois_precedents": 2,
    "experience_totale": 15,
    "annees_entreprise": 10,
    "annees_poste": 5,
    "annees_derniere_promotion": 1,
    "annees_responsable_actuel": 5,
    "heures_semaine": 38,
    "heures_supplementaires": False,
    "employes_supervision": 5,
    "evaluation_precedente": 4,
    "evaluation_actuelle": 4,
    "satisfaction_environnement": 4,
    "satisfaction_travail": 4,
    "satisfaction_equipe": 4,
    "satisfaction_balance": 4,
    "augmentation_salaire": 20,
    "participation_pee": 10,
    "formations_completees": 5
}

# ========================================
# 🧪 TEST 1: Health Check
# ========================================

def test_health_check():
    """Test del endpoint /health"""
    print_test("TEST 1: Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"API Health: {data['status']}")
            print_info("Modèle", data.get('modele'))
            print_info("Scaler", data.get('scaler'))
            print_info("Features count", data.get('features_count'))
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Erreur de connexion: {str(e)}")
        return False

# ========================================
# 🧪 TEST 2: Test Prediction (default)
# ========================================

def test_prediction_default():
    """Test du endpoint /test-prediction avec valeurs par défaut"""
    print_test("TEST 2: Test Prediction (Valeurs par défaut)")
    
    try:
        response = requests.post(
            f"{BASE_URL}/test-prediction",
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Prédiction réussie")
            print_info("Prédiction", data['prediction'])
            print_info("Probabilité d'abandon", f"{data['probabilite_abandon']}%")
            print_info("Confiance du modèle", f"{data['confiance_modele']}%")
            print_info("Recommandation", data['recommandation'][:50] + "...")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            print_error(response.text)
            return False
            
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
        return False

# ========================================
# 🧪 TEST 3: Single Prediction (Normal)
# ========================================

def test_single_prediction():
    """Test du endpoint /predict avec un employé normal"""
    print_test("TEST 3: Single Prediction (Employé Normal)")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=EMPLOYEE_NORMAL,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Prédiction réussie")
            print_info("Âge", "35 ans")
            print_info("Salaire", "5000€")
            print_info("Prédiction", data['prediction'])
            print_info("Probabilité d'abandon", f"{data['probabilite_abandon']}%")
            print_info("Ancienneté", f"{EMPLOYEE_NORMAL['annees_entreprise']} ans")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            print_error(response.text)
            return False
            
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
        return False

# ========================================
# 🧪 TEST 4: High Risk Employee
# ========================================

def test_high_risk_employee():
    """Test avec un employé à risque élevé"""
    print_test("TEST 4: High Risk Employee")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=EMPLOYEE_HIGH_RISK,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Prédiction réussie")
            print_info("Profil", "Employé à risque potentiel")
            print_info("Âge", "28 ans")
            print_info("Ancienneté", "1 an seulement")
            print_info("Satisfaction environnement", "1/4 (Très bas)")
            print_info("Prédiction", data['prediction'])
            print_info("Probabilité d'abandon", f"{data['probabilite_abandon']}%")
            if data['prediction'] == "Risque Élevé":
                print_success("✓ Correctement identifié comme à risque élevé")
            else:
                print_warning("Cet employé devrait être à risque élevé")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
        return False

# ========================================
# 🧪 TEST 5: Low Risk Employee
# ========================================

def test_low_risk_employee():
    """Test avec un employé à faible risque"""
    print_test("TEST 5: Low Risk Employee")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=EMPLOYEE_LOW_RISK,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Prédiction réussie")
            print_info("Profil", "Employé stable et satisfait")
            print_info("Âge", "45 ans")
            print_info("Ancienneté", "10 ans")
            print_info("Satisfaction moyenne", "4/4 (Excellent)")
            print_info("Prédiction", data['prediction'])
            print_info("Probabilité d'abandon", f"{data['probabilite_abandon']}%")
            if data['prediction'] == "Risque Faible":
                print_success("✓ Correctement identifié comme à risque faible")
            else:
                print_warning("Cet employé devrait être à risque faible")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
        return False

# ========================================
# 🧪 TEST 6: Validation d'erreurs
# ========================================

def test_validation_errors():
    """Test de la validation d'erreurs Pydantic"""
    print_test("TEST 6: Validation d'Erreurs Pydantic")
    
    # Test 1: Âge invalide (trop bas)
    invalid_data_1 = EMPLOYEE_NORMAL.copy()
    invalid_data_1['age'] = 10
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data_1,
            timeout=TIMEOUT
        )
        
        if response.status_code == 422:
            print_success("✓ Âge invalide (10) rejeté correctement")
        else:
            print_warning(f"Comportement inattendu: {response.status_code}")
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
    
    # Test 2: Salaire invalide (négatif)
    invalid_data_2 = EMPLOYEE_NORMAL.copy()
    invalid_data_2['salaire'] = -1000
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data_2,
            timeout=TIMEOUT
        )
        
        if response.status_code == 422:
            print_success("✓ Salaire invalide (-1000) rejeté correctement")
        else:
            print_warning(f"Comportement inattendu: {response.status_code}")
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
    
    # Test 3: Genre invalide
    invalid_data_3 = EMPLOYEE_NORMAL.copy()
    invalid_data_3['genre'] = "Autre"
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data_3,
            timeout=TIMEOUT
        )
        
        if response.status_code == 422:
            print_success("✓ Genre invalide (Autre) rejeté correctement")
        else:
            print_warning(f"Comportement inattendu: {response.status_code}")
    except Exception as e:
        print_error(f"Erreur: {str(e)}")

# ========================================
# 🧪 TEST 7: Bulk Prediction
# ========================================

def test_bulk_prediction():
    """Test du endpoint /predict-bulk"""
    print_test("TEST 7: Bulk Prediction (5 employés)")
    
    bulk_data = {
        "employes": [
            EMPLOYEE_NORMAL,
            EMPLOYEE_HIGH_RISK,
            EMPLOYEE_LOW_RISK,
            EMPLOYEE_NORMAL.copy(),
            EMPLOYEE_HIGH_RISK.copy()
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict-bulk",
            json=bulk_data,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Prédictions en masse réussies: {data['total']} employés")
            print_info("Employés à risque élevé", data['risque_eleve_count'])
            print_info("Taux de risque élevé", f"{data['taux_risque_eleve']}%")
            print_info("Premiers résultats:")
            for i, pred in enumerate(data['predictions'][:3]):
                print(f"   {i+1}. {pred['prediction']} ({pred['probabilite_abandon']}%)")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
        return False

# ========================================
# 🧪 TEST 8: Info du Modèle
# ========================================

def test_model_info():
    """Test du endpoint /info-modele"""
    print_test("TEST 8: Informations du Modèle")
    
    try:
        response = requests.get(
            f"{BASE_URL}/info-modele",
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Informations du modèle récupérées")
            print_info("Type", data.get('modele_type'))
            print_info("Statut", data.get('status'))
            print_info("Nombre de features", data.get('features_count'))
            print_info("Seuil optimal", data.get('seuil_optimal'))
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
        return False

# ========================================
# 🚀 LANCER TOUS LES TESTS
# ========================================

def run_all_tests():
    """Lance tous les tests"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}🚀 SUITE DE TESTS API FASTAPI{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    # Vérifier la connexion
    print(f"\n{Colors.YELLOW}Vérification de la connexion à {BASE_URL}...{Colors.END}")
    time.sleep(1)
    
    results = []
    
    # Lancer tous les tests
    results.append(("Health Check", test_health_check()))
    results.append(("Test Prediction (Default)", test_prediction_default()))
    results.append(("Single Prediction", test_single_prediction()))
    results.append(("High Risk Employee", test_high_risk_employee()))
    results.append(("Low Risk Employee", test_low_risk_employee()))
    test_validation_errors()  # N'a pas de retour booléen
    results.append(("Bulk Prediction", test_bulk_prediction()))
    results.append(("Model Info", test_model_info()))
    
    # Résumé
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}📊 RÉSUMÉ DES TESTS{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{Colors.GREEN}✅ PASS{Colors.END}" if result else f"{Colors.RED}❌ FAIL{Colors.END}"
        print(f"{test_name}: {status}")
    
    print(f"\n{Colors.BLUE}Total: {passed}/{total} tests réussis{Colors.END}\n")
    
    if passed == total:
        print(f"{Colors.GREEN}🎉 TOUS LES TESTS RÉUSSIS!{Colors.END}\n")
    else:
        print(f"{Colors.RED}⚠️  Certains tests ont échoué{Colors.END}\n")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrompus par l'utilisateur{Colors.END}\n")
    except Exception as e:
        print(f"\n{Colors.RED}Erreur générale: {str(e)}{Colors.END}\n")
