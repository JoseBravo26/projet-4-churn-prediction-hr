import gradio as gr
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 📦 CHARGER MODÈLE, SCALER ET SEUIL
# ========================================
try:
    modele = joblib.load('models/lr_model_opt.pkl')
    scaler = joblib.load('models/scaler.pkl')
    dict_seuil = joblib.load('models/seuil_opt.pkl')
    meilleur_seuil = dict_seuil['meilleur_seuil_lr']
    
    print(f"✅ Modèle chargé")
    print(f"✅ Scaler chargé")
    print(f"✅ Seuil optimal : {meilleur_seuil:.4f}")
    
    # DÉTECTER LES FEATURES CORRECTS DU MODÈLE
    if hasattr(scaler, 'feature_names_in_'):
        noms_features = list(scaler.feature_names_in_)
        print(f"✅ Features du modèle ({len(noms_features)}): {noms_features[:10]}...")
    else:
        noms_features = None
        print(f"⚠️ Pas de feature_names_in_ détectés")
        
except FileNotFoundError as e:
    print(f"❌ Fichier manquant: {str(e)}")
    modele = None
    scaler = None
    meilleur_seuil = None
    noms_features = None
except Exception as e:
    print(f"❌ Erreur au chargement: {str(e)}")
    modele = None
    scaler = None
    meilleur_seuil = None
    noms_features = None

# ========================================
# 🔧 FONCTION DE PRÉTRAITEMENT DES DONNÉES
# ========================================
def pretraiter_donnees(age, salaire, emplois_precedents, heures_semaine, experience_totale,
                       annees_entreprise, annees_poste, satisfaction_environnement, evaluation_precedente, 
                       niveau_hierarchique, satisfaction_travail, satisfaction_equipe, satisfaction_balance, 
                       evaluation_actuelle, heures_supplementaires, augmentation_salaire, participation_pee, 
                       formations_completees, employes_supervision, distance, annees_derniere_promotion, 
                       annees_responsable_actuel, genre, etat_civil, departement, domaine_etude, poste_freq_deplacement):
    """
    Crée un DataFrame avec les features TRANSFORMÉES comme le modèle l'attend
    """
    
    # Créer DataFrame avec les données brutes
    donnees = {
        'age': [age],
        'revenu_mensuel': [salaire],
        'nombre_experiences_precedentes': [emplois_precedents],
        'nombre_heures_travailless': [heures_semaine],
        'annee_experience_totale': [experience_totale],
        'annees_dans_l_entreprise': [annees_entreprise],
        'annees_dans_le_poste_actuel': [annees_poste],
        'satisfaction_employee_environnement': [satisfaction_environnement],
        'note_evaluation_precedente': [evaluation_precedente],
        'niveau_hierarchique_poste': [niveau_hierarchique],
        'satisfaction_employee_nature_travail': [satisfaction_travail],
        'satisfaction_employee_equipe': [satisfaction_equipe],
        'satisfaction_employee_equilibre_pro_perso': [satisfaction_balance],
        'note_evaluation_actuelle': [evaluation_actuelle],
        'heure_supplementaires': [1 if heures_supplementaires else 0],
        'augementation_salaire_precedente': [augmentation_salaire],
        'nombre_participation_pee': [participation_pee],
        'nb_formations_suivies': [formations_completees],
        'nombre_employee_sous_responsabilite': [employes_supervision],
        'distance_domicile_travail': [distance],
        'annees_depuis_la_derniere_promotion': [annees_derniere_promotion],
        'annes_sous_responsable_actuel': [annees_responsable_actuel],
        'genre': [1 if genre == "Féminin" else 0],
        'est_marie': [1 if etat_civil == "Marié(e)" else 0],
        'departement': [departement],
        'domaine_etude': [domaine_etude],
        'freq_deplacement': [poste_freq_deplacement]
    }
    
    df = pd.DataFrame(donnees)
    
    # ========================================
    # FEATURE ENGINEERING
    # ========================================
    
    # 1. Créer les features numériques calculées
    df['revenu_par_age'] = df['revenu_mensuel'] / (df['age'] + 1)
    df['ratio_exp_entreprise'] = df['annee_experience_totale'] / (df['annees_dans_l_entreprise'] + 1)
    
    # 2. Créer les groupes d'âge (deleted)
    
    # 3. Créer le niveau de poste
    df['poste_level'] = df['niveau_hierarchique_poste']
    
    # 4. Créer le niveau de fréquence de déplacement
    if poste_freq_deplacement.lower() == 'rare':
        df['freq_deplacement_level'] = 1
    elif poste_freq_deplacement.lower() == 'modéré':
        df['freq_deplacement_level'] = 2
    else:  # Fréquent
        df['freq_deplacement_level'] = 3
    
    # 5. Satisfaction moyenne
    satisfactions = [satisfaction_environnement, satisfaction_travail, satisfaction_equipe, satisfaction_balance]
    df['satisfaccion_media'] = np.mean(satisfactions)
    
    # 6. One-hot encoding pour les catégories
    
    # Departements
    departements_possibles = ['Consulting', 'Ressources Humaines', 'IT', 'Finance', 'Marketing']
    for dept in departements_possibles:
        col_name = f'departement_{dept}'
        df[col_name] = 1 if departement == dept else 0
    
    # Domaines d'étude
    domaines_possibles = ['Entrepreunariat', 'Infra & Cloud', 'Marketing', 'Ressources Humaines', 'Transformation Digitale', 'Autres']
    for domaine in domaines_possibles:
        col_name = f'domaine_etude_{domaine}'
        df[col_name] = 1 if domaine_etude == domaine else 0
    
    # 7. Ajouter la colonne % (pour compatibilité des noms)
    df['% augementation_salaire_precedente'] = df['augementation_salaire_precedente']
    
    # 8. Ajouter niveau_education (par défaut)
    df['niveau_education'] = 3
    
    # 9. Sélectionner UNIQUEMENT les colonnes attendues par le modèle
    # En utilisant les features detectés du scaler si disponibles
    if noms_features is not None:
        # Utiliser les features du scaler
        colonnes_attendues = noms_features
    else:
        # Utiliser les colonnes par défaut
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
    
    # Créer DataFrame final avec les colonnes correctes
    df_final = pd.DataFrame()
    
    for col in colonnes_attendues:
        if col in df.columns:
            df_final[col] = df[col]
        else:
            # Remplir avec 0 si manquante
            df_final[col] = 0
    
    return df_final

# ========================================
# 🔮 FONCTION DE PRÉDICTION
# ========================================
def predire_churn(age, salaire, emplois_precedents, heures_semaine, experience_totale,
                 annees_entreprise, annees_poste, satisfaction_environnement, evaluation_precedente, 
                 niveau_hierarchique, satisfaction_travail, satisfaction_equipe, satisfaction_balance, 
                 evaluation_actuelle, heures_supplementaires, augmentation_salaire, participation_pee, 
                 formations_completees, employes_supervision, distance, annees_derniere_promotion, 
                 annees_responsable_actuel, genre, etat_civil, departement, domaine_etude, poste_freq_deplacement):
    
    if modele is None or scaler is None:
        return "❌ Erreur: Modèle ou Scaler non chargés correctement.\n\nVérifiez que les fichiers suivants existent dans le dossier 'models/':\n- lr_model_opt.pkl\n- scaler.pkl\n- seuil_opt.pkl"
    
    try:
        # Prétraiter les données
        donnees_pretraitees = pretraiter_donnees(
            age, salaire, emplois_precedents, heures_semaine, experience_totale,
            annees_entreprise, annees_poste, satisfaction_environnement, evaluation_precedente, 
            niveau_hierarchique, satisfaction_travail, satisfaction_equipe, satisfaction_balance, 
            evaluation_actuelle, heures_supplementaires, augmentation_salaire, participation_pee, 
            formations_completees, employes_supervision, distance, annees_derniere_promotion, 
            annees_responsable_actuel, genre, etat_civil, departement, domaine_etude, poste_freq_deplacement
        )
        
        print(f"✅ Données prétraitées: {len(donnees_pretraitees.columns)} colonnes")
        
        # Normaliser
        donnees_normalisees = scaler.transform(donnees_pretraitees)
        
        # Prédiction
        probabilites = modele.predict_proba(donnees_normalisees)[0]
        prob_abandon = probabilites[1]
        
        # Appliquer le seuil
        prediction = 1 if prob_abandon >= meilleur_seuil else 0
        
        # Résultats
        pourcentage_abandon = prob_abandon * 100
        pourcentage_seuil = meilleur_seuil * 100
        
        if prediction == 1:
            resultat = "🔴 **RISQUE ÉLEVÉ D'ABANDON**"
            recommandation = "⚠️ Une intervention immédiate est recommandée (augmentation, promotion, avantages, etc.)"
        else:
            resultat = "🟢 **RISQUE FAIBLE**"
            recommandation = "✅ Employé stable, maintenir la relation positive"
        
        sortie = f"""{resultat}

**Probabilité d'Abandon:** {pourcentage_abandon:.1f}%
**Seuil Appliqué:** {pourcentage_seuil:.2f}%

{recommandation}

---
**Confiance du Modèle:** {max(probabilites)*100:.1f}%
**Détail:** Score de risque de {pourcentage_abandon:.1f}%
        """
        return sortie
        
    except Exception as e:
        message_erreur = str(e)
        print(f"❌ Erreur en prédiction: {message_erreur}")
        return f"❌ Erreur: {message_erreur}\n\n💡 Vérifiez que tous les champs sont correctement remplis."

# ========================================
# 🎨 INTERFACE GRADIO
# ========================================
with gr.Blocks(title="Prédicteur de Churn", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
# 👥 Prédicteur de Churn des Employés
## Prédis si un employé risque d'abandonner l'entreprise

**Remplis les champs ci-dessous et clique sur "PRÉDIRE"**
    """)
    
    # Section 1: Information Personnelle
    with gr.Group():
        gr.Markdown("### 📝 Information Personnelle")
        with gr.Row():
            age = gr.Number(value=35, label="Âge", info="Âge de l'employé")
            genre = gr.Radio(["Masculin", "Féminin"], value="Masculin", label="Genre")
            etat_civil = gr.Radio(["Célibataire", "Marié(e)", "Divorcé(e)"], value="Marié(e)", label="État Civil")
        with gr.Row():
            salaire = gr.Number(value=5000, label="Salaire Mensuel (€)", info="Salaire brut mensuel")
            distance = gr.Number(value=5, label="Distance Domicile (km)", info="Km de trajet")
    
    # Section 2: Entreprise et Poste
    with gr.Group():
        gr.Markdown("### 🏢 Informations Entreprise et Poste")
        with gr.Row():
            departement = gr.Dropdown(
                ["Consulting", "Ressources Humaines", "IT", "Finance", "Marketing"],
                value="Consulting", label="Département"
            )
            domaine_etude = gr.Dropdown(
                ["Entrepreunariat", "Infra & Cloud", "Marketing", "Ressources Humaines", "Transformation Digitale", "Autres"],
                value="Transformation Digitale", label="Domaine d'Étude"
            )
        with gr.Row():
            poste_freq_deplacement = gr.Radio(["Rare", "Modéré", "Fréquent"], value="Modéré", label="Fréquence de Déplacement")
            niveau_hierarchique = gr.Slider(1, 5, 2, label="Niveau Hiérarchique (1-5)")
    
    # Section 3: Expérience
    with gr.Group():
        gr.Markdown("### 💼 Expérience Professionnelle")
        with gr.Row():
            emplois_precedents = gr.Number(value=3, label="Emplois Antérieurs")
            experience_totale = gr.Number(value=8, label="Années d'Expérience Totale")
        with gr.Row():
            annees_entreprise = gr.Number(value=5, label="Années dans l'Entreprise")
            annees_poste = gr.Number(value=2, label="Années au Poste Actuel")
        with gr.Row():
            annees_derniere_promotion = gr.Number(value=1, label="Années depuis Dernière Promotion")
            annees_responsable_actuel = gr.Number(value=3, label="Années sous Responsable Actuel")
    
    # Section 4: Travail
    with gr.Group():
        gr.Markdown("### 📊 Évaluation et Travail")
        with gr.Row():
            heures_semaine = gr.Number(value=40, label="Heures/Semaine")
            evaluation_precedente = gr.Slider(1, 4, 3, label="Évaluation Précédente (1-4)")
            evaluation_actuelle = gr.Slider(1, 4, 3, label="Évaluation Actuelle (1-4)")
        with gr.Row():
            employes_supervision = gr.Number(value=0, label="Employés Supervisés")
            heures_supplementaires = gr.Checkbox(value=False, label="Travaille Heures Supplémentaires?")
    
    # Section 5: Satisfaction
    with gr.Group():
        gr.Markdown("### 😊 Niveaux de Satisfaction (1-4)")
        with gr.Row():
            satisfaction_environnement = gr.Slider(1, 4, 3, label="Environnement")
            satisfaction_travail = gr.Slider(1, 4, 3, label="Type de Travail")
            satisfaction_equipe = gr.Slider(1, 4, 3, label="Équipe")
            satisfaction_balance = gr.Slider(1, 4, 3, label="Équilibre Vie-Travail")
    
    # Section 6: Compensation
    with gr.Group():
        gr.Markdown("### 💰 Compensation")
        with gr.Row():
            augmentation_salaire = gr.Number(value=15, label="Dernière Augmentation (%)")
            participation_pee = gr.Number(value=1, label="Participation Plan Actions")
            formations_completees = gr.Number(value=2, label="Formations Complétées")
    
    # Boutons
    gr.Markdown("---")
    with gr.Row():
        bouton_predire = gr.Button("🔮 PRÉDIRE", variant="primary", size="lg")
        bouton_reinitialiser = gr.Button("🔄 Réinitialiser", size="lg")
    
    # Sortie
    sortie = gr.Markdown("Le résultat apparaîtra ici...")
    
    # Actions des boutons
    bouton_predire.click(
        fn=predire_churn,
        inputs=[age, salaire, emplois_precedents, heures_semaine, experience_totale,
                annees_entreprise, annees_poste, satisfaction_environnement, evaluation_precedente, 
                niveau_hierarchique, satisfaction_travail, satisfaction_equipe, satisfaction_balance, 
                evaluation_actuelle, heures_supplementaires, augmentation_salaire, participation_pee, 
                formations_completees, employes_supervision, distance, annees_derniere_promotion, 
                annees_responsable_actuel, genre, etat_civil, departement, domaine_etude, poste_freq_deplacement],
        outputs=sortie
    )
    
    bouton_reinitialiser.click(
        fn=lambda: (35, 5000, 3, 40, 8, 5, 2, 3, 3, 2, 3, 3, 3, 3, False, 15, 1, 2, 0, 5, 1, 3, 
                    "Masculin", "Marié(e)", "Consulting", "Transformation Digitale", "Modéré", "Le résultat apparaîtra ici..."),
        outputs=[age, salaire, emplois_precedents, heures_semaine, experience_totale,
                annees_entreprise, annees_poste, satisfaction_environnement, evaluation_precedente, 
                niveau_hierarchique, satisfaction_travail, satisfaction_equipe, satisfaction_balance, 
                evaluation_actuelle, heures_supplementaires, augmentation_salaire, participation_pee, 
                formations_completees, employes_supervision, distance, annees_derniere_promotion, 
                annees_responsable_actuel, genre, etat_civil, departement, domaine_etude, poste_freq_deplacement, sortie]
    )
    
    gr.Markdown("---\n**Modèle ML optimisé pour prédiction de churn** 🚀")

if __name__ == "__main__":
    demo.launch(share=False)