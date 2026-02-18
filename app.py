import gradio as gr
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 📦 CHARGER MODÈLE, SCALER ET SEUIL
# ========================================
model = joblib.load('models/lr_model_opt.pkl')
scaler = joblib.load('models/scaler.pkl')
seuil_dict = joblib.load('models/seuil_opt.pkl')
meilleur_seuil = seuil_dict['meilleur_seuil_lr']

print(f"✅ Modèle chargé")
print(f"✅ Scaler chargé")
print(f"✅ Seuil optimal : {meilleur_seuil:.4f}")

# ========================================
# 📋 NOMS DES FEATURES DANS LE BON ORDRE
# ========================================
# IMPORTANT : Cet ordre DOIT correspondre à l'ordre d'entraînement
feature_names = [
    'age', 'revenu_mensuel', 'nombre_experiences_precedentes',
    'nombre_heures_travailless', 'annee_experience_totale',
    'annees_dans_l_entreprise', 'annees_dans_le_poste_actuel',
    'satisfaction_employee_environnement', 'note_evaluation_precedente',
    'niveau_hierarchique_poste', 'satisfaction_employee_nature_travail',
    'satisfaction_employee_equipe', 'satisfaction_employee_equilibre_pro_perso',
    'note_evaluation_actuelle', 'heure_supplementaires',
    'augementation_salaire_precedente', 'nombre_participation_pee',
    'nb_formations_suivies', 'nombre_employee_sous_responsabilite',
    'distance_domicile_travail', 'niveau_education',
    'annees_depuis_la_derniere_promotion', 'annes_sous_responsable_actuel'
]

# ========================================
# 🔮 FONCTION DE PRÉDICTION
# ========================================
def predict_churn(age, revenu, exp_prev, horas_trabajo, exp_total,
                  años_empresa, años_puesto, sat_env, eval_prev, nivel_jer,
                  sat_trabajo, sat_equipo, sat_balance, eval_actual, 
                  horas_extra, aumento_sal, part_pee, formaciones, 
                  empleados_bajo, distancia, nivel_edu, años_promocion, 
                  años_responsable):
    
    try:
        # Créer DataFrame avec les valeurs d'entrée
        input_data = pd.DataFrame([[
            age, revenu, exp_prev, horas_trabajo, exp_total,
            años_empresa, años_puesto, sat_env, eval_prev, nivel_jer,
            sat_trabajo, sat_equipo, sat_balance, eval_actual, 
            horas_extra, aumento_sal, part_pee, formaciones, 
            empleados_bajo, distancia, nivel_edu, años_promocion, 
            años_responsable
        ]], columns=feature_names)
        
        # Normaliser les caractéristiques
        input_scaled = scaler.transform(input_data)
        
        # Prédiction avec probabilité
        proba = model.predict_proba(input_scaled)
        prob_churn = proba  # Probabilité d'abandon (classe 1)
        
        # Appliquer le seuil optimal
        prediction = 1 if prob_churn >= meilleur_seuil else 0
        
        # Générer le résultat détaillé
        if prediction == 1:
            resultat = "⚠️ **RISQUE ÉLEVÉ D'ABANDON**"
            couleur = "🔴"
            recommandation = "Intervention immédiate recommandée (rétention, avantages, etc.)"
        else:
            resultat = "✅ **FAIBLE RISQUE**"
            couleur = "🟢"
            recommandation = "Employé avec probabilité faible d'abandon."
        
        # Créer le message de sortie
        output_text = f"""
{couleur} {resultat}

**Probabilité de Churn :** {prob_churn*100:.1f}%
**Seuil Appliqué :** {meilleur_seuil*100:.2f}%
**Prédiction :** {'Quittera l\'entreprise' if prediction == 1 else 'Restera dans l\'entreprise'}

**Recommandation :** {recommandation}

---
**Confiance du Modèle :** {max(proba)*100:.1f}%
        """
        
        return output_text
        
    except Exception as e:
        return f"❌ Erreur dans la prédiction : {str(e)}"

# ========================================
# 🎨 INTERFACE GRADIO
# ========================================
def creer_interface():
    with gr.Blocks(title="Prédicteur de Churn - RH", theme=gr.themes.Soft()) as demo:
        
        # En-tête
        gr.Markdown("""
# 👥 Prédicteur de Churn des Employés
## Prédis si un employé risque de quitter l'entreprise
        
---
**Remplis les champs de l'employé et clique sur "Prédire" pour obtenir l'analyse de risque.**
        """)
        
        # SECTION 1 : INFORMATIONS PERSONNELLES ET PROFESSIONNELLES
        with gr.Group():
            gr.Markdown("### 📝 Informations Personnelles et Professionnelles")
            with gr.Row():
                with gr.Column():
                    age = gr.Slider(
                        label="Âge",
                        minimum=18, maximum=65, value=35, step=1,
                        info="Âge de l'employé"
                    )
                    revenu = gr.Number(
                        label="Revenu Mensuel (€)",
                        value=5000,
                        info="Salaire mensuel brut"
                    )
                    niveau_edu = gr.Slider(
                        label="Niveau d'Éducation",
                        minimum=1, maximum=5, value=3, step=1,
                        info="1=Maximum, 5=Minimum"
                    )
                
                with gr.Column():
                    distancia = gr.Slider(
                        label="Distance Domicile-Travail (km)",
                        minimum=0, maximum=50, value=5, step=1,
                        info="Distance de trajet"
                    )
                    horas_trabajo = gr.Number(
                        label="Heures de Travail/Semaine",
                        value=80,
                        info="Heures travaillées par semaine"
                    )
        
        # SECTION 2 : EXPÉRIENCE
        with gr.Group():
            gr.Markdown("### 💼 Expérience et Trajectoire")
            with gr.Row():
                with gr.Column():
                    exp_prev = gr.Slider(
                        label="Expériences Précédentes",
                        minimum=0, maximum=20, value=3, step=1,
                        info="Nombre d'emplois antérieurs"
                    )
                    exp_total = gr.Slider(
                        label="Années d'Expérience Totale",
                        minimum=0, maximum=50, value=8, step=1,
                        info="Expérience professionnelle accumulée"
                    )
                
                with gr.Column():
                    años_empresa = gr.Slider(
                        label="Années dans l'Entreprise",
                        minimum=0, maximum=40, value=5, step=1,
                        info="Ancienneté dans l'entreprise"
                    )
                    años_puesto = gr.Slider(
                        label="Années au Poste Actuel",
                        minimum=0, maximum=30, value=3, step=1,
                        info="Temps au poste actuel"
                    )
        
        # SECTION 3 : ÉVALUATION ET PERFORMANCE
        with gr.Group():
            gr.Markdown("### 📊 Évaluation et Performance")
            with gr.Row():
                with gr.Column():
                    eval_prev = gr.Slider(
                        label="Évaluation Précédente",
                        minimum=1, maximum=4, value=3, step=1,
                        info="Note de l'évaluation précédente"
                    )
                    eval_actual = gr.Slider(
                        label="Évaluation Actuelle",
                        minimum=1, maximum=4, value=3, step=1,
                        info="Note de l'évaluation actuelle"
                    )
                
                with gr.Column():
                    nivel_jer = gr.Slider(
                        label="Niveau Hiérarchique",
                        minimum=1, maximum=5, value=2, step=1,
                        info="1=Bas, 5=Haut"
                    )
                    empleados_bajo = gr.Slider(
                        label="Employés sous Responsabilité",
                        minimum=0, maximum=50, value=0, step=1,
                        info="Nombre de personnes supervisées"
                    )
        
        # SECTION 4 : SATISFACTION
        with gr.Group():
            gr.Markdown("### 😊 Niveaux de Satisfaction (1-4)")
            with gr.Row():
                with gr.Column():
                    sat_env = gr.Slider(
                        label="Satisfaction Environnement",
                        minimum=1, maximum=4, value=3, step=1,
                        info="Satisfaction avec l'environnement de travail"
                    )
                    sat_trabajo = gr.Slider(
                        label="Satisfaction Nature du Travail",
                        minimum=1, maximum=4, value=3, step=1,
                        info="Aime-t-il ce qu'il fait ?"
                    )
                
                with gr.Column():
                    sat_equipo = gr.Slider(
                        label="Satisfaction Équipe",
                        minimum=1, maximum=4, value=3, step=1,
                        info="Satisfaction avec les collègues"
                    )
                    sat_balance = gr.Slider(
                        label="Satisfaction Équilibre Vie-Travail",
                        minimum=1, maximum=4, value=3, step=1,
                        info="Équilibre vie personnelle-professionnelle ?"
                    )
        
        # SECTION 5 : COMPENSATION ET AVANTAGES
        with gr.Group():
            gr.Markdown("### 💰 Compensation et Avantages")
            with gr.Row():
                with gr.Column():
                    aumento_sal = gr.Number(
                        label="Dernier Augmentation Salaire (%)",
                        value=15,
                        info="Pourcentage de la dernière augmentation"
                    )
                    horas_extra = gr.Checkbox(
                        label="Travaille Heures Supplémentaires ?",
                        value=False,
                        info="Réalise-t-il des heures extraordinaires ?"
                    )
                
                with gr.Column():
                    part_pee = gr.Slider(
                        label="Participation Plan Actions",
                        minimum=0, maximum=5, value=1, step=1,
                        info="Participation en PEE/plans"
                    )
                    formaciones = gr.Slider(
                        label="Formations Complétées",
                        minimum=0, maximum=10, value=2, step=1,
                        info="Nombre de cours réalisés"
                    )
        
        # SECTION 6 : PROGRESSION
        with gr.Group():
            gr.Markdown("### 🚀 Progression et Carrière")
            with gr.Row():
                with gr.Column():
                    años_promocion = gr.Slider(
                        label="Années depuis Dernière Promotion",
                        minimum=0, maximum=20, value=1, step=1,
                        info="Quand a eu lieu la dernière promotion ?"
                    )
                    años_responsable = gr.Slider(
                        label="Années sous Responsable Actuel",
                        minimum=0, maximum=20, value=3, step=1,
                        info="Temps avec manager/responsable actuel"
                    )
        
        # BOUTONS D'ACTION
        gr.Markdown("---")
        with gr.Row():
            predict_btn = gr.Button("🔮 Prédire le Risque de Churn", variant="primary", size="lg")
            reset_btn = gr.Button("🔄 Réinitialiser", size="lg")
        
        # OUTPUT
        output = gr.Markdown(label="Résultat")
        
        # FONCTIONS DES BOUTONS
        predict_btn.click(
            predict_churn,
            inputs=[age, revenu, exp_prev, horas_trabajo, exp_total,
                    años_empresa, años_puesto, sat_env, eval_prev, nivel_jer,
                    sat_trabajo, sat_equipo, sat_balance, eval_actual, 
                    horas_extra, aumento_sal, part_pee, formaciones, 
                    empleados_bajo, distancia, nivel_edu, años_promocion, 
                    años_responsable],
            outputs=output
        )
        
        reset_btn.click(
            lambda: (35, 5000, 3, 80, 8, 5, 3, 3, 3, 2, 3, 3, 3, 3, False, 15, 1, 2, 0, 5, 3, 1, 3, ""),
            outputs=[age, revenu, exp_prev, horas_trabajo, exp_total,
                    años_empresa, años_puesto, sat_env, eval_prev, nivel_jer,
                    sat_trabajo, sat_equipo, sat_balance, eval_actual, 
                    horas_extra, aumento_sal, part_pee, formaciones, 
                    empleados_bajo, distancia, nivel_edu, años_promocion, 
                    años_responsable, output]
        )
        
        # Pied de page
        gr.Markdown(f"""
---
**ℹ️ Informations :**
- Modèle : Logistic Regression Optimisé
- Données d'entraînement : 1 470 employés
- Seuil optimal : {meilleur_seuil*100:.2f}%
- Précision du modèle : ~95%

**Développé avec Scikit-learn, Gradio et Hugging Face Spaces**
        """)
        
    return demo

# ========================================
# 🚀 EXÉCUTER L'APPLICATION
# ========================================
if __name__ == "__main__":
    demo = creer_interface()
    demo.launch(share=False)
