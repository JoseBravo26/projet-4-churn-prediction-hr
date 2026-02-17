import gradio as gr
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Chargement modèles (HF Spaces les trouve auto)
model = joblib.load("models/lr_model_opt.pkl")
scaler = joblib.load("models/scaler.pkl")
seuil_opt = joblib.load("models/seuil_opt.pkl")["meilleur_seuil_lr"]

# Features dans l'ordre EXACT (copié de ton notebook)
FEATURES = [
    'satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',
    'time_spend_company', 'Work_accident', 'left', 'promotion_last_5years',
    'sales_sales', 'sales_accounting', 'sales_hr', 'sales_technical',
    'salary_low', 'salary_medium', 'salary_high'
]

print("✅ Modèles chargés ! Seuil optimal:", seuil_opt)

def predict_churn(file, satisfaction_level, last_evaluation, number_project, 
                  average_montly_hours, time_spend_company, Work_accident,
                  promotion_last_5years, sales, salary):
    """
    Prédiction churn individuel OU batch CSV
    """
    
    # 1. Prédiction INDIVIDUELLE (formulaire)
    if file is None:
        # Créer DataFrame avec features dans ORDRE EXACT
        data = {
            'satisfaction_level': [satisfaction_level],
            'last_evaluation': [last_evaluation],
            'number_project': [number_project],
            'average_montly_hours': [average_montly_hours],
            'time_spend_company': [time_spend_company],
            'Work_accident': [Work_accident],
            'left': [0],  # dummy
            'promotion_last_5years': [promotion_last_5years],
            'sales_sales': [1 if sales == "sales" else 0],
            'sales_accounting': [1 if sales == "accounting" else 0],
            'sales_hr': [1 if sales == "hr" else 0],
            'sales_technical': [1 if sales == "technical" else 0],
            'salary_low': [1 if salary == "low" else 0],
            'salary_medium': [1 if salary == "medium" else 0],
            'salary_high': [1 if salary == "high" else 0]
        }
        
        df = pd.DataFrame(data)[FEATURES]
        
        # Scale + prédiction
        X_scaled = scaler.transform(df)
        proba = model.predict_proba(X_scaled)[0, 1]
        pred = 1 if proba >= seuil_opt else 0
        
        risque = "🔴 **RISQUE ÉLEVÉ** (>50%)" if pred == 1 else "🟢 **FAIBLE RISQUE**"
        
        return f"{risque}\n\n📊 **Probabilité churn**: {proba:.1%}"
    
    # 2. Prédiction BATCH (CSV upload)
    else:
        try:
            df = pd.read_csv(file)
            print(f"📁 CSV chargé: {len(df)} lignes")
            
            # Vérifier features
            missing = [col for col in FEATURES if col not in df.columns]
            if missing:
                return f"❌ Features manquantes: {missing}"
            
            # Prédictions batch
            X = df[FEATURES]
            X_scaled = scaler.transform(X)
            probas = model.predict_proba(X_scaled)[:, 1]
            preds = (probas >= seuil_opt).astype(int)
            
            # Ajouter colonnes résultats
            df["proba_churn"] = probas
            df["pred_churn"] = preds
            df["risque"] = df["pred_churn"].map({0: "🟢 Faible", 1: "🔴 Élevé"})
            
            # Stats
            churn_rate = (preds == 1).mean()
            n_churn = (preds == 1).sum()
            
            return (
                df[["proba_churn", "pred_churn", "risque"]].round(3).to_csv(index=False),
                f"📈 **{n_churn}/{len(df)} employés** en risque churn ({churn_rate:.1%})\n"
                f"**Moyenne proba**: {probas.mean():.1%}"
            )
        except Exception as e:
            return f"❌ Erreur CSV: {str(e)}"

# Interface Gradio
with gr.Blocks(title="🚀 Prédiction Churn RH", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔮 **Prédicteur Churn RH** (LogisticRegression Optimisée)
    
    **Meilleur modèle** : F1-test optimisé + faible overfitting
    Upload CSV **OU** teste un employé individuellement 👇
    """)
    
    with gr.Tab("🎯 Test Individuel"):
        with gr.Row():
            with gr.Column(scale=1):
                satisfaction = gr.Slider(0.0, 1.0, value=0.7, label="Satisfaction (0-1)")
                evaluation = gr.Slider(0.0, 1.0, value=0.8, label="Last Evaluation (0-1)")
                projects = gr.Slider(2, 10, value=4, step=1, label="Nb Projets")
                hours = gr.Slider(100, 300, value=200, step=10, label="Heures/mois")
                seniority = gr.Slider(1, 10, value=3, step=1, label="Ancienneté (ans)")
                accident = gr.Checkbox(label="Accident travail")
                promotion = gr.Checkbox(label="Promotion 5 ans")
            
            with gr.Column(scale=1):
                sales = gr.Dropdown(["sales", "accounting", "hr", "technical", "management"], 
                                  value="sales", label="Département")
                salary = gr.Dropdown(["low", "medium", "high"], value="medium", label="Salaire")
        
        predict_btn = gr.Button("🔮 Prédire Churn", variant="primary")
        result_single = gr.Markdown()
    
    with gr.Tab("📊 Batch CSV"):
        csv_input = gr.File(label="Upload CSV (colonnes FEATURES requises)")
        predict_csv_btn = gr.Button("🚀 Analyser Dataset", variant="primary")
        result_csv = gr.Textbox(label="Résultats CSV (téléchargeable)")
        stats_csv = gr.Markdown()
    
    # Événements
    predict_btn.click(predict_churn, 
                     inputs=[satisfaction, evaluation, projects, hours, seniority, 
                            accident, promotion, sales, salary], 
                     outputs=result_single)
    
    predict_csv_btn.click(predict_churn, inputs=csv_input, 
                         outputs=[result_csv, stats_csv])

if __name__ == "__main__":
    demo.launch()
