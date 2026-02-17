import gradio as gr
import joblib
import pandas as pd
from catboost import CatBoostClassifier

# Charge mod√®le
model = CatBoostClassifier()
model.load_model("models/catboost_final.pkl")

def predict_churn(satisfaction, tenure, salaire, nb_projets):
    data = pd.DataFrame({
        'satisfaction': [satisfaction],
        'tenure': [tenure],
        'revenumensuel': [salaire],
        'nombre_projets': [nb_projets]
    })
    proba = model.predict_proba(data)[0][1] * 100
    return f"{'Ì¥¥ √âLEV√â' if proba > 50 else 'Ìø¢ FAIBLE'} : {proba:.1f}% churn"

demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Slider(0, 1, label="Satisfaction", step=0.1),
        gr.Slider(0, 10, label="Anciennet√©", step=1),
        gr.Slider(20000, 150000, label="Salaire", step=1000),
        gr.Slider(0, 10, label="Nb projets", step=1)
    ],
    outputs=gr.Markdown(),
    title="Ì∫Ä Pr√©diction Churn HR CatBoost"
)

if __name__ == "__main__":
    demo.launch()
