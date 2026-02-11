# Projet 4 : Prédiction du Churn des Employés (HR Analytics)

Modèle ML CatBoost pour prédire les départs (recall priorisé : 0.57 sur test, ROC-AUC 0.81).

## Installation
1. Clonez : `git clone https://github.com/JoseBravo26/projet-4-churn-prediction-hr.git`
2. `cd projet-4-churn-prediction-hr`
3. `pip install -r requirements.txt`
4. Lancez notebook : `jupyter notebook notebooks/Projet-4-Churn.ipynb`

## Usage
- Entraînez : `python src/model_training.py`
- Prédisez : `python src/predict.py --data data/test.csv`

## Résultats
| Métrique | Test |
|----------|------|
| Accuracy | 0.85 |
| Recall (départs) | 0.57 |
| F1 | 0.55 |
| ROC-AUC | 0.81 |

Features clés : satisfaction, tenure, salaire. [file:31]

## Branches & Tags
- `main` : Production
- `develop` : Dev
- Tags : `v1.0.0`

Licence : MIT
