# Projet IFT712 - Leaf Classification

Projet de session pour **IFT712 - Techniques d'apprentissage** autour du
dataset **Kaggle Leaf Classification**.

Cette version reconstruit la base du projet avec une architecture modulaire :

- le notebook EDA existant est conserve ;
- le code reutilisable vit dans `src/` ;
- les notebooks modeles suivent tous la meme progression ;
- les artefacts de la nouvelle version sont ecrits dans `results/`.

## Structure

```text
Projet_Session_IFT712/
├── data/
│   └── raw/
│       └── train.csv
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Modele_Perceptron.ipynb
│   ├── 03_Modele_Regression_Logistique.ipynb
│   ├── 04_Modele_SVM.ipynb
│   ├── 05_Modele_Foret_Aleatoire.ipynb
│   ├── 06_Modele_AdaBoost.ipynb
│   ├── 07_Modele_MLP.ipynb
│   ├── 08_Comparaison_Globale.ipynb
│   └── 09_Bonus_Ensemble.ipynb
├── results/
│   ├── metrics/
│   ├── grid_search/
│   ├── predictions/
│   ├── figures/
│   └── models/
├── src/
│   ├── config.py
│   ├── donnees.py
│   ├── data_manager.py
│   ├── pipelines.py
│   ├── modeles.py
│   ├── models.py
│   ├── evaluation.py
│   ├── evaluator.py
│   ├── visualisation.py
│   ├── plotting.py
│   ├── resultats.py
│   └── __init__.py
├── models/                  # artefacts legacy de l'ancienne version
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Ordre d'execution

1. Placer `train.csv` dans `data/raw/`.
2. Executer `01_Exploratory_Data_Analysis.ipynb`.
3. Executer les notebooks `02` a `07`.
4. Executer `08_Comparaison_Globale.ipynb`.
5. Completer ensuite `09_Bonus_Ensemble.ipynb`.

## Regle commune des notebooks modeles

Chaque notebook `02` a `07` suit la meme structure :

1. baseline simple sans preprocessing ;
2. variante amelioree avec preprocessing adapte ;
3. tuning en boilerplate seulement pour la v1.

## Notes

- L'EDA actuel est garde tel quel.
- Les wrappers `src/data_manager.py`, `src/models.py` et `src/evaluator.py`
  sont presents pour ne pas casser l'ancien notebook.
- Le dossier `models/` existant reste en place comme legacy, mais la nouvelle
  source de verite pour les artefacts est `results/`.
