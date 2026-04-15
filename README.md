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
│   ├── 06_Modele_Arbre_Decision.ipynb
│   ├── 07_Modele_MLP.ipynb
│   └── 08_Comparaison_Globale.ipynb
├── results/
│   ├── metrics/
│   ├── grid_search/
│   ├── predictions/
│   ├── figures/
│   └── models/
├── src/
│   ├── config.py
│   ├── data/
│   ├── evaluation/
│   ├── experiments/
│   ├── models/
│   ├── plots/
│   ├── artifacts/
│   ├── notebook_support.py
│   ├── cli/
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

## Regle commune des notebooks modeles

Chaque notebook `02` a `07` suit la meme structure :

1. baseline simple sans preprocessing ;
2. variante amelioree avec preprocessing adapte ;
3. tuning et comparaison finale avec validation croisee imbriquee.

## Notes

- L'EDA actuel est garde tel quel.
- Les facades `src/modeles.py`, `src/pipelines.py` et `src/visualisation.py`
  preservent les imports utilises par les notebooks.
- Le dossier `models/` existant reste en place comme legacy, mais la nouvelle
  source de verite pour les artefacts est `results/`.
- Les scores `tuned` sauvegardes dans `results/metrics/` proviennent d'une
  validation croisee imbriquee, afin d'eviter le biais de selection.
- Les figures prefixees `exploratory__` sont des diagnostics descriptifs
  refittes sur le sous-ensemble de developpement (`80 %`); elles ne doivent
  pas etre interpretees comme preuve confirmatoire au meme titre que les
  metriques CV/nested CV.
