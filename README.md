# Projet IFT712 - Leaf Classification

Projet de session pour **IFT712 - Techniques d'apprentissage** autour du
dataset **Kaggle Leaf Classification**.

Cette version reconstruit la base du projet avec une architecture modulaire :

- le notebook EDA existant est conservé ;
- le code réutilisable vit dans `src/` ;
- les notebooks modèles suivent tous la même progression ;
- les artefacts de la nouvelle version sont écrits dans `results/`.

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

## Utilisation rapide

1. Installer l'environnement :

```bash
pip install -r requirements.txt
```

2. Lancer une étude complète :

```bash
python -m src.cli.main --all
```

3. Ouvrir et exécuter les notebooks :

```bash
jupyter notebook
```

## Ordre d'exécution

1. Placer `train.csv` dans `data/raw/`.
2. Exécuter `01_Exploratory_Data_Analysis.ipynb`.
3. Exécuter les notebooks `02` à `07`.
4. Exécuter `08_Comparaison_Globale.ipynb`.

## Règle commune des notebooks modèles

Chaque notebook `02` à `07` suit la même structure :

1. référence simple sans prétraitement ;
2. variante améliorée avec prétraitement adapté ;
3. ajustement d'hyperparamètres et comparaison finale avec validation croisée imbriquée.

## Notes

- L'EDA actuel est gardé tel quel.
- Les facades `src/modeles.py`, `src/pipelines.py` et `src/visualisation.py`
  préservent les imports utilisés par les notebooks.
- Le dossier `models/` existant reste en place comme legacy, mais la nouvelle
  source de vérité pour les artefacts est `results/`.
- Les scores `tuned` sauvegardés dans `results/metrics/` proviennent d'une
  validation croisée imbriquée, afin d'éviter le biais de sélection.
- Les figures préfixées `exploratory__` sont des diagnostics descriptifs
  réappris sur le sous-ensemble de développement (`80 %`) ; elles ne doivent
  pas être interprétées comme preuve confirmatoire au même titre que les
  métriques de CV et de validation croisée imbriquée.
