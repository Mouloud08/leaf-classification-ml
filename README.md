# Projet IFT 712 - Leaf Classification

Projet de session pour le cours **IFT 712 - Techniques d'apprentissage**.

## Auteurs

TODO: ajouter les noms des membres de l'equipe.

## Description

Ce depot contient l'initialisation du projet de classification multiclasse des feuilles
du concours Kaggle *Leaf Classification*. La structure a ete preparee pour accueillir :

- le chargement et la preparation des donnees;
- les pipelines de pretraitement;
- les six modeles demandes;
- l'evaluation et l'analyse d'erreurs;
- les notebooks d'exploration, d'entrainement et d'analyse finale.

## Structure du projet

```text
projet_leaf_ift712/
├── data/
│   └── raw/
├── models/
├── notebooks/
├── src/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Execution suggeree

1. Telecharger `train.csv` depuis Kaggle et le placer dans `data/raw/`.
2. Completer les `TODO` dans les modules Python.
3. Executer les notebooks dans l'ordre :
   1. `notebooks/01_Exploratory_Data_Analysis.ipynb`
   2. `notebooks/02_Model_Training_and_Tuning.ipynb`
   3. `notebooks/03_Evaluation_and_Error_Analysis.ipynb`

## Dataset

Lien Kaggle :
TODO: ajouter l'URL officielle du challenge Leaf Classification.

## Etat actuel

Le depot est **initialise** avec des squelettes de code, des signatures de classes et
des commentaires `TODO` pour guider l'implementation complete.
