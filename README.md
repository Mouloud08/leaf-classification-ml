# Projet IFT712 - Leaf Classification

Projet de classification multiclasse sur le dataset **Kaggle Leaf Classification** dans le cadre du cours **IFT712 - Techniques d'apprentissage**.

L'objectif du dépôt est de comparer plusieurs familles de modèles supervisés sur un même protocole expérimental, puis de documenter leurs performances, leurs erreurs et leurs compromis dans des notebooks d'analyse.

Le rapport final remis pour le projet est disponible dans [`rapport/rapport_leaf_classification_final.pdf`](rapport/rapport_leaf_classification_final.pdf).

## Ce que fait le projet

Le projet étudie six modèles :

- `perceptron`
- `regression_logistique`
- `svm`
- `foret_aleatoire`
- `arbre_decision`
- `mlp`

Pour chacun, on retrouve généralement :

- une ou plusieurs variantes exploratoires ;
- une version ajustée (`tuned`) évaluée plus rigoureusement ;
- des métriques, prédictions, figures et tableaux d'analyse ;
- une comparaison finale avec les autres modèles.

Le dépôt est organisé pour séparer :

- le **code réutilisable** dans `src/` ;
- les **artefacts produits** dans `results/` ;
- les **notebooks d'analyse** dans `notebooks/`.

## Point d'entrée principal : les notebooks

Les notebooks sont le moyen principal pour comprendre, relancer et présenter le projet.  
Ce sont eux qu'une personne découvrant le dépôt devrait ouvrir en priorité.

Ordre recommandé :

1. `01_Exploratory_Data_Analysis.ipynb`
2. `02_Modele_Perceptron.ipynb`
3. `03_Modele_Regression_Logistique.ipynb`
4. `04_Modele_SVM.ipynb`
5. `05_Modele_Foret_Aleatoire.ipynb`
6. `06_Modele_Arbre_Decision.ipynb`
7. `07_Modele_MLP.ipynb`
8. `08_Comparaison_Globale.ipynb`

En pratique :

- `01` sert à comprendre le dataset ;
- `02` à `07` détaillent l'étude de chaque modèle ;
- `08` rassemble la comparaison globale et les conclusions quantitatives.

## Architecture du dépôt

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
│   ├── global/
│   └── <modele>/
│       ├── metrics/
│       ├── predictions/
│       ├── tuning/
│       └── figures/
├── rapport/
│   └── rapport_leaf_classification_final.pdf
├── src/
│   ├── artifacts/
│   ├── cli/
│   ├── data/
│   ├── evaluation/
│   ├── experiments/
│   ├── models/
│   ├── plots/
│   ├── config.py
│   ├── notebook_support.py
│   └── __init__.py
├── tests/
├── requirements.txt
└── README.md
```

### Rôle des dossiers principaux

- `data/raw/` : données d'entrée, en particulier `train.csv`.
- `notebooks/` : analyses principales du projet.
- `results/` : sorties générées par le pipeline expérimental.
- `rapport/` : version finale du rapport de projet.
- `src/data/` : chargement et préparation des données.
- `src/models/` : définitions des modèles.
- `src/evaluation/` : métriques, validation croisée, évaluation.
- `src/experiments/` : orchestration des variantes, tuning et exécutions complètes.
- `src/artifacts/` : lecture, écriture et validation des artefacts dans `results/`.
- `src/notebook_support.py` : helpers utilisés par les notebooks.
- `tests/` : tests automatisés du code, des modèles et des artefacts.

## Installation

PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Le jeu d'entraînement utilisé par le projet est inclus dans :

```text
data/raw/train.csv
```

Pour vérifier l'installation :

```powershell
python -m pytest -q
```

## Utilisation des notebooks

Une fois l'environnement prêt :

```powershell
jupyter notebook
```

Les notebooks constituent le workflow principal pour explorer et présenter le projet.

Ils ne sont pas indépendants du reste du code :

- les notebooks modèles `02` à `07` s'appuient sur les mêmes briques expérimentales que la CLI ;
- le notebook `08_Comparaison_Globale.ipynb` relit principalement les artefacts déjà produits dans `results/` ;
- les figures, tableaux et métriques importants doivent donc rester cohérents avec ce qui est généré par le pipeline canonique.

Autrement dit :

- la **CLI** sert surtout à produire ou régénérer proprement les artefacts ;
- les **notebooks** servent surtout à analyser, interpréter et présenter ces résultats ;
- `results/` joue le rôle de base commune entre les deux.

## CLI

La CLI sert à exécuter le pipeline expérimental sans passer notebook par notebook.  
Elle est utile pour régénérer proprement les résultats dans `results/`.

Point d'entrée :

```powershell
python -m src.cli.main --help
```

### Commande la plus importante

```powershell
python -m src.cli.main --all
```

Cette commande lance l'étude sur tous les modèles définis par le projet.  
Elle exécute les variantes, le tuning lorsque prévu, puis régénère les résultats dans `results/`.

### Exemples utiles

Exécuter tous les modèles :

```powershell
python -m src.cli.main --all
```

Exécuter seulement certains modèles :

```powershell
python -m src.cli.main --models regression_logistique svm mlp
```

Réutiliser les résultats déjà présents :

```powershell
python -m src.cli.main --all --skip-existing
```

Forcer une réexécution complète :

```powershell
python -m src.cli.main --all --force
```

Désactiver le tuning ou les figures :

```powershell
python -m src.cli.main --models perceptron svm --no-tuning --no-figures
```

## Relation entre CLI et notebooks

La CLI et les notebooks sont conçus pour fonctionner ensemble.

- la CLI exécute l'étude de manière batch et écrit les sorties dans `results/` ;
- les notebooks utilisent ce même socle de code et ces mêmes artefacts pour construire l'analyse ;
- certains notebooks peuvent rejouer une partie du workflow modèle par modèle, mais ils restent alignés sur la même architecture expérimentale ;
- la comparaison globale doit être lue à partir des artefacts canoniques de `results/`.

Le chemin recommandé est donc :

1. régénérer `results/` avec la CLI si nécessaire ;
2. ouvrir les notebooks dans l'ordre `01` à `08` ;
3. utiliser les notebooks pour l'analyse détaillée et la présentation finale.

## Résumé rapide

Si quelqu'un reprend le projet, le chemin le plus naturel est :

1. installer l'environnement ;
2. placer `train.csv` dans `data/raw/` ;
3. lancer la CLI si besoin pour régénérer `results/` ;
4. ouvrir les notebooks et suivre l'analyse dans l'ordre `01` à `08`.
