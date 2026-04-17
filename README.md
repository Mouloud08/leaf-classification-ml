# Projet IFT712 - Leaf Classification

Étude de classification multiclasse sur le dataset **Kaggle Leaf Classification** dans le cadre du cours **IFT712 - Techniques d'apprentissage**.

Le dépôt est organisé comme un projet reproductible :
- le code exécutable et réutilisable vit dans `src/` ;
- les artefacts canoniques sont écrits dans `results/` ;
- les notebooks servent à l'analyse, à l'interprétation et à la rédaction ;
- le rapport final est produit à partir des artefacts canoniques, puis publié en Markdown/HTML/PDF et en LaTeX.

## Objectif

Comparer six familles de modèles dans un cadre méthodologique strict :
- `perceptron`
- `regression_logistique`
- `svm`
- `foret_aleatoire`
- `arbre_decision`
- `mlp`

Le protocole distingue explicitement :
- l'exploration des variantes non tunées ;
- l'estimation confirmatoire des variantes `tuned` en validation croisée imbriquée ;
- la corroboration secondaire sur un split gelé, qui ne redéfinit jamais le classement principal.

## Fonctionnalités

- CLI unifiée pour exécuter un ou plusieurs modèles.
- Layout canonique des résultats dans `results/<modele>/...`.
- Notebooks dédiés par modèle et notebook global de synthèse.
- Génération des figures et des sorties de comparaison globale.
- Vérification non destructive de la cohérence des artefacts et des notebooks.
- Pipeline de rapport en Markdown/HTML/PDF et publication LaTeX.

## Structure du dépôt

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
│   │   └── comparison/
│   └── <modele>/
│       ├── figures/
│       ├── metrics/
│       ├── predictions/
│       └── tuning/
├── scripts/
│   ├── build_report_pdf.py
│   ├── normalize_report_artifacts.py
│   ├── recompute_report_artifacts.py
│   └── report_readiness_check.py
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
├── docs/
│   ├── figures/
│   ├── rapport_parts/
│   └── IFT712_Rapport/
├── models/                  # artefacts legacy
├── requirements.txt
└── README.md
```

## Prérequis

- Python 3.x
- `pip`
- le fichier `train.csv` placé dans `data/raw/`

Pour la génération du PDF LaTeX final, un moteur LaTeX local est nécessaire, par exemple **MiKTeX** sur Windows.

## Installation

### Environnement virtuel

PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si tu n'utilises pas PowerShell, adapte simplement la commande d'activation de l'environnement.

## Démarrage rapide

1. Placer `train.csv` dans `data/raw/`.
2. Installer les dépendances.
3. Exécuter l'étude complète.
4. Ouvrir les notebooks si tu veux régénérer l'analyse narrative.
5. Vérifier la cohérence du rapport.

Commande minimale :

```powershell
python -m src.cli.main --all
```

Puis :

```powershell
jupyter notebook
python scripts\report_readiness_check.py
```

## Utilisation de la CLI

Point d'entrée principal :

```powershell
python -m src.cli.main --help
```

Exemples utiles :

### Exécuter toute l'étude

```powershell
python -m src.cli.main --all --figures core
```

### Exécuter seulement certains modèles

```powershell
python -m src.cli.main --models regression_logistique svm mlp --figures core
```

### Régénérer en forçant la réexécution

```powershell
python -m src.cli.main --all --force --figures all --verbose
```

### Réutiliser les résultats déjà présents

```powershell
python -m src.cli.main --all --skip-existing
```

### Désactiver le tuning ou les figures

```powershell
python -m src.cli.main --models perceptron svm --no-tuning --no-figures
```

## Workflow recommandé

### 1. Génération des artefacts canoniques

Utiliser la CLI pour peupler ou mettre à jour `results/`.

Cette étape est la base reproductible du projet. Les notebooks et le rapport doivent s'appuyer sur ces artefacts, et non recalculer ou recopier manuellement des valeurs importantes.

### 2. Analyse notebook

Ordre recommandé :

1. `01_Exploratory_Data_Analysis.ipynb`
2. `02` à `07` pour les analyses par modèle
3. `08_Comparaison_Globale.ipynb`

Les notebooks ne constituent pas la source de vérité principale des métriques. Ils lisent et interprètent les artefacts canoniques produits dans `results/`.

### 3. Rapport

Les sections du rapport en Markdown sont éditées dans :

```text
docs/rapport_parts/
```

Le build Markdown/HTML/PDF s'appuie sur :

```powershell
python scripts\build_report_pdf.py
```

Le livrable LaTeX se trouve dans :

```text
docs/IFT712_Rapport/rapport/main.tex
```

## Source de vérité et conventions

### Artefacts canoniques

La source de vérité scientifique du dépôt est :

- `results/<modele>/...`
- `results/global/...`

En particulier :

- les métriques confirmatoires finales vivent dans `results/<modele>/metrics/tuned.json`
- les prédictions servant aux analyses d'erreurs vivent dans `results/<modele>/predictions/`
- les sorties de comparaison globale vivent dans `results/global/comparison/`

### Ce qui ne doit plus servir de source primaire

- `results/metrics/` plat, conservé seulement comme reliquat de migration
- les anciens artefacts sous `models/`
- les PDF, HTML et Markdown fusionnés déjà générés dans `docs/`

### Statut méthodologique

- les variantes non `tuned` sont **exploratoires**
- la variante `tuned` en `nested_cv` porte l'argument **confirmatoire**
- `secondary_holdout` est une **corroboration secondaire uniquement**

Le split gelé ne doit jamais être utilisé pour redéfinir le classement principal des modèles.

## Vérification et qualité

### Vérification de préparation du rapport

```powershell
python scripts\report_readiness_check.py
```

Ce script vérifie notamment :
- la présence et la cohérence des artefacts `tuned`
- la structure attendue des notebooks du rapport
- l'absence de certains tokens ou headings interdits

### Tests

```powershell
pytest -q
```

Si tu veux cibler les tests les plus liés au pipeline de rapport :

```powershell
pytest tests\test_report_readiness.py tests\test_experiments.py tests\test_methodology_helpers.py -q
```

## Rapport et artefacts générés

### À éditer manuellement

- `docs/rapport_parts/*.md`
- `docs/IFT712_Rapport/rapport/main.tex`

### À considérer comme générés

- `docs/rapport_parts_pdf/`
- `docs/rapport_leaf_classification*.md`
- `docs/rapport_leaf_classification*.html`
- `docs/rapport_leaf_classification*.pdf`
- une grande partie de `docs/figures/` lorsqu'elle est régénérée à partir des artefacts canoniques

Autrement dit : on édite les sources, pas les exports.

## Notes importantes

- L'EDA historique est conservé et réutilisé.
- Les notebooks s'appuient sur les réexports exposés par `src/__init__.py` et `src/artifacts/__init__.py` pour garder des imports stables.
- Les figures `exploratory__` restent descriptives et ne servent pas de preuve confirmatoire.
- Le dépôt contient à la fois un pipeline de rapport Markdown et une publication LaTeX finale ; dans les deux cas, les chiffres importants doivent venir des artefacts canoniques de `results/`.

## Résumé opérationnel

Si tu veux juste refaire tourner le projet proprement :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.cli.main --all --figures core
python scripts\report_readiness_check.py
jupyter notebook
```

Si tu veux juste reconstruire le rapport Markdown :

```powershell
python scripts\build_report_pdf.py
```
