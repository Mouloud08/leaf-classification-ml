"""Couche d'orchestration des notebooks classifieurs prêts pour le rapport.

Fusionne les champs d'affichage propres aux notebooks (titre, prose,
diagnostics) avec les spécifications d'étude exécutables définies dans
``experiments.model_specs``, et expose des fonctions utilitaires appelées
cellule par cellule.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from .artifacts import (
    UntunedVariantResult,
    charger_predictions_modele,
)
from .config import (
    CV_N_JOBS,
    N_SPLITS,
    RANDOM_SEED,
    RESULTS_DIR,
    TIMING_POLICY,
)
from .data import charger_donnees_modelisation
from .evaluation import (
    calculer_metriques_par_classe,
    evaluer_dummy_classifier,
    predictions_oof,
    top_confusions,
)
from .evaluation.diagnostics_specifiques import importance_oob_foret
from .evaluation.sanity_checks import verifier_bruit_aleatoire

# Ré-export des spécifications d'étude depuis ``experiments``
from .experiments.model_specs import obtenir_model_study_spec
from .experiments.shared import (
    construire_estimateur_variante,
    calculer_metriques_complementarite_erreurs,
    creer_cv,
    preparer_tuning_modele,
)
from .experiments.study_spec import UntunedVariantSpec
from .models import MODELES

# ---------------------------------------------------------------------------
# Interne : champs d'affichage propres aux notebooks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _NotebookDisplaySpec:
    """Champs propres aux notebooks, fusionnés à la volée avec ModelStudySpec."""

    title: str
    objective: str
    experiment_design: tuple[str, ...]
    expected_signal: tuple[str, ...]
    exploratory_diagnostics: tuple[str, ...] = ()


_NOTEBOOK_DISPLAY_SPECS: dict[str, _NotebookDisplaySpec] = {
    "perceptron": _NotebookDisplaySpec(
        title="02 - Modèle Perceptron",
        objective=(
            "Mesurer le plafond d'une référence linéaire minimale et l'effet "
            "de la mise à l'échelle, de la PCA et de l'ajustement."
        ),
        experiment_design=(
            "Le Perceptron sert de test de séparabilité linéaire rapide.",
            "La mise à l'échelle est traitée comme un prérequis expérimental, "
            "pas comme un détail de confort.",
            "La PCA reste une variante exploratoire à confirmer uniquement "
            "via la validation croisée imbriquée.",
        ),
        expected_signal=(
            "Un grand saut entre default et variantes scalées confirme que "
            "l'échelle des variables dominait le comportement du modèle.",
            "Un gain faible après ajustement ou PCA reste compatible avec "
            "une référence linéaire à capacité limitée.",
        ),
        exploratory_diagnostics=("tuning_heatmap",),
    ),
    "regression_logistique": _NotebookDisplaySpec(
        title="03 - Modèle Régression Logistique",
        objective=(
            "Établir le modèle probabiliste de référence et documenter ses "
            "performances confirmatoires."
        ),
        experiment_design=(
            "La régression logistique est la référence linéaire probabiliste du projet.",
            "La mise à l'échelle est obligatoire pour un conditionnement "
            "numérique stable.",
            "Les probabilités out-of-fold servent à documenter la calibration, "
            "le top-k et la confiance prédictive.",
        ),
        expected_signal=(
            "La mise à l'échelle doit apporter un gain majeur par rapport à "
            "la variante brute.",
            "Une calibration exploitable doit être visible uniquement sur "
            "les probabilités OOF de la validation croisée imbriquée.",
        ),
        exploratory_diagnostics=("tuning_heatmap", "probabilities"),
    ),
    "svm": _NotebookDisplaySpec(
        title="04 - Modèle SVM RBF",
        objective=(
            "Mesurer la valeur ajoutée d'un noyau RBF sur des données "
            "correctement mises à l'échelle."
        ),
        experiment_design=(
            "Le SVM RBF reste sensible à l'échelle ; toutes les variantes "
            "prétraitées passent par StandardScaler.",
            "La sélection finale conserve probability=True pour que le coût "
            "temporel publié inclue l'estimation probabiliste.",
            "Le diagnostic principal de l'ajustement reste la surface C x "
            "gamma, lue séparément par état PCA.",
        ),
        expected_signal=(
            "La mise à l'échelle est un prérequis ; sans elle, le SVM doit "
            "rester nettement moins performant.",
            "Le noyau RBF n'est justifié scientifiquement que si le gain "
            "confirmé compense son coût en temps.",
        ),
        exploratory_diagnostics=("tuning_heatmap",),
    ),
    "foret_aleatoire": _NotebookDisplaySpec(
        title="05 - Modèle Forêt Aléatoire",
        objective=(
            "Quantifier la robustesse d'un ensemble d'arbres sans "
            "prétraitement et documenter son interprétabilité exploratoire."
        ),
        experiment_design=(
            "La forêt aléatoire est étudiée sans mise à l'échelle ni PCA.",
            "La variante de profondeur restreinte sert de contrôle simple de "
            "complexité, pas de référence améliorée officielle.",
            "Les importances publiées dans le notebook doivent venir d'un "
            "diagnostic OOB, et rester exploratoires.",
        ),
        expected_signal=(
            "Une forte performance sans prétraitement confirmerait la "
            "robustesse du modèle sur ce tabulaire multiclasses.",
            "Les importances OOB ne doivent servir qu'à illustrer des "
            "tendances, pas à inférer une causalité.",
        ),
        exploratory_diagnostics=("permutation_importance",),
    ),
    "arbre_decision": _NotebookDisplaySpec(
        title="06 - Modèle Arbre de décision",
        objective="Documenter le compromis performance/complexité d'un arbre unique et ses limites de généralisation.",
        experiment_design=(
            "L'arbre de décision est analysé comme modèle interprétable mais instable.",
            "La variante de profondeur restreinte reste un contrôle de complexité simple.",
            "L'ajustement étudie explicitement la profondeur, la taille de "
            "feuille et l'élagage par complexité de coût.",
        ),
        expected_signal=(
            "L'arbre unique devrait rester derrière la forêt en performance et en stabilité inter-plis.",
            "Sa valeur scientifique principale est interprétative : "
            "profondeur, élagage et principales erreurs doivent raconter la "
            "limite de généralisation.",
        ),
        exploratory_diagnostics=("complexity_curve",),
    ),
    "mlp": _NotebookDisplaySpec(
        title="07 - Modèle MLP",
        objective=(
            "Tester la sensibilité d'un réseau dense au prétraitement, à "
            "l'ajustement et à la variabilité inter-plis."
        ),
        experiment_design=(
            "Le MLP requiert une mise à l'échelle stricte pour stabiliser l'optimisation.",
            "La PCA est laissée en compétition interne mais ne doit être interprétée "
            "qu'à travers la validation croisée imbriquée.",
            "La courbe de perte et les diagnostics de confiance restent exploratoires "
            "s'ils proviennent d'un réapprentissage global.",
        ),
        expected_signal=(
            "Sans mise à l'échelle, le MLP doit s'effondrer ou devenir très "
            "instable ; avec elle, il doit redevenir compétitif.",
            "L'ajustement ne mérite une conclusion positive que si la "
            "validation croisée imbriquée confirme un gain au-delà des "
            "variantes exploratoires bien prétraitées.",
        ),
        exploratory_diagnostics=("tuning_heatmap", "loss_curve", "probabilities"),
    ),
}


# ---------------------------------------------------------------------------
# Public : spécification fusionnée, contexte et fonctions de recherche
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpecificationModeleNotebook:
    """Configuration complète d'un notebook pour un classifieur donné."""

    nom_modele: str
    titre: str
    objectif: str
    design_experimental: tuple[str, ...]
    signal_attendu: tuple[str, ...]
    variantes_non_tunees: tuple[UntunedVariantSpec, ...]
    tuning_avec_pipeline: bool
    probabilites_disponibles: bool = False
    axes_heatmap_tuning: tuple[str, str] | None = None
    colonne_facette_tuning: str | None = None
    diagnostics_exploratoires: tuple[str, ...] = ()

    @property
    def model_name(self) -> str:
        return self.nom_modele

    @property
    def title(self) -> str:
        return self.titre

    @property
    def objective(self) -> str:
        return self.objectif

    @property
    def experiment_design(self) -> tuple[str, ...]:
        return self.design_experimental

    @property
    def expected_signal(self) -> tuple[str, ...]:
        return self.signal_attendu

    @property
    def untuned_variants(self) -> tuple[UntunedVariantSpec, ...]:
        return self.variantes_non_tunees

    @property
    def tuning_uses_pipeline(self) -> bool:
        return self.tuning_avec_pipeline

    @property
    def supports_probabilities(self) -> bool:
        return self.probabilites_disponibles

    @property
    def tuning_heatmap_axes(self) -> tuple[str, str] | None:
        return self.axes_heatmap_tuning

    @property
    def tuning_facet_column(self) -> str | None:
        return self.colonne_facette_tuning

    @property
    def exploratory_diagnostics(self) -> tuple[str, ...]:
        return self.diagnostics_exploratoires


NotebookModelSpec = SpecificationModeleNotebook


@dataclass(frozen=True)
class ContexteNotebook:
    """Contexte d'exécution partagé entre les cellules d'un notebook."""

    specification: SpecificationModeleNotebook
    definition_modele: dict[str, Any]
    racine_sortie: Path
    X: pd.DataFrame
    y: Any
    X_corroboration: pd.DataFrame
    y_corroboration: Any
    label_encoder: Any
    metadonnees_split: dict[str, Any]
    cv_evaluation: StratifiedKFold
    cv_tuning: StratifiedKFold
    politique_temps: str = TIMING_POLICY

    @property
    def spec(self) -> SpecificationModeleNotebook:
        return self.specification

    @property
    def definition(self) -> dict[str, Any]:
        return self.definition_modele

    @property
    def output_root(self) -> Path:
        return self.racine_sortie

    @property
    def X_holdout(self) -> pd.DataFrame:
        return self.X_corroboration

    @property
    def y_holdout(self) -> Any:
        return self.y_corroboration

    @property
    def split_metadata(self) -> dict[str, Any]:
        return self.metadonnees_split

    @property
    def cv_eval(self) -> StratifiedKFold:
        return self.cv_evaluation

    @property
    def timing_policy(self) -> str:
        return self.politique_temps


NotebookContext = ContexteNotebook


def obtenir_specification_modele_notebook(model_name: str) -> SpecificationModeleNotebook:
    """Retourne la configuration partagée de notebook pour un modèle."""
    key = model_name.lower()
    if key not in _NOTEBOOK_DISPLAY_SPECS:
        raise KeyError(f"Configuration notebook inconnue: {model_name}")
    display = _NOTEBOOK_DISPLAY_SPECS[key]
    study_spec = obtenir_model_study_spec(key)
    return SpecificationModeleNotebook(
        nom_modele=study_spec.model_name,
        titre=display.title,
        objectif=display.objective,
        design_experimental=display.experiment_design,
        signal_attendu=display.expected_signal,
        variantes_non_tunees=study_spec.untuned_variants,
        tuning_avec_pipeline=study_spec.tuning_uses_pipeline,
        probabilites_disponibles=study_spec.supports_probabilities,
        axes_heatmap_tuning=study_spec.tuning_heatmap_axes,
        colonne_facette_tuning=study_spec.tuning_facet_column,
        diagnostics_exploratoires=display.exploratory_diagnostics,
    )


# Dictionnaire public calculé, rétrocompatible avec les scripts et notebooks
# qui itèrent sur NOTEBOOK_MODEL_SPECS.
SPECIFICATIONS_MODELES_NOTEBOOK: dict[str, SpecificationModeleNotebook] = {
    key: obtenir_specification_modele_notebook(key)
    for key in _NOTEBOOK_DISPLAY_SPECS
}

NOTEBOOK_MODEL_SPECS = SPECIFICATIONS_MODELES_NOTEBOOK
obtenir_notebook_model_spec = obtenir_specification_modele_notebook


def creer_contexte_notebook(
    model_name: str,
    *,
    output_root: str | Path | None = None,
) -> ContexteNotebook:
    """Construit le contexte d'exécution partagé d'un notebook classifieur."""
    spec = obtenir_specification_modele_notebook(model_name)
    definition = MODELES[model_name.lower()]().definition()
    racine_sortie = Path(output_root) if output_root is not None else RESULTS_DIR
    split = charger_donnees_modelisation(output_root=racine_sortie)
    cv = creer_cv(n_splits=N_SPLITS, random_seed=RANDOM_SEED)
    return ContexteNotebook(
        specification=spec,
        definition_modele=definition,
        racine_sortie=racine_sortie,
        X=split.X_dev,
        y=split.y_dev,
        X_corroboration=split.X_corroboration,
        y_corroboration=split.y_corroboration,
        label_encoder=split.label_encoder,
        metadonnees_split=split.metadata,
        cv_evaluation=cv,
        cv_tuning=cv,
    )


creer_notebook_context = creer_contexte_notebook


# ---------------------------------------------------------------------------
# Utilitaires d'exécution des notebooks
# ---------------------------------------------------------------------------

def evaluer_variantes_untuned(
    contexte: ContexteNotebook,
) -> list[UntunedVariantResult]:
    """Évalue et persiste toutes les variantes non tunées du notebook."""
    from .experiments.runner import executer_variantes_untuned as _run_untuned
    from .experiments.study_spec import StudySpec

    study = StudySpec(
        models=[contexte.spec.model_name],
        output_root=contexte.output_root,
    )
    spec = obtenir_model_study_spec(contexte.spec.model_name)
    return _run_untuned(spec, contexte.X, contexte.y, study)


def executer_tuning_confirmatoire(
    contexte: ContexteNotebook,
    resultats_untuned: list[UntunedVariantResult] | None = None,
) -> dict[str, Any]:
    """Exécute le workflow partagé de tuning confirmatoire d'un notebook."""
    from .experiments.runner import executer_tuning_bundle
    from .experiments.study_spec import StudySpec

    model_name = contexte.spec.model_name
    study_spec = obtenir_model_study_spec(model_name)
    study = StudySpec(
        models=[model_name],
        n_splits=contexte.cv_tuning.n_splits,
        random_seed=contexte.cv_tuning.random_state,
        output_root=contexte.output_root,
    )

    bundle = executer_tuning_bundle(
        study_spec,
        X=contexte.X,
        y=contexte.y,
        label_encoder=contexte.label_encoder,
        untuned_results=resultats_untuned or evaluer_variantes_untuned(contexte),
        X_holdout=contexte.X_holdout,
        y_holdout=contexte.y_holdout,
        split_metadata=contexte.split_metadata,
        study=study,
    )
    return {
        key: value
        for key, value in bundle.items()
        if key != "tuning_result"
    }


def charger_ou_generer_predictions_variante(
    model_name: str,
    variante: str,
    X: pd.DataFrame,
    y: Any,
    label_encoder: Any | None = None,
    *,
    root: str | Path | None = None,
) -> pd.DataFrame:
    """Retourne un tableau standard de prédictions pour une variante.

    Les prédictions confirmatoires de la variante ajustée sont chargées depuis
    les artefacts canoniques. Les variantes non ajustées sont recalculées en
    prédictions hors pli avec le protocole partagé de validation croisée simple
    afin que les notebooks puissent comparer les matrices de confusion sur
    toutes les variantes.
    """
    racine = Path(root) if root is not None else None
    try:
        return charger_predictions_modele(model_name, variante, root=racine)
    except FileNotFoundError:
        pass

    spec = obtenir_model_study_spec(model_name)
    variante_norm = variante.lower()
    variante_spec = next(
        (v for v in spec.untuned_variants if v.name.lower() == variante_norm),
        None,
    )
    if variante_spec is None:
        raise FileNotFoundError(
            f"Aucune prédiction disponible et variante inconnue : "
            f"{model_name}/{variante}"
        )

    estimateur = construire_estimateur_variante(model_name, variante_spec)
    return predictions_oof(
        estimateur,
        X,
        y,
        label_encoder=label_encoder,
    )


# ---------------------------------------------------------------------------
# Constructeurs de tableaux pour notebooks
# ---------------------------------------------------------------------------

def construire_tableau_variantes(
    resultats_untuned: list[UntunedVariantResult],
    mesures_tuned: dict[str, Any],
) -> pd.DataFrame:
    """Construit le tableau synthèse sûr pour le rapport des variantes."""
    from .evaluation.diagnostics_generiques import tableau_variantes
    return tableau_variantes(
        [r.to_dict() for r in resultats_untuned], mesures_tuned,
    )


def construire_tableau_corroboration_secondaire(
    mesures_tuned: dict[str, Any],
) -> pd.DataFrame:
    """Construit un tableau compact variante ajustée vs référence de corroboration."""
    secondary_holdout = mesures_tuned.get("secondary_holdout", {})
    if not isinstance(secondary_holdout, dict) or not secondary_holdout.get("enabled"):
        return pd.DataFrame()

    reference = secondary_holdout.get("reference", {})
    tuned = secondary_holdout.get("tuned", {})
    tableau = pd.DataFrame(
        [
            {
                "variante": reference.get("variant", secondary_holdout.get("reference_variant")),
                "f1_macro_holdout": reference.get("f1_macro"),
                "accuracy_holdout": reference.get("accuracy"),
                "role": "reference",
            },
            {
                "variante": "tuned",
                "f1_macro_holdout": tuned.get("f1_macro"),
                "accuracy_holdout": tuned.get("accuracy"),
                "role": "tuned",
            },
        ]
    )
    tableau["corroboration_scope"] = secondary_holdout.get(
        "corroboration_scope", "corroboration_only"
    )
    tableau["ranking_eligible"] = bool(secondary_holdout.get("ranking_eligible", False))
    tableau["origin"] = secondary_holdout.get("origin", "exploratory_refit")
    tableau["delta_f1_vs_reference"] = (
        tableau["f1_macro_holdout"] - float(reference.get("f1_macro", np.nan))
    )
    return tableau


construire_tableau_holdout_secondaire = construire_tableau_corroboration_secondaire


def construire_tableau_corroboration_descriptive_variantes(
    mesures_tuned: dict[str, Any],
) -> pd.DataFrame:
    """Retourne toutes les variantes publiées pour la corroboration descriptive."""
    lignes = mesures_tuned.get("descriptive_holdout_all_variants", [])
    if not isinstance(lignes, list) or not lignes:
        return pd.DataFrame()

    tableau = pd.DataFrame(lignes).rename(
        columns={
            "f1_macro": "f1_macro_holdout",
            "accuracy": "accuracy_holdout",
        }
    )
    if tableau.empty or "variante" not in tableau.columns:
        return pd.DataFrame()

    reference_rows = tableau.loc[tableau["role"] == "reference", "f1_macro_holdout"]
    reference_f1 = float(reference_rows.iloc[0]) if not reference_rows.empty else np.nan
    tableau["delta_f1_vs_reference"] = (
        tableau["f1_macro_holdout"] - reference_f1
    )
    return ordonner_tableau_variantes(tableau)


construire_tableau_holdout_descriptif_variantes = (
    construire_tableau_corroboration_descriptive_variantes
)


def ordonner_tableau_variantes(tableau: pd.DataFrame) -> pd.DataFrame:
    """Ordonne les variantes de façon stable et lisible pour le rapport."""
    if tableau.empty or "variante" not in tableau.columns:
        return tableau.copy()

    ordre_variantes = {
        "default": 0,
        "scaled_only": 1,
        "scaled_pca": 2,
        "restricted_depth": 3,
        "tuned": 99,
    }
    return (
        tableau.copy()
        .assign(
            _ordre=lambda df: df["variante"].astype("string").map(
                lambda v: ordre_variantes.get(str(v), 50)
            )
        )
        .sort_values(["_ordre", "variante"])
        .drop(columns="_ordre")
        .reset_index(drop=True)
    )


def charger_tableau_variantes_modele(
    nom_modele: str,
    root: Path | None = None,
) -> pd.DataFrame:
    """Charge toutes les métriques canoniques d'un modèle depuis `results/`."""
    metrics_dir = Path(root or RESULTS_DIR) / nom_modele / "metrics"
    lignes: list[dict[str, Any]] = []
    for chemin in sorted(metrics_dir.glob("*.json")):
        with open(chemin, encoding="utf-8") as fh:
            contenu = json.load(fh)
        lignes.append({"variante": chemin.stem, **contenu})
    tableau = pd.DataFrame(lignes)
    if tableau.empty:
        raise FileNotFoundError(
            f"Aucune métrique canonique trouvée pour {nom_modele}."
        )
    return ordonner_tableau_variantes(tableau)


def construire_tableau_variantes_rapport(tableau: pd.DataFrame) -> pd.DataFrame:
    """Retourne un tableau compact de variantes prêt pour le rapport."""
    if tableau.empty:
        return tableau.copy()

    statut = tableau.get("evaluation_stage", pd.Series(index=tableau.index, dtype="object")).map(
        {
            "baseline": "exploratoire",
            "improved_untuned": "exploratoire",
            "tuned": "confirmatoire",
        }
    )
    colonnes = [
        "variante",
        "val_f1_macro_mean",
        "val_accuracy_mean",
        "evaluation_stage",
    ]
    present = [c for c in colonnes if c in tableau.columns]
    retour = tableau.loc[:, present].copy()
    retour["statut"] = statut.fillna("")
    retour = retour.rename(
        columns={
            "val_f1_macro_mean": "f1_macro",
            "val_accuracy_mean": "accuracy",
        }
    )
    return retour


def construire_resume_confusions_variantes(
    model_name: str,
    summary_table: pd.DataFrame,
    X: pd.DataFrame,
    y: Any,
    label_encoder: Any,
) -> pd.DataFrame:
    """Construit un résumé compact par variante avant les matrices de confusion."""
    ordre_variantes = {
        "default": 0,
        "scaled_only": 1,
        "scaled_pca": 2,
        "restricted_depth": 3,
        "tuned": 99,
    }
    variantes_str = summary_table["variante"].astype("string")
    catalogue_variantes = (
        summary_table[
            [
                "variante",
                "evaluation_stage",
                "val_f1_macro_mean",
                "val_accuracy_mean",
                "generalization_gap_f1_macro",
            ]
        ]
        .dropna(subset=["variante"])
        .drop_duplicates(subset=["variante", "evaluation_stage"])
        .assign(
            variante=variantes_str,
            _ordre=lambda df: df["variante"].map(lambda v: ordre_variantes.get(str(v), 50)),
        )
        .sort_values(["_ordre", "variante"])
        .reset_index(drop=True)
    )

    lignes: list[dict[str, Any]] = []
    for ligne in catalogue_variantes.to_dict("records"):
        variante = str(ligne["variante"])
        predictions_variante = charger_ou_generer_predictions_variante(
            model_name,
            variante,
            X,
            y,
            label_encoder=label_encoder,
        )
        confusion = top_confusions(
            predictions_variante["y_true"].to_numpy(),
            predictions_variante["y_pred"].to_numpy(),
            label_encoder=label_encoder,
            top_n=1,
        )
        if confusion.empty:
            confusion_dominante = "aucune"
            erreurs_dominantes = 0
        else:
            premiere = confusion.iloc[0]
            confusion_dominante = (
                f"{premiere['classe_vraie']} -> {premiere['classe_predite']}"
            )
            erreurs_dominantes = int(premiere["nombre_erreurs"])

        lignes.append(
            {
                "variante": variante,
                "evaluation_stage": ligne["evaluation_stage"],
                "val_f1_macro_mean": ligne["val_f1_macro_mean"],
                "val_accuracy_mean": ligne["val_accuracy_mean"],
                "generalization_gap_f1_macro": ligne["generalization_gap_f1_macro"],
                "confusion_dominante": confusion_dominante,
                "nombre_erreurs_dominantes": erreurs_dominantes,
            }
        )

    return pd.DataFrame(lignes)


def construire_tableau_gaps(
    resultats_untuned: list[UntunedVariantResult],
) -> pd.DataFrame:
    """Retourne le tableau des écarts de généralisation non tunés."""
    from .evaluation.diagnostics_generiques import tableau_gaps
    return tableau_gaps([r.to_dict() for r in resultats_untuned])


def construire_tableau_plis_externes(nested_tuned: dict[str, Any]) -> pd.DataFrame:
    """Retourne les métriques des plis externes dans un format notebook."""
    return nested_tuned["outer_fold_metrics"].copy()


def calculer_metriques_classes_depuis_predictions(
    predictions_df: pd.DataFrame,
    label_encoder: Any,
) -> pd.DataFrame:
    """Calcule les métriques par classe depuis un tableau de prédictions."""
    return calculer_metriques_par_classe(
        predictions_df["y_true"].to_numpy(),
        predictions_df["y_pred"].to_numpy(),
        label_encoder=label_encoder,
    )


def calculer_importance_permutation_exploratoire(
    estimateur: Any,
    X: pd.DataFrame,
    y: Any,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Ajuste un estimateur exploratoire et calcule l'importance par permutation."""
    modele = clone(estimateur)
    modele.fit(X, y)
    importance = permutation_importance(
        modele,
        X,
        y,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        n_jobs=CV_N_JOBS,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def calculer_importance_permutation_oob_foret(
    estimateur: Any,
    X: pd.DataFrame,
    y: Any,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Expose le diagnostic d'importance OOB pour le notebook foret."""
    return importance_oob_foret(estimateur, X, y, n_repeats=n_repeats)


def calculer_importance_oob_foret_exploratoire(
    estimateur: Any,
    X: pd.DataFrame,
    y: Any,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Calcule une importance OOB exploratoire specifique a la foret aleatoire."""
    return importance_oob_foret(
        estimateur=estimateur,
        X=X,
        y=y,
        n_repeats=n_repeats,
    )


def construire_tableau_plis_tuned_global(
    X: pd.DataFrame,
    y: Any,
    predictions_par_modele: dict[str, pd.DataFrame],
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Reconstruit les métriques par pli externe à partir des prédictions OOF tuned."""
    from sklearn.metrics import accuracy_score, f1_score

    cv = creer_cv(n_splits=n_splits, random_seed=random_state)
    y_array = y.to_numpy() if hasattr(y, "to_numpy") else pd.Series(y).to_numpy()

    lignes: list[dict[str, Any]] = []
    for index_pli, (_, test_idx) in enumerate(cv.split(X, y_array), start=1):
        y_true_pli = y_array[test_idx]
        for modele, predictions in predictions_par_modele.items():
            y_pred_pli = predictions["y_pred"].to_numpy()[test_idx]
            lignes.append(
                {
                    "fold": index_pli,
                    "modele": modele,
                    "val_accuracy": float(accuracy_score(y_true_pli, y_pred_pli)),
                    "val_f1_macro": float(
                        f1_score(y_true_pli, y_pred_pli, average="macro")
                    ),
                }
            )

    return pd.DataFrame(lignes)


def construire_tableau_corroboration_globale(
    modeles: list[str] | tuple[str, ...],
    root: Path | None = None,
) -> pd.DataFrame:
    """Agrège les corroborations disponibles pour les modèles éligibles."""
    lignes: list[dict[str, Any]] = []
    for nom_modele in modeles:
        chemin = Path(root or RESULTS_DIR) / nom_modele / "metrics" / "tuned.json"
        if not chemin.exists():
            continue
        contenu = json.loads(chemin.read_text(encoding="utf-8"))
        secondary_holdout = contenu.get("secondary_holdout", {})
        if not isinstance(secondary_holdout, dict) or not secondary_holdout.get("enabled"):
            continue
        lignes.append(
            {
                "modele": nom_modele,
                "reference_variant": secondary_holdout.get("reference_variant"),
                "nested_cv_f1": contenu.get("val_f1_macro_mean"),
                "holdout_tuned_f1": secondary_holdout.get("tuned", {}).get("f1_macro"),
                "holdout_reference_f1": secondary_holdout.get("reference", {}).get("f1_macro"),
                "delta_holdout": (
                    secondary_holdout.get("tuned", {}).get("f1_macro", np.nan)
                    - secondary_holdout.get("reference", {}).get("f1_macro", np.nan)
                ),
            }
        )
    return pd.DataFrame(lignes)


construire_tableau_holdout_global = construire_tableau_corroboration_globale


def construire_tableau_complementarite_erreurs(
    predictions_par_modele: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare le recouvrement des erreurs entre modèles via les prédictions OOF."""
    lignes: list[dict[str, Any]] = []
    modeles = list(predictions_par_modele.keys())
    for index_a, modele_a in enumerate(modeles):
        for modele_b in modeles[index_a + 1 :]:
            metriques = calculer_metriques_complementarite_erreurs(
                predictions_par_modele[modele_a],
                predictions_par_modele[modele_b],
            )
            lignes.append({"modele_a": modele_a, "modele_b": modele_b, **metriques})

    return pd.DataFrame(lignes)


def extraire_images_notebook(
    notebook_path: str | Path,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
    formats: tuple[str, ...] = ("image/png", "image/jpeg", "image/svg+xml"),
) -> list[Path]:
    """Extrait les images déjà embarquées dans un notebook Jupyter.

    La fonction lit les sorties persistées dans le fichier ``.ipynb`` et écrit
    chaque image trouvée sur disque. Elle n'exécute pas le notebook et ne peut
    donc pas recréer des figures absentes des sorties sauvegardées.
    """

    notebook = Path(notebook_path)
    contenu = json.loads(notebook.read_text(encoding="utf-8"))
    dossier_sortie = Path(output_dir) if output_dir is not None else notebook.with_suffix("")
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    extensions = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/svg+xml": "svg",
    }

    def _normaliser_payload(payload: Any) -> str:
        if isinstance(payload, list):
            return "".join(str(fragment) for fragment in payload)
        return str(payload)

    chemins: list[Path] = []
    nom_prefixe = prefix or notebook.stem

    for index_cellule, cellule in enumerate(contenu.get("cells", []), start=1):
        for index_sortie, sortie in enumerate(cellule.get("outputs", []), start=1):
            donnees = sortie.get("data")
            if not isinstance(donnees, dict):
                continue
            for mime in formats:
                if mime not in donnees:
                    continue

                extension = extensions[mime]
                chemin = (
                    dossier_sortie
                    / f"{nom_prefixe}_cell{index_cellule:03d}_out{index_sortie:02d}.{extension}"
                )
                payload = _normaliser_payload(donnees[mime])

                if mime == "image/svg+xml":
                    chemin.write_text(payload, encoding="utf-8")
                else:
                    chemin.write_bytes(base64.b64decode(payload))

                chemins.append(chemin)
                break

    return chemins


def executer_verifications_validite(
    contexte: ContexteNotebook,
) -> dict[str, Any]:
    """Exécute les vérifications standards de validité scientifique d'un notebook."""
    study_spec = obtenir_model_study_spec(contexte.spec.model_name)
    estimateur_validite, _ = preparer_tuning_modele(study_spec)

    dummy = evaluer_dummy_classifier(
        contexte.X,
        contexte.y,
        n_splits=contexte.cv_eval.n_splits,
        random_state=contexte.cv_eval.random_state,
    )
    bruit = verifier_bruit_aleatoire(
        estimateur_validite,
        contexte.y,
        n_features=contexte.X.shape[1],
        random_state=contexte.cv_eval.random_state,
    )

    note = (
        "Les variantes non tunées restent exploratoires ; les figures réapprises "
        "sur le sous-ensemble de développement (80 %) aussi. Les prédictions "
        "hors pli de la variante ajustée sont des diagnostics descriptifs "
        "confirmatoires, mais seule la performance de la variante ajustée en validation croisée imbriquée sert au "
        "classement final inter-modèles."
    )

    return {
        "dummy": pd.DataFrame([dummy]).assign(reference="DummyClassifier"),
        "random_noise": pd.DataFrame([bruit]).assign(reference="Bruit aleatoire"),
        "note": note,
    }
