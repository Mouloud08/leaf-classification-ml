"""Orchestration layer for report-ready classifier notebooks.

Merges notebook-specific display fields (title, prose, diagnostics) with
executable study specs from ``experiments.model_specs``, and exposes
convenience functions that notebooks call cell-by-cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from .config import CV_N_JOBS, N_SPLITS, RANDOM_SEED, TIMING_POLICY
from .data import charger_donnees
from .evaluation import (
    calculer_metriques_par_classe,
    evaluer_modele_cv,
    evaluer_modele_tuned_nested_cv,
)
from .models import MODELES
from .artifacts import (
    UntunedVariantResult,
    model_paths,
    sauvegarder_mesures,
    sauvegarder_predictions,
    sauvegarder_probabilites_oof,
    sauvegarder_resultats_grid_search,
)

# Re-export study specs from experiments
from .experiments.model_specs import obtenir_model_study_spec
from .experiments.shared import (
    ajouter_gaps,
    construire_estimateur_variante,
    construire_tuning_run_result,
    creer_cv,
    executer_grid_search_exploratoire,
    preparer_tuning_modele,
)
from .experiments.study_spec import UntunedVariantSpec


# ---------------------------------------------------------------------------
# Internal: notebook-only display fields
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _NotebookDisplaySpec:
    """Notebook-only fields — merged with ModelStudySpec at lookup time."""

    title: str
    objective: str
    experiment_design: tuple[str, ...]
    exploratory_diagnostics: tuple[str, ...] = ()


_NOTEBOOK_DISPLAY_SPECS: dict[str, _NotebookDisplaySpec] = {
    "perceptron": _NotebookDisplaySpec(
        title="02 - Modele Perceptron",
        objective="Mesurer le plafond d'une baseline lineaire minimale et l'effet du scaling, de la PCA et du tuning.",
        experiment_design=(
            "Le Perceptron sert de test de separabilite lineaire rapide.",
            "Le scaling est traite comme un pre-requis experimental, pas comme un detail de confort.",
            "La PCA reste une variante exploratoire a confirmer uniquement via la selection nested CV.",
        ),
        exploratory_diagnostics=("tuning_heatmap",),
    ),
    "regression_logistique": _NotebookDisplaySpec(
        title="03 - Modele Regression Logistique",
        objective="Etablir la baseline probabiliste principale et documenter ses performances confirmatoires.",
        experiment_design=(
            "La regression logistique est la reference lineaire probabiliste du projet.",
            "Le scaling est obligatoire pour un conditionnement numerique stable.",
            "Les probabilites out-of-fold servent a documenter calibration, top-k et confiance predictive.",
        ),
        exploratory_diagnostics=("tuning_heatmap", "probabilities"),
    ),
    "svm": _NotebookDisplaySpec(
        title="04 - Modele SVM RBF",
        objective="Mesurer la valeur ajoutee d'un noyau RBF sur des donnees correctement scalees.",
        experiment_design=(
            "Le SVM RBF reste sensible a l'echelle; toutes les variantes pretraitees passent par StandardScaler.",
            "La selection finale conserve probability=True pour que le cout temporel publie inclue l'estimation probabiliste.",
            "Le diagnostic principal du tuning reste la surface C x gamma, lue separement par etat PCA.",
        ),
        exploratory_diagnostics=("tuning_heatmap",),
    ),
    "foret_aleatoire": _NotebookDisplaySpec(
        title="05 - Modele Foret Aleatoire",
        objective="Quantifier la robustesse d'un ensemble d'arbres sans preprocessing et documenter son interpretabilite exploratoire.",
        experiment_design=(
            "La foret aleatoire est etudiee sans scaling ni PCA.",
            "La variante de profondeur restreinte sert de sanity check de complexite, pas de baseline amelioree officielle.",
            "Les importances publiees dans le notebook doivent venir de permutation importance, et rester exploratoires.",
        ),
        exploratory_diagnostics=("permutation_importance",),
    ),
    "arbre_decision": _NotebookDisplaySpec(
        title="06 - Modele Arbre de decision",
        objective="Documenter le compromis performance/complexite d'un arbre unique et ses limites de generalisation.",
        experiment_design=(
            "L'arbre de decision est analyse comme modele interpretable mais instable.",
            "La variante de profondeur restreinte reste un controle de complexite simple.",
            "Le tuning etudie explicitement la profondeur, la taille de feuille et l'elagage cost-complexity.",
        ),
        exploratory_diagnostics=("complexity_curve", "feature_importance"),
    ),
    "mlp": _NotebookDisplaySpec(
        title="07 - Modele MLP",
        objective="Tester la sensibilite d'un reseau dense au preprocessing, au tuning et a la variabilite inter-plis.",
        experiment_design=(
            "Le MLP requiert un scaling strict pour stabiliser l'optimisation.",
            "La PCA est laissee en competition interne mais ne doit etre interpretee qu'a travers la nested CV.",
            "La loss curve et les diagnostics de confiance restent exploratoires s'ils proviennent d'un refit global.",
        ),
        exploratory_diagnostics=("tuning_heatmap", "loss_curve", "probabilities"),
    ),
}


# ---------------------------------------------------------------------------
# Public: merged spec, context, and lookup
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NotebookModelSpec:
    """Decision-complete notebook configuration for one classifier."""

    model_name: str
    title: str
    objective: str
    experiment_design: tuple[str, ...]
    untuned_variants: tuple[UntunedVariantSpec, ...]
    tuning_uses_pipeline: bool
    supports_probabilities: bool = False
    tuning_heatmap_axes: tuple[str, str] | None = None
    tuning_facet_column: str | None = None
    exploratory_diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class NotebookContext:
    """Execution context shared by notebook cells."""

    spec: NotebookModelSpec
    definition: dict[str, Any]
    X: pd.DataFrame
    y: Any
    label_encoder: Any
    cv_eval: StratifiedKFold    # display only (n_splits) — evaluation functions create their own CV
    cv_tuning: StratifiedKFold  # n_splits and random_state used by tuning helpers
    timing_policy: str = TIMING_POLICY


def obtenir_notebook_model_spec(model_name: str) -> NotebookModelSpec:
    """Return the shared notebook configuration for a model."""
    key = model_name.lower()
    if key not in _NOTEBOOK_DISPLAY_SPECS:
        raise KeyError(f"Configuration notebook inconnue: {model_name}")
    display = _NOTEBOOK_DISPLAY_SPECS[key]
    study_spec = obtenir_model_study_spec(key)
    return NotebookModelSpec(
        model_name=study_spec.model_name,
        title=display.title,
        objective=display.objective,
        experiment_design=display.experiment_design,
        untuned_variants=study_spec.untuned_variants,
        tuning_uses_pipeline=study_spec.tuning_uses_pipeline,
        supports_probabilities=study_spec.supports_probabilities,
        tuning_heatmap_axes=study_spec.tuning_heatmap_axes,
        tuning_facet_column=study_spec.tuning_facet_column,
        exploratory_diagnostics=display.exploratory_diagnostics,
    )


# Computed public dict — backwards-compatible with scripts and notebooks
# that iterate over NOTEBOOK_MODEL_SPECS.
NOTEBOOK_MODEL_SPECS: dict[str, NotebookModelSpec] = {
    key: obtenir_notebook_model_spec(key) for key in _NOTEBOOK_DISPLAY_SPECS
}


def creer_notebook_context(model_name: str) -> NotebookContext:
    """Build the shared execution context for a classifier notebook."""
    spec = obtenir_notebook_model_spec(model_name)
    definition = MODELES[model_name.lower()]().definition()
    X, y, label_encoder = charger_donnees()
    cv = creer_cv(n_splits=N_SPLITS, random_seed=RANDOM_SEED)
    return NotebookContext(
        spec=spec,
        definition=definition,
        X=X,
        y=y,
        label_encoder=label_encoder,
        cv_eval=cv,
        cv_tuning=cv,
    )


# ---------------------------------------------------------------------------
# Notebook execution helpers
# ---------------------------------------------------------------------------

def evaluer_variantes_untuned(
    contexte: NotebookContext,
) -> list[UntunedVariantResult]:
    """Evaluate and persist every untuned variant defined for the notebook."""
    paths = model_paths(contexte.spec.model_name)
    resultats: list[UntunedVariantResult] = []
    for variante in contexte.spec.untuned_variants:
        estimateur = construire_estimateur_variante(
            contexte.spec.model_name,
            variante,
        )
        mesures = ajouter_gaps(evaluer_modele_cv(estimateur, contexte.X, contexte.y))
        selection_protocol = "simple_cv"
        evaluation_stage = (
            "baseline" if variante.name == "default" else "improved_untuned"
        )

        result = UntunedVariantResult(
            variante=variante.name,
            selection_protocol=selection_protocol,
            evaluation_stage=evaluation_stage,
            val_accuracy_mean=mesures["val_accuracy_mean"],
            val_accuracy_std=mesures["val_accuracy_std"],
            val_f1_macro_mean=mesures["val_f1_macro_mean"],
            val_f1_macro_std=mesures["val_f1_macro_std"],
            train_accuracy_mean=mesures["train_accuracy_mean"],
            train_f1_macro_mean=mesures["train_f1_macro_mean"],
            generalization_gap_accuracy=mesures["generalization_gap_accuracy"],
            generalization_gap_f1_macro=mesures["generalization_gap_f1_macro"],
            elapsed_seconds=mesures["elapsed_seconds"],
            timing_policy=contexte.timing_policy,
            n_splits=mesures["n_splits"],
            fit_time_mean=mesures["fit_time_mean"],
            score_time_mean=mesures["score_time_mean"],
        )

        sauvegarder_mesures(
            contexte.spec.model_name,
            variante.name,
            result.to_dict(),
            paths=paths,
        )
        resultats.append(result)
    return resultats


def executer_tuning_confirmatoire(
    contexte: NotebookContext,
) -> dict[str, Any]:
    """Run exploratory grid-search and confirmatory nested CV for a notebook."""
    model_name = contexte.spec.model_name
    paths = model_paths(model_name)
    study_spec = obtenir_model_study_spec(model_name)
    n_splits = contexte.cv_tuning.n_splits
    random_seed = contexte.cv_tuning.random_state

    grid, resultats_grid, temps_tuning = executer_grid_search_exploratoire(
        spec=study_spec,
        X=contexte.X,
        y=contexte.y,
        n_splits=n_splits,
        random_seed=random_seed,
    )
    chemin_grid = sauvegarder_resultats_grid_search(
        model_name, resultats_grid, paths=paths,
    )

    estimateur_grid, grille_tuning = preparer_tuning_modele(study_spec)
    nested_tuned = evaluer_modele_tuned_nested_cv(
        estimateur_ou_pipeline=estimateur_grid,
        param_grid=grille_tuning,
        X=contexte.X,
        y=contexte.y,
        label_encoder=contexte.label_encoder,
        outer_splits=n_splits,
        inner_splits=n_splits,
        random_state=random_seed,
        return_probabilities=contexte.spec.supports_probabilities,
    )

    mesures_tuned = construire_tuning_run_result(
        nested_tuned=nested_tuned,
        grid=grid,
        timing_policy=contexte.timing_policy,
        exploratory_tuning_seconds=temps_tuning,
    ).to_dict()
    chemin_tuned = sauvegarder_mesures(
        model_name, "tuned", mesures_tuned, paths=paths,
    )

    chemin_predictions = sauvegarder_predictions(
        model_name,
        "tuned",
        nested_tuned["oof_predictions"],
        paths=paths,
    )

    chemin_probabilites = None
    if nested_tuned.get("oof_probabilities") is not None:
        chemin_probabilites = sauvegarder_probabilites_oof(
            model_name,
            "tuned",
            nested_tuned["oof_probabilities"],
            paths=paths,
        )

    return {
        "grid": grid,
        "grid_results": resultats_grid,
        "grid_path": chemin_grid,
        "nested": nested_tuned,
        "metrics": mesures_tuned,
        "metrics_path": chemin_tuned,
        "predictions_path": chemin_predictions,
        "probabilities_path": chemin_probabilites,
        "exploratory_tuning_seconds": temps_tuning,
    }


# ---------------------------------------------------------------------------
# Notebook table builders
# ---------------------------------------------------------------------------

def construire_tableau_variantes(
    resultats_untuned: list[UntunedVariantResult],
    mesures_tuned: dict[str, Any],
) -> pd.DataFrame:
    """Build the report-safe summary table for notebook variants."""
    from .evaluation.diagnostics_generiques import tableau_variantes
    return tableau_variantes(
        [r.to_dict() for r in resultats_untuned], mesures_tuned,
    )


def construire_tableau_gaps(
    resultats_untuned: list[UntunedVariantResult],
) -> pd.DataFrame:
    """Return the untuned-only generalization gap table."""
    from .evaluation.diagnostics_generiques import tableau_gaps
    return tableau_gaps([r.to_dict() for r in resultats_untuned])


def construire_tableau_plis_externes(nested_tuned: dict[str, Any]) -> pd.DataFrame:
    """Return the nested outer-fold metrics in a notebook-friendly format."""
    return nested_tuned["outer_fold_metrics"].copy()


def calculer_metriques_classes_depuis_predictions(
    predictions_df: pd.DataFrame,
    label_encoder: Any,
) -> pd.DataFrame:
    """Compute per-class metrics from a standard prediction table."""
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
    """Fit an exploratory estimator on full data and compute permutation importance."""
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
