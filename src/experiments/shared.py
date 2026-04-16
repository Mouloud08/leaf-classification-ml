"""Utilitaires partagés pour les exécutions batch, la CLI et les notebooks."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from ..artifacts import TuningRunResult, UntunedVariantResult
from ..config import CV_N_JOBS
from ..models.pipelines import creer_estimateur, creer_pipeline, obtenir_grille_tuning
from .study_spec import ModelStudySpec, UntunedVariantSpec


def creer_cv(n_splits: int, random_seed: int) -> StratifiedKFold:
    """Construit la CV stratifiee standard du projet."""
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )


def construire_estimateur_variante(
    model_name: str,
    variante: UntunedVariantSpec,
) -> Any:
    """Construit l'estimateur effectif d'une variante declarative."""
    if variante.kind == "simple":
        estimateur = creer_estimateur(model_name)
    elif variante.kind == "pipeline_no_pca":
        estimateur = creer_pipeline(model_name, utiliser_pca=False)
    elif variante.kind == "pipeline_with_pca":
        estimateur = creer_pipeline(model_name, utiliser_pca=True)
    else:
        raise ValueError(f"Type de variante non supporte: {variante.kind}")

    if variante.params:
        estimateur = clone(estimateur)
        estimateur.set_params(**variante.params)
    return estimateur


def ajouter_gaps(mesures: dict[str, Any]) -> dict[str, Any]:
    """Ajoute les generalization gaps aux metriques de CV simple."""
    resultat = mesures.copy()
    resultat["generalization_gap_accuracy"] = (
        resultat["train_accuracy_mean"] - resultat["val_accuracy_mean"]
    )
    resultat["generalization_gap_f1_macro"] = (
        resultat["train_f1_macro_mean"] - resultat["val_f1_macro_mean"]
    )
    return resultat


def normaliser_parametre_pour_json(valeur: Any) -> Any:
    """Rend les meilleurs hyperparametres serialisables."""
    return repr(valeur) if hasattr(valeur, "get_params") else valeur


def _ordre_simplicite_variante(variante: str) -> int:
    ordre = {
        "default": 0,
        "scaled_only": 1,
        "scaled_pca": 2,
        "restricted_depth": 3,
    }
    return ordre.get(str(variante), 50)


def preparer_tuning_modele(
    spec: ModelStudySpec,
) -> tuple[Any, dict[str, list[Any]]]:
    """Construit l'estimateur et la grille de tuning d'un modele."""
    if spec.tuning_uses_pipeline:
        estimateur = creer_pipeline(spec.model_name, utiliser_pca=True)
        return estimateur, obtenir_grille_tuning(spec.model_name)
    return creer_estimateur(spec.model_name), obtenir_grille_tuning(
        spec.model_name,
        prefixer_modele=False,
    )


def executer_grid_search_exploratoire(
    spec: ModelStudySpec,
    X: pd.DataFrame,
    y: Any,
    n_splits: int,
    random_seed: int,
    n_jobs: int = CV_N_JOBS,
) -> tuple[GridSearchCV, pd.DataFrame, float]:
    """Exécute la recherche sur grille exploratoire partagée par notebooks et runner."""
    estimateur_grid, grille_tuning = preparer_tuning_modele(spec)
    cv_tuning = creer_cv(n_splits=n_splits, random_seed=random_seed)

    debut_tuning = perf_counter()
    grid = GridSearchCV(
        estimator=estimateur_grid,
        param_grid=grille_tuning,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        refit="f1_macro",
        cv=cv_tuning,
        n_jobs=n_jobs,
        return_train_score=True,
    )
    grid.fit(X, y)
    temps_tuning = perf_counter() - debut_tuning

    resultats_grid = pd.DataFrame(grid.cv_results_)
    resultats_grid["generalization_gap_accuracy"] = (
        resultats_grid["mean_train_accuracy"] - resultats_grid["mean_test_accuracy"]
    )
    resultats_grid["generalization_gap_f1_macro"] = (
        resultats_grid["mean_train_f1_macro"] - resultats_grid["mean_test_f1_macro"]
    )
    return grid, resultats_grid, temps_tuning


def construire_tuning_run_result(
    nested_tuned: dict[str, Any],
    grid: GridSearchCV,
    timing_policy: str,
    exploratory_tuning_seconds: float,
    secondary_holdout: dict[str, Any] | None = None,
    descriptive_holdout_all_variants: list[dict[str, Any]] | None = None,
) -> TuningRunResult:
    """Assemble le résultat confirmé d'un tuning à partir des artefacts bruts."""
    mesures_nested = nested_tuned["metrics"]
    return TuningRunResult(
        val_accuracy_mean=mesures_nested["val_accuracy_mean"],
        val_accuracy_std=mesures_nested["val_accuracy_std"],
        val_f1_macro_mean=mesures_nested["val_f1_macro_mean"],
        val_f1_macro_std=mesures_nested["val_f1_macro_std"],
        elapsed_seconds=mesures_nested["elapsed_seconds"],
        timing_policy=timing_policy,
        n_splits=mesures_nested["n_splits"],
        inner_splits=mesures_nested["inner_splits"],
        best_params_exploratory={
            k: normaliser_parametre_pour_json(v)
            for k, v in grid.best_params_.items()
        },
        best_params_per_fold=[
            {k: normaliser_parametre_pour_json(v) for k, v in row.items()}
            for row in nested_tuned["best_params_per_fold"].to_dict(orient="records")
        ],
        outer_fold_metrics=nested_tuned["outer_fold_metrics"].to_dict(orient="records"),
        exploratory_tuning_seconds=exploratory_tuning_seconds,
        fit_time_mean=mesures_nested["fit_time_mean"],
        score_time_mean=mesures_nested["score_time_mean"],
        secondary_holdout=secondary_holdout or {},
        descriptive_holdout_all_variants=descriptive_holdout_all_variants or [],
    )


def selectionner_meilleure_variante_untuned(
    spec: ModelStudySpec,
    untuned_results: list[UntunedVariantResult],
) -> UntunedVariantSpec:
    """Retourne la spécification de la meilleure variante non ajustée."""
    meilleure_variante = max(
        untuned_results,
        key=lambda item: (
            item.val_f1_macro_mean,
            item.val_accuracy_mean,
            -_ordre_simplicite_variante(item.variante),
        ),
    )
    return next(
        variante
        for variante in spec.untuned_variants
        if variante.name == meilleure_variante.variante
    )
