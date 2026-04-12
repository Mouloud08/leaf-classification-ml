"""Runner principal — execute l'etude pour un ou plusieurs modeles."""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from ..artifacts import (
    StudyRunResult,
    TuningRunResult,
    UntunedVariantResult,
    charger_predictions,
    model_paths,
    sauvegarder_mesures,
    sauvegarder_predictions,
    sauvegarder_probabilites_oof,
    sauvegarder_resultats_grid_search,
)
from ..config import TIMING_POLICY
from ..data import charger_donnees
from ..evaluation import evaluer_modele_cv, evaluer_modele_tuned_nested_cv
from .model_specs import obtenir_model_study_spec
from .shared import (
    ajouter_gaps,
    construire_estimateur_variante,
    construire_tuning_run_result,
    executer_grid_search_exploratoire,
    preparer_tuning_modele,
)
from .study_spec import ModelStudySpec, StudySpec, UntunedVariantSpec

logger = logging.getLogger(__name__)


def _charger_json(chemin) -> dict[str, Any]:
    return json.loads(chemin.read_text(encoding="utf-8"))


def _charger_resultat_untuned(paths, variante: UntunedVariantSpec) -> UntunedVariantResult:
    contenu = _charger_json(paths.metrics_file(variante.name))
    return UntunedVariantResult(
        variante=contenu["variante"],
        selection_protocol=contenu["selection_protocol"],
        evaluation_stage=contenu["evaluation_stage"],
        val_accuracy_mean=contenu["val_accuracy_mean"],
        val_accuracy_std=contenu["val_accuracy_std"],
        val_f1_macro_mean=contenu["val_f1_macro_mean"],
        val_f1_macro_std=contenu["val_f1_macro_std"],
        train_accuracy_mean=contenu["train_accuracy_mean"],
        train_f1_macro_mean=contenu["train_f1_macro_mean"],
        generalization_gap_accuracy=contenu["generalization_gap_accuracy"],
        generalization_gap_f1_macro=contenu["generalization_gap_f1_macro"],
        elapsed_seconds=contenu["elapsed_seconds"],
        timing_policy=contenu["timing_policy"],
        n_splits=contenu["n_splits"],
        fit_time_mean=contenu.get("fit_time_mean", 0.0),
        score_time_mean=contenu.get("score_time_mean", 0.0),
    )


def _charger_resultat_tuning(paths) -> TuningRunResult | None:
    chemin = paths.metrics_file("tuned")
    if not chemin.exists():
        return None

    contenu = _charger_json(chemin)
    return TuningRunResult(
        val_accuracy_mean=contenu["val_accuracy_mean"],
        val_accuracy_std=contenu["val_accuracy_std"],
        val_f1_macro_mean=contenu["val_f1_macro_mean"],
        val_f1_macro_std=contenu["val_f1_macro_std"],
        elapsed_seconds=contenu["elapsed_seconds"],
        timing_policy=contenu["timing_policy"],
        n_splits=contenu["n_splits"],
        inner_splits=contenu["inner_splits"],
        selection_protocol=contenu["selection_protocol"],
        evaluation_stage=contenu["evaluation_stage"],
        best_params_exploratory=contenu.get("best_params_exploratory", {}),
        best_params_per_fold=contenu.get("best_params_per_fold", []),
        exploratory_tuning_seconds=contenu.get("exploratory_tuning_seconds", 0.0),
        fit_time_mean=contenu.get("fit_time_mean", 0.0),
        score_time_mean=contenu.get("score_time_mean", 0.0),
    )


def _peut_reutiliser_artefacts(
    spec: ModelStudySpec,
    study: StudySpec,
) -> bool:
    if study.force or not study.skip_existing:
        return False

    paths = model_paths(spec.model_name, root=study.output_root)
    fichiers_untuned = [
        paths.metrics_file(variante.name) for variante in spec.untuned_variants
    ]
    if not all(chemin.exists() for chemin in fichiers_untuned):
        return False

    if study.no_tuning:
        return True

    tuned_metrics = paths.metrics_file("tuned")
    tuned_predictions = paths.predictions_file("tuned")
    return tuned_metrics.exists() and tuned_predictions.exists()


def _charger_resultats_existants(
    spec: ModelStudySpec,
    study: StudySpec,
) -> StudyRunResult:
    paths = model_paths(spec.model_name, root=study.output_root)
    untuned_results = [
        _charger_resultat_untuned(paths, variante)
        for variante in spec.untuned_variants
    ]
    tuning_result = None if study.no_tuning else _charger_resultat_tuning(paths)
    return StudyRunResult(
        model_name=spec.model_name,
        untuned_results=untuned_results,
        tuning_result=tuning_result,
    )

def charger_predictions_tuned_si_disponibles(
    model_name: str,
    study: StudySpec,
) -> pd.DataFrame | None:
    """Charge les predictions tuned si elles existent deja sur disque."""
    paths = model_paths(model_name, root=study.output_root)
    chemin = paths.predictions_file("tuned")
    if not chemin.exists():
        return None
    return charger_predictions(chemin)


def executer_variantes_untuned(
    spec: ModelStudySpec,
    X: pd.DataFrame,
    y: Any,
    study: StudySpec,
) -> list[UntunedVariantResult]:
    """Evalue toutes les variantes non tunees d'un modele."""
    paths = model_paths(spec.model_name, root=study.output_root)
    resultats: list[UntunedVariantResult] = []

    for variante in spec.untuned_variants:
        logger.info("  Variante: %s", variante.name)
        estimateur = construire_estimateur_variante(spec.model_name, variante)
        mesures = ajouter_gaps(evaluer_modele_cv(estimateur, X, y))
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
            timing_policy=TIMING_POLICY,
            n_splits=mesures["n_splits"],
            fit_time_mean=mesures["fit_time_mean"],
            score_time_mean=mesures["score_time_mean"],
        )

        sauvegarder_mesures(
            spec.model_name,
            variante.name,
            result.to_dict(),
            paths=paths,
        )
        resultats.append(result)

    return resultats


def executer_tuning(
    spec: ModelStudySpec,
    X: pd.DataFrame,
    y: Any,
    label_encoder: Any,
    study: StudySpec,
) -> TuningRunResult:
    """Execute le tuning exploratoire et la nested CV confirmatoire."""
    paths = model_paths(spec.model_name, root=study.output_root)

    logger.info("  Tuning exploratoire...")
    grid, resultats_grid, temps_tuning = executer_grid_search_exploratoire(
        spec=spec,
        X=X,
        y=y,
        n_splits=study.n_splits,
        random_seed=study.random_seed,
    )
    sauvegarder_resultats_grid_search(spec.model_name, resultats_grid, paths=paths)

    logger.info("  Nested CV confirmatoire...")
    estimateur_grid, grille_tuning = preparer_tuning_modele(spec)
    nested_tuned = evaluer_modele_tuned_nested_cv(
        estimateur_ou_pipeline=estimateur_grid,
        param_grid=grille_tuning,
        X=X,
        y=y,
        label_encoder=label_encoder,
        outer_splits=study.n_splits,
        inner_splits=study.n_splits,
        random_state=study.random_seed,
        return_probabilities=spec.supports_probabilities,
    )

    tuning_result = construire_tuning_run_result(
        nested_tuned=nested_tuned,
        grid=grid,
        timing_policy=TIMING_POLICY,
        exploratory_tuning_seconds=temps_tuning,
    )

    sauvegarder_mesures(
        spec.model_name,
        "tuned",
        tuning_result.to_dict(),
        paths=paths,
    )

    sauvegarder_predictions(
        spec.model_name,
        "tuned",
        nested_tuned["oof_predictions"],
        paths=paths,
    )

    if nested_tuned.get("oof_probabilities") is not None:
        sauvegarder_probabilites_oof(
            spec.model_name,
            "tuned",
            nested_tuned["oof_probabilities"],
            paths=paths,
        )

    return tuning_result


def executer_modele(
    model_name: str,
    study: StudySpec,
) -> StudyRunResult:
    """Execute l'etude complete pour un modele."""
    spec = obtenir_model_study_spec(model_name)
    logger.info("=== %s ===", spec.title)

    if _peut_reutiliser_artefacts(spec, study):
        logger.info("Reutilisation des artefacts existants pour %s", spec.model_name)
        return _charger_resultats_existants(spec, study)

    X, y, label_encoder = charger_donnees()

    # Variantes non tunees
    logger.info("Variantes non tunees...")
    untuned_results = executer_variantes_untuned(spec, X, y, study)

    # Tuning
    tuning_result = None
    if not study.no_tuning:
        logger.info("Tuning...")
        tuning_result = executer_tuning(spec, X, y, label_encoder, study)

    return StudyRunResult(
        model_name=model_name,
        untuned_results=untuned_results,
        tuning_result=tuning_result,
    )


def executer_etude(study: StudySpec) -> list[StudyRunResult]:
    """Execute l'etude complete pour tous les modeles specifies."""
    resultats: list[StudyRunResult] = []
    for model_name in study.models:
        try:
            result = executer_modele(model_name, study)
            resultats.append(result)
        except Exception as e:
            logger.error("Erreur pour %s: %s", model_name, e)
            resultats.append(
                StudyRunResult(
                    model_name=model_name,
                    success=False,
                    error=str(e),
                )
            )
    return resultats
