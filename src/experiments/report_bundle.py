"""Génération du rapport global et du manifeste de l'étude."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..artifacts import (
    StudyRunResult,
    canonical_artifact_completeness,
    global_paths,
)
from ..artifacts.schema import METRICS_SCHEMA_VERSION
from .study_spec import StudySpec


def generer_manifeste(
    study: StudySpec,
    resultats: list[StudyRunResult],
) -> dict[str, Any]:
    """Génère le manifeste JSON de l'étude."""
    paths = global_paths(root=study.output_root)
    paths.ensure_dirs()

    manifeste = {
        "date": datetime.now(timezone.utc).isoformat(),
        "schema_version": METRICS_SCHEMA_VERSION,
        "config": {
            "n_splits": study.n_splits,
            "random_seed": study.random_seed,
            "primary_metric": study.primary_metric,
            "figures_mode": study.figures_mode,
            "no_tuning": study.no_tuning,
        },
        "models": {},
    }

    for result in resultats:
        model_entry: dict[str, Any] = {
            "success": result.success,
        }
        if result.error:
            model_entry["error"] = result.error
        if result.success:
            model_entry["n_untuned_variants"] = len(result.untuned_results)
            model_entry["tuned"] = result.tuning_result is not None
            if result.tuning_result is not None:
                model_entry["tuned_f1_macro"] = result.tuning_result.val_f1_macro_mean
            if result.figures is not None:
                model_entry["figures_core"] = result.figures.core_generated
                model_entry["figures_advanced"] = result.figures.advanced_generated
                model_entry["figures_errors"] = result.figures.errors
        manifeste["models"][result.model_name] = model_entry

    chemin = paths.manifest_file
    chemin.write_text(
        json.dumps(manifeste, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifeste


def generer_comparaison_globale(
    resultats: list[StudyRunResult],
    study: StudySpec,
) -> pd.DataFrame:
    """Génère le tableau de comparaison globale et le sauvegarde."""
    paths = global_paths(root=study.output_root)
    paths.ensure_dirs()

    lignes = []
    for result in resultats:
        if not result.success:
            continue
        # Meilleure variante non ajustée
        best_untuned = None
        for r in result.untuned_results:
            if (
                best_untuned is None
                or r.val_f1_macro_mean > best_untuned.val_f1_macro_mean
            ):
                best_untuned = r

        if best_untuned is not None:
            lignes.append(
                {
                    "modele": result.model_name,
                    "variante": (
                        f"meilleure_variante_non_ajustee ({best_untuned.variante})"
                    ),
                    "val_f1_macro_mean": best_untuned.val_f1_macro_mean,
                    "val_f1_macro_std": best_untuned.val_f1_macro_std,
                    "val_accuracy_mean": best_untuned.val_accuracy_mean,
                    "elapsed_seconds": best_untuned.elapsed_seconds,
                    "evaluation_stage": best_untuned.evaluation_stage,
                }
            )

        if result.tuning_result is not None:
            lignes.append(
                {
                    "modele": result.model_name,
                    "variante": "ajustee",
                    "val_f1_macro_mean": result.tuning_result.val_f1_macro_mean,
                    "val_f1_macro_std": result.tuning_result.val_f1_macro_std,
                    "val_accuracy_mean": result.tuning_result.val_accuracy_mean,
                    "elapsed_seconds": result.tuning_result.elapsed_seconds,
                    "evaluation_stage": "tuned",
                }
            )

    df = pd.DataFrame(lignes)
    if not df.empty:
        df = df.sort_values("val_f1_macro_mean", ascending=False).reset_index(drop=True)
        df.to_csv(paths.metrics_summary_file, index=False, encoding="utf-8")

        completeness = canonical_artifact_completeness(
            study.models,
            root=study.output_root,
        )
        if bool(completeness["canonical_complete"].all()):
            ranked_source = df[df["evaluation_stage"] == "tuned"].copy()
            if ranked_source.empty:
                ranked_source = df.copy()
            ranked = (
                ranked_source.loc[:, ["modele", "val_f1_macro_mean"]]
                .rename(columns={"val_f1_macro_mean": "best_f1_macro"})
                .sort_values("best_f1_macro", ascending=False)
                .reset_index(drop=True)
            )
            ranked["rank"] = range(1, len(ranked) + 1)
            ranked.to_csv(paths.ranked_models_file, index=False, encoding="utf-8")

    return df
