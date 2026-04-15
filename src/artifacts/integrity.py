"""Contrôles d'intégrité scientifique pour les artefacts canoniques."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ..config import RESULTS_DIR
from .io import charger_predictions
from .paths import model_paths

_METRIC_TOLERANCE = 1e-12


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def recompute_metrics_from_predictions(
    predictions_df: pd.DataFrame,
) -> dict[str, float]:
    """Recalcule les métriques du rapport depuis un tableau OOF standardisé."""
    y_true = predictions_df["y_true"].to_numpy()
    y_pred = predictions_df["y_pred"].to_numpy()
    return {
        "val_accuracy_mean": float(accuracy_score(y_true, y_pred)),
        "val_f1_macro_mean": float(f1_score(y_true, y_pred, average="macro")),
    }


def verifier_integrite_artefacts_modele(
    model_name: str,
    root: Path | None = None,
) -> dict[str, Any]:
    """Vérifie les artefacts canoniques ``tuned`` d'un modèle."""
    results_root = root or RESULTS_DIR
    paths = model_paths(model_name, root=results_root)

    canonical_metrics_path = paths.metrics_file("tuned")
    canonical_predictions_path = paths.predictions_file("tuned")

    status = "ok"
    issues: list[str] = []
    saved_accuracy = None
    saved_f1 = None
    recomputed_accuracy = None
    recomputed_f1 = None
    metrics_match_predictions = False

    if not canonical_metrics_path.exists():
        status = "missing_canonical_artifacts"
        issues.append("métriques canoniques tuned manquantes")
    if not canonical_predictions_path.exists():
        status = "missing_canonical_artifacts"
        issues.append("prédictions canoniques tuned manquantes")

    if canonical_metrics_path.exists() and canonical_predictions_path.exists():
        canonical_metrics = _load_json(canonical_metrics_path)
        canonical_predictions = charger_predictions(canonical_predictions_path)
        recomputed = recompute_metrics_from_predictions(canonical_predictions)
        saved_accuracy = float(canonical_metrics.get("val_accuracy_mean"))
        saved_f1 = float(canonical_metrics.get("val_f1_macro_mean"))
        recomputed_accuracy = recomputed["val_accuracy_mean"]
        recomputed_f1 = recomputed["val_f1_macro_mean"]
        metrics_match_predictions = (
            abs(saved_accuracy - recomputed_accuracy) <= _METRIC_TOLERANCE
            and abs(saved_f1 - recomputed_f1) <= _METRIC_TOLERANCE
        )
        if not metrics_match_predictions:
            status = "canonical_metric_mismatch"
            issues.append(
                "les métriques canoniques tuned ne correspondent pas aux métriques "
                "recalculées depuis les prédictions canoniques tuned"
            )

    return {
        "modele": model_name,
        "status": status,
        "canonical_metrics_exists": canonical_metrics_path.exists(),
        "canonical_predictions_exists": canonical_predictions_path.exists(),
        "canonical_metrics_path": str(canonical_metrics_path),
        "canonical_predictions_path": str(canonical_predictions_path),
        "val_accuracy_mean_saved": saved_accuracy,
        "val_f1_macro_mean_saved": saved_f1,
        "val_accuracy_mean_recomputed": recomputed_accuracy,
        "val_f1_macro_mean_recomputed": recomputed_f1,
        "metrics_match_predictions": metrics_match_predictions,
        "issues": issues,
    }


def canonical_artifact_completeness(
    modeles: list[str] | tuple[str, ...] | set[str],
    root: Path | None = None,
) -> pd.DataFrame:
    """Résume si chaque modèle dispose des preuves canoniques du classement."""
    results_root = root or RESULTS_DIR
    rows: list[dict[str, Any]] = []
    for model_name in modeles:
        paths = model_paths(model_name, root=results_root)
        metrics_exists = paths.metrics_file("tuned").exists()
        predictions_exists = paths.predictions_file("tuned").exists()
        rows.append(
            {
                "modele": model_name,
                "canonical_tuned_metrics": metrics_exists,
                "canonical_tuned_predictions": predictions_exists,
                "canonical_complete": metrics_exists and predictions_exists,
            }
        )
    return pd.DataFrame(rows).sort_values("modele").reset_index(drop=True)


def artifact_integrity_table(
    modeles: list[str] | tuple[str, ...] | set[str],
    root: Path | None = None,
) -> pd.DataFrame:
    """Construit un tableau synthèse d'intégrité des artefacts ``tuned``."""
    return pd.DataFrame(
        [
            verifier_integrite_artefacts_modele(model_name, root=root)
            for model_name in modeles
        ]
    ).sort_values("modele").reset_index(drop=True)
