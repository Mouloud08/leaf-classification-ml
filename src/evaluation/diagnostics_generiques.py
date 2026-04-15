"""Diagnostics génériques produits pour tous les modèles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metriques import (
    calculer_metriques_par_classe,
    construire_matrice_confusion,
    extraire_top_confusions,
)


def tableau_variantes(
    resultats_untuned: list[dict[str, Any]],
    mesures_tuned: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Construit le tableau récapitulatif des variantes."""
    colonnes = [
        "variante",
        "selection_protocol",
        "evaluation_stage",
        "val_accuracy_mean",
        "val_accuracy_std",
        "val_f1_macro_mean",
        "val_f1_macro_std",
        "train_accuracy_mean",
        "train_f1_macro_mean",
        "generalization_gap_accuracy",
        "generalization_gap_f1_macro",
        "elapsed_seconds",
        "timing_policy",
    ]
    lignes = list(resultats_untuned)
    if mesures_tuned is not None:
        lignes.append({"variante": "tuned", **mesures_tuned})
    tableau = pd.DataFrame(lignes)
    tableau = tableau.reindex(columns=colonnes)
    return tableau.sort_values("val_f1_macro_mean", ascending=False).reset_index(
        drop=True
    )


def tableau_gaps(resultats_untuned: list[dict[str, Any]]) -> pd.DataFrame:
    """Retourne le tableau des écarts de généralisation des variantes non tunées."""
    return pd.DataFrame(
        [
            {
                "variante": ligne["variante"],
                "generalization_gap_accuracy": ligne["generalization_gap_accuracy"],
                "generalization_gap_f1_macro": ligne["generalization_gap_f1_macro"],
            }
            for ligne in resultats_untuned
        ]
    ).round(4)


def tableau_plis_externes(nested_result: dict[str, Any]) -> pd.DataFrame:
    """Retourne les métriques par pli externe de la nested CV."""
    return nested_result["outer_fold_metrics"].copy()


def top_confusions(
    y_vrai: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: Any | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Extrait les paires de classes les plus confondues."""
    return extraire_top_confusions(
        y_vrai,
        y_pred,
        label_encoder=label_encoder,
        top_n=top_n,
    )


def classes_difficiles(
    y_vrai: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: Any | None = None,
    top_n: int = 15,
    metrique: str = "recall",
) -> pd.DataFrame:
    """Retourne les classes avec le plus faible score sur une métrique."""
    tableau = calculer_metriques_par_classe(y_vrai, y_pred, label_encoder=label_encoder)
    return (
        tableau.sort_values(metrique, ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )


def matrice_confusion_normalisee(
    y_vrai: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: Any | None = None,
) -> pd.DataFrame:
    """Construit la matrice de confusion normalisée."""
    return construire_matrice_confusion(
        y_vrai, y_pred, label_encoder=label_encoder, normalize="true"
    )


def comparaison_baseline_best_tuned(
    resultats_untuned: list[dict[str, Any]],
    mesures_tuned: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compare la référence simple, la meilleure variante non tunée et `tuned`."""
    lignes = []

    baseline = None
    best_untuned = None
    for r in resultats_untuned:
        stage = r.get("evaluation_stage", "")
        if stage == "baseline":
            baseline = r
        if best_untuned is None or r.get("val_f1_macro_mean", 0) > best_untuned.get(
            "val_f1_macro_mean", 0
        ):
            best_untuned = r

    if baseline is not None:
        lignes.append({"etiquette": "baseline", **baseline})
    if best_untuned is not None and best_untuned is not baseline:
        lignes.append({"etiquette": "best_untuned", **best_untuned})
    if mesures_tuned is not None:
        lignes.append({"etiquette": "tuned", "variante": "tuned", **mesures_tuned})

    colonnes = [
        "etiquette",
        "variante",
        "val_f1_macro_mean",
        "val_f1_macro_std",
        "val_accuracy_mean",
        "elapsed_seconds",
    ]
    df = pd.DataFrame(lignes)
    return df.reindex(columns=[c for c in colonnes if c in df.columns])
