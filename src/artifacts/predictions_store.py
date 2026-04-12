"""Lecture et ecriture des predictions et probabilites."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import ModelArtifactPaths


def _slugifier(valeur: str) -> str:
    return valeur.lower().replace(" ", "_")


def sauvegarder_predictions(
    nom_modele: str,
    variante: str,
    tableau: pd.DataFrame,
    paths: ModelArtifactPaths | None = None,
) -> Path:
    """Sauvegarde les predictions detaillees."""
    if paths is not None:
        paths.ensure_dirs()
        chemin = paths.predictions_file(variante)
    else:
        from ..config import PREDICTIONS_DIR
        chemin = (
            PREDICTIONS_DIR
            / f"{_slugifier(nom_modele)}__{_slugifier(variante)}__predictions.csv"
        )
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tableau.to_csv(chemin, index=False, encoding="utf-8")
    return chemin


def sauvegarder_probabilites_oof(
    nom_modele: str,
    variante: str,
    tableau: pd.DataFrame,
    paths: ModelArtifactPaths | None = None,
) -> Path:
    """Sauvegarde les probabilites out-of-fold."""
    if paths is not None:
        paths.ensure_dirs()
        chemin = paths.probabilities_file(variante)
    else:
        from ..config import PREDICTIONS_DIR
        chemin = (
            PREDICTIONS_DIR
            / f"{_slugifier(nom_modele)}__{_slugifier(variante)}__probabilities.csv"
        )
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tableau.to_csv(chemin, index=False, encoding="utf-8")
    return chemin


def charger_predictions(chemin: Path) -> pd.DataFrame:
    """Charge un fichier de predictions."""
    return pd.read_csv(chemin, encoding="utf-8")


def charger_probabilites(chemin: Path) -> pd.DataFrame:
    """Charge un fichier de probabilites."""
    return pd.read_csv(chemin, encoding="utf-8")
