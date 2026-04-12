"""Lecture et ecriture des resultats de tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import ModelArtifactPaths


def _slugifier(valeur: str) -> str:
    return valeur.lower().replace(" ", "_")


def _ecrire_json(chemin: Path, contenu: dict[str, Any]) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(
        json.dumps(contenu, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return chemin


def sauvegarder_resultats_grid_search(
    nom_modele: str,
    tableau: pd.DataFrame,
    paths: ModelArtifactPaths | None = None,
) -> Path:
    """Sauvegarde le tableau brut d'un GridSearchCV."""
    if paths is not None:
        paths.ensure_dirs()
        chemin = paths.grid_search_file()
    else:
        from ..config import GRID_SEARCH_DIR
        chemin = GRID_SEARCH_DIR / f"{_slugifier(nom_modele)}__grid_search.csv"
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tableau.to_csv(chemin, index=False, encoding="utf-8")
    return chemin


def charger_resultats_grid_search(chemin: Path) -> pd.DataFrame:
    """Charge un fichier de resultats de grid search."""
    return pd.read_csv(chemin, encoding="utf-8")
