"""Lecture et ecriture des artefacts tabulaires et figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..config import RESULTS_DIR
from .paths import GlobalArtifactPaths, ModelArtifactPaths, model_paths


def _resolve_model_paths(
    nom_modele: str,
    paths: ModelArtifactPaths | None = None,
) -> ModelArtifactPaths:
    if paths is not None:
        paths.ensure_dirs()
        return paths

    chemins_modele = model_paths(nom_modele, root=RESULTS_DIR)
    chemins_modele.ensure_dirs()
    return chemins_modele


def sauvegarder_predictions(
    nom_modele: str,
    variante: str,
    tableau: pd.DataFrame,
    paths: ModelArtifactPaths | None = None,
) -> Path:
    """Sauvegarde les predictions detaillees."""
    chemins_modele = _resolve_model_paths(nom_modele, paths=paths)
    chemin = chemins_modele.predictions_file(variante)
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
    chemins_modele = _resolve_model_paths(nom_modele, paths=paths)
    chemin = chemins_modele.probabilities_file(variante)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tableau.to_csv(chemin, index=False, encoding="utf-8")
    return chemin


def charger_predictions(chemin: Path) -> pd.DataFrame:
    """Charge un fichier de predictions."""
    return pd.read_csv(chemin, encoding="utf-8")


def charger_probabilites(chemin: Path) -> pd.DataFrame:
    """Charge un fichier de probabilites."""
    return pd.read_csv(chemin, encoding="utf-8")


def charger_predictions_modele(
    nom_modele: str,
    variante: str = "tuned",
    root: Path | None = None,
) -> pd.DataFrame:
    """Charge les predictions d'un modele a partir du layout canonique."""
    chemins_modele = model_paths(nom_modele, root=root or RESULTS_DIR)
    return charger_predictions(chemins_modele.predictions_file(variante))


def charger_probabilites_modele(
    nom_modele: str,
    variante: str = "tuned",
    root: Path | None = None,
) -> pd.DataFrame:
    """Charge les probabilites OOF d'un modele a partir du layout canonique."""
    chemins_modele = model_paths(nom_modele, root=root or RESULTS_DIR)
    return charger_probabilites(chemins_modele.probabilities_file(variante))


def sauvegarder_resultats_grid_search(
    nom_modele: str,
    tableau: pd.DataFrame,
    paths: ModelArtifactPaths | None = None,
) -> Path:
    """Sauvegarde le tableau brut d'un GridSearchCV."""
    chemins_modele = _resolve_model_paths(nom_modele, paths=paths)
    chemin = chemins_modele.grid_search_file()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tableau.to_csv(chemin, index=False, encoding="utf-8")
    return chemin


def charger_resultats_grid_search(chemin: Path) -> pd.DataFrame:
    """Charge un fichier de résultats de recherche sur grille."""
    return pd.read_csv(chemin, encoding="utf-8")


def sauvegarder_figure(
    fig: plt.Figure,
    name: str,
    paths: ModelArtifactPaths,
    category: str = "core",
    dpi: int = 150,
) -> Path:
    """Sauvegarde une figure matplotlib dans le dossier du modele."""
    paths.ensure_dirs()
    chemin = paths.figure_file(name, category=category)
    fig.savefig(chemin, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return chemin


def sauvegarder_figure_globale(
    fig: plt.Figure,
    name: str,
    paths: GlobalArtifactPaths,
    dpi: int = 150,
) -> Path:
    """Sauvegarde une figure dans le dossier global."""
    paths.ensure_dirs()
    chemin = paths.figures_dir / f"{name}.png"
    fig.savefig(chemin, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return chemin
