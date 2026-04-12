"""Sauvegarde des figures dans l'arborescence par modele."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .paths import GlobalArtifactPaths, ModelArtifactPaths


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
