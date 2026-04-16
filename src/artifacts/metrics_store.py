"""Lecture et ecriture des artefacts de metriques."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import RESULTS_DIR
from .paths import ModelArtifactPaths, model_paths
from .validation import (
    MetricsArtifactValidationError,
    normaliser_mesures_pour_schema,
    valider_artefacts_mesures,
)


def _ecrire_json(chemin: Path, contenu: dict[str, Any]) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(
        json.dumps(contenu, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return chemin


def _iter_metric_files(metrics_dir: Path) -> list[Path]:
    """Retourne les fichiers JSON de métriques.

    Les fichiers peuvent provenir des dossiers canoniques ou explicites à plat.
    """
    if not metrics_dir.exists():
        return []

    if metrics_dir.name == "metrics":
        return sorted(metrics_dir.glob("*.json"))

    canonical_files = sorted(metrics_dir.glob("*/metrics/*.json"))
    if canonical_files:
        return canonical_files

    return sorted(metrics_dir.glob("*.json"))


def _validate_with_policy(
    enregistrements: list[dict[str, Any]],
    sources: list[str],
    validation: str,
) -> None:
    erreurs = valider_artefacts_mesures(enregistrements, sources=sources)
    if validation == "strict" and erreurs:
        raise MetricsArtifactValidationError("\n".join(erreurs))
    if validation == "warn" and erreurs:
        warnings.warn("\n".join(erreurs), stacklevel=2)


def sauvegarder_mesures(
    nom_modele: str,
    variante: str,
    mesures: dict[str, Any],
    paths: ModelArtifactPaths | None = None,
) -> Path:
    """Sauvegarde les metriques d'une variante."""
    contenu = normaliser_mesures_pour_schema(nom_modele, variante, mesures)
    if paths is not None:
        paths.ensure_dirs()
        chemin = paths.metrics_file(variante)
    else:
        # Sans chemins explicites, on sauvegarde toujours dans le layout canonique.
        chemins_modele = model_paths(nom_modele, root=RESULTS_DIR)
        chemins_modele.ensure_dirs()
        chemin = chemins_modele.metrics_file(variante)
    return _ecrire_json(chemin, contenu)


def charger_mesures(
    validation: str = "none",
    modeles: list[str] | tuple[str, ...] | set[str] | None = None,
    metrics_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Charge les metriques depuis le layout canonique par defaut.

    Si ``metrics_dir`` pointe explicitement vers un dossier ``metrics`` plat, la
    fonction charge ce dossier tel quel, ce qui reste utile pour certains tests
    et pour certains tests unitaires.
    """
    if metrics_dir is None:
        metrics_dir = RESULTS_DIR

    enregistrements: list[dict[str, Any]] = []
    sources: list[str] = []
    modeles_normalises = None
    if modeles is not None:
        modeles_normalises = {str(m).lower() for m in modeles}
    fichiers = _iter_metric_files(metrics_dir)
    if not fichiers:
        return enregistrements

    for chemin in fichiers:
        enregistrement = json.loads(chemin.read_text(encoding="utf-8"))
        if modeles_normalises is not None:
            modele = str(enregistrement.get("modele", "")).lower()
            if modele not in modeles_normalises:
                continue
        enregistrements.append(enregistrement)
        sources.append(str(chemin))

    _validate_with_policy(
        enregistrements,
        sources=sources,
        validation=validation,
    )

    return enregistrements


def charger_tableau_mesures(
    validation: str = "none",
    modeles: list[str] | tuple[str, ...] | set[str] | None = None,
    metrics_dir: Path | None = None,
) -> pd.DataFrame:
    """Retourne les metriques sous forme tabulaire."""
    return pd.DataFrame(
        charger_mesures(
            validation=validation,
            modeles=modeles,
            metrics_dir=metrics_dir,
        )
    )


def charger_tableau_mesures_valides(
    modeles: list[str] | tuple[str, ...] | set[str] | None = None,
    metrics_dir: Path | None = None,
) -> pd.DataFrame:
    """Charge uniquement des artefacts valides pour la synthese finale."""
    return charger_tableau_mesures(
        validation="strict",
        modeles=modeles,
        metrics_dir=metrics_dir,
    )
