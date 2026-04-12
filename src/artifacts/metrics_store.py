"""Lecture et ecriture des artefacts de metriques."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import ModelArtifactPaths, model_paths
from .validation import (
    LEGACY_KEYS,
    MetricsArtifactValidationError,
    normaliser_mesures_pour_schema,
    valider_artefacts_mesures,
)


def _slugifier(valeur: str) -> str:
    return valeur.lower().replace(" ", "_")


def _ecrire_json(chemin: Path, contenu: dict[str, Any]) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(
        json.dumps(contenu, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return chemin


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
        # Fallback vers l'ancien layout plat
        from ..config import METRICS_DIR
        chemin = METRICS_DIR / f"{_slugifier(nom_modele)}__{_slugifier(variante)}.json"
    return _ecrire_json(chemin, contenu)


def charger_mesures(
    validation: str = "none",
    exclure_legacy: bool = False,
    modeles: list[str] | tuple[str, ...] | set[str] | None = None,
    metrics_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Charge tous les JSON de metriques actuellement disponibles."""
    if metrics_dir is None:
        from ..config import METRICS_DIR
        metrics_dir = METRICS_DIR

    enregistrements: list[dict[str, Any]] = []
    sources: list[str] = []
    modeles_normalises = None
    if modeles is not None:
        modeles_normalises = {str(m).lower() for m in modeles}
    if not metrics_dir.exists():
        return enregistrements

    for chemin in sorted(metrics_dir.glob("*.json")):
        enregistrement = json.loads(chemin.read_text(encoding="utf-8"))
        if modeles_normalises is not None:
            modele = str(enregistrement.get("modele", "")).lower()
            if modele not in modeles_normalises:
                continue
        enregistrements.append(enregistrement)
        sources.append(str(chemin))

    if exclure_legacy:
        enregistrements_filtres: list[dict[str, Any]] = []
        sources_filtrees: list[str] = []
        for enregistrement, source in zip(enregistrements, sources, strict=False):
            cle = (
                f"{str(enregistrement.get('modele', '')).lower()}"
                f"__{str(enregistrement.get('variante', '')).lower()}"
            )
            if cle in LEGACY_KEYS:
                continue
            enregistrements_filtres.append(enregistrement)
            sources_filtrees.append(source)
        enregistrements = enregistrements_filtres
        sources = sources_filtrees

    erreurs = valider_artefacts_mesures(enregistrements, sources=sources)
    if validation == "strict" and erreurs:
        raise MetricsArtifactValidationError("\n".join(erreurs))
    if validation == "warn" and erreurs:
        warnings.warn("\n".join(erreurs), stacklevel=2)

    return enregistrements


def charger_tableau_mesures(
    validation: str = "none",
    exclure_legacy: bool = False,
    modeles: list[str] | tuple[str, ...] | set[str] | None = None,
    metrics_dir: Path | None = None,
) -> pd.DataFrame:
    """Retourne les metriques sous forme tabulaire."""
    return pd.DataFrame(
        charger_mesures(
            validation=validation,
            exclure_legacy=exclure_legacy,
            modeles=modeles,
            metrics_dir=metrics_dir,
        )
    )


def charger_tableau_mesures_valides(
    exclure_legacy: bool = True,
    modeles: list[str] | tuple[str, ...] | set[str] | None = None,
    metrics_dir: Path | None = None,
) -> pd.DataFrame:
    """Charge uniquement des artefacts valides pour la synthese finale."""
    return charger_tableau_mesures(
        validation="strict",
        exclure_legacy=exclure_legacy,
        modeles=modeles,
        metrics_dir=metrics_dir,
    )
