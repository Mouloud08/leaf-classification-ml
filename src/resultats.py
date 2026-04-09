"""Lecture et ecriture des artefacts de resultats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import GRID_SEARCH_DIR, METRICS_DIR


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
) -> Path:
    """Sauvegarde les mesures d'une variante dans `results/metrics`."""
    nom_fichier = f"{_slugifier(nom_modele)}__{_slugifier(variante)}.json"
    contenu = {"modele": nom_modele, "variante": variante, **mesures}
    return _ecrire_json(METRICS_DIR / nom_fichier, contenu)


def sauvegarder_placeholder_tuning(
    nom_modele: str,
    contenu: dict[str, Any],
) -> Path:
    """Sauvegarde la section tuning boilerplate d'un notebook."""
    nom_fichier = f"{_slugifier(nom_modele)}__tuning_placeholder.json"
    return _ecrire_json(GRID_SEARCH_DIR / nom_fichier, contenu)


def charger_mesures() -> list[dict[str, Any]]:
    """Charge tous les JSON de mesures actuellement disponibles."""
    enregistrements: list[dict[str, Any]] = []
    if not METRICS_DIR.exists():
        return enregistrements

    for chemin in sorted(METRICS_DIR.glob("*.json")):
        enregistrements.append(json.loads(chemin.read_text(encoding="utf-8")))
    return enregistrements


def charger_tableau_mesures() -> pd.DataFrame:
    """Retourne les mesures sous forme tabulaire."""
    return pd.DataFrame(charger_mesures())

