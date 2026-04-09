"""Pont de compatibilite pour les imports `src.models`."""

from .modeles import (
    MODEL_REGISTRY,
    creer_modele_simple,
    initialiser_grille_tuning,
    noms_modeles,
    obtenir_definition_modele,
)

__all__ = [
    "MODEL_REGISTRY",
    "creer_modele_simple",
    "initialiser_grille_tuning",
    "noms_modeles",
    "obtenir_definition_modele",
]

