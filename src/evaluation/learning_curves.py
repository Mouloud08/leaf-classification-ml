"""Ré-export des courbes d'apprentissage depuis les diagnostics spécifiques."""

from .diagnostics_specifiques import (
    calculer_courbe_apprentissage,
    compute_learning_curve,
)

__all__ = ["calculer_courbe_apprentissage", "compute_learning_curve"]
