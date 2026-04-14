"""Modele Foret Aleatoire."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from ..config import RANDOM_SEED
from .base import ModeleBase


class ModeleForetAleatoire(ModeleBase):
    nom = "foret_aleatoire"
    nom_affiche = "Foret aleatoire"
    famille = "ensemble"
    description = "Ensemble d'arbres capable de capturer des interactions non lineaires."
    hypothese = "La foret aleatoire devrait etre competitive sans aucun scaling."
    utiliser_pca_par_defaut = False
    necessite_scaling = False
    supports_probabilities = True

    supported_diagnostics = (
        "oob_vs_n_estimators",
        "importance_dispersion",
        "validation_curve_min_samples",
        "score_vs_temps",
    )
    core_figures = (
        "oob_vs_n_estimators",
        "importance_dispersion",
    )
    advanced_figures = ("validation_curve_min_samples", "score_vs_temps")

    def creer_estimateur(self):
        return RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )

    def grille_tuning(self) -> dict:
        return {
            "modele__n_estimators": [200, 400],
            "modele__max_depth": [None, 20],
            "modele__min_samples_leaf": [1, 2],
            "modele__max_features": ["sqrt"],
        }
