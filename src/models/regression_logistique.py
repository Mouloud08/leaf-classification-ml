"""Modele Regression Logistique."""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from ..config import PCA_COMPONENTS, RANDOM_SEED
from .base import ModeleBase


class ModeleRegressionLogistique(ModeleBase):
    nom = "regression_logistique"
    nom_affiche = "Regression logistique"
    famille = "lineaire"
    description = "Modele lineaire probabiliste robuste pour la multiclassification."
    hypothese = "La regularisation et le scaling devraient nettement ameliorer le score."
    utiliser_pca_par_defaut = True
    necessite_scaling = True
    supports_probabilities = True

    supported_diagnostics = (
        "regularization_path",
        "coefficients",
        "calibration",
        "pr_curve_micro",
        "validation_curve_C",
    )
    core_figures = (
        "regularization_path",
        "coefficients",
        "calibration",
        "pr_curve_micro",
    )
    advanced_figures = ("validation_curve_C",)

    def creer_estimateur(self):
        return LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=2500,
            solver="lbfgs",
        )

    def grille_tuning(self) -> dict:
        return {
            "pca": [
                "passthrough",
                PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED),
            ],
            "modele__C": [0.1, 1.0, 10.0, 100.0],
        }
