"""Modele Perceptron."""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron

from ..config import PCA_EXPLAINED_VARIANCE, RANDOM_SEED
from .base import ModeleBase


class ModelePerceptron(ModeleBase):
    nom = "perceptron"
    nom_affiche = "Perceptron"
    famille = "lineaire"
    description = "Modele lineaire simple servant de baseline rapide."
    hypothese = "Le Perceptron donne une baseline correcte mais limitera la separabilite non lineaire."
    utiliser_pca_par_defaut = True
    necessite_scaling = True
    supports_probabilities = False

    supported_diagnostics = (
        "convergence",
        "decision_function_histogram",
        "coefficients",
        "confusion_non_separables",
    )
    core_figures = (
        "convergence",
        "decision_function_histogram",
        "coefficients",
        "confusion_non_separables",
    )
    advanced_figures = ()

    def creer_estimateur(self):
        return Perceptron(random_state=RANDOM_SEED, max_iter=2000, tol=1e-3)

    def grille_tuning(self) -> dict:
        return {
            "pca": [
                "passthrough",
                PCA(n_components=PCA_EXPLAINED_VARIANCE, random_state=RANDOM_SEED),
            ],
            "modele__alpha": [0.0001, 0.001],
            "modele__penalty": [None, "l2", "l1"],
        }
