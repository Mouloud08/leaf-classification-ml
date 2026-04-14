"""Modele SVM RBF."""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.svm import SVC

from ..config import PCA_EXPLAINED_VARIANCE, RANDOM_SEED
from .base import ModeleBase


class ModeleSVM(ModeleBase):
    nom = "svm"
    nom_affiche = "SVM RBF"
    famille = "noyau"
    description = "Modele a noyau non lineaire potentiellement tres performant sur ce dataset."
    hypothese = "Le SVM RBF devrait mieux separer les zones ou les classes se chevauchent."
    utiliser_pca_par_defaut = True
    necessite_scaling = True
    supports_probabilities = True

    supported_diagnostics = (
        "heatmap_C_gamma",
        "support_vectors",
        "learning_curve",
        "scaling_impact",
        "confusions_difficiles",
    )
    core_figures = (
        "heatmap_C_gamma",
        "support_vectors",
        "learning_curve",
    )
    advanced_figures = ("scaling_impact", "confusions_difficiles")

    def creer_estimateur(self):
        return SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)

    def grille_tuning(self) -> dict:
        return {
            "pca": [
                "passthrough",
                PCA(n_components=PCA_EXPLAINED_VARIANCE, random_state=RANDOM_SEED),
            ],
            "modele__C": [1.0, 10.0],
            "modele__gamma": ["scale", 0.01, 0.1],
        }
