"""Modele Arbre de Decision."""

from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

from ..config import RANDOM_SEED
from .base import ModeleBase


class ModeleArbreDecision(ModeleBase):
    nom = "arbre_decision"
    nom_affiche = "Arbre de decision"
    famille = "arbre"
    description = (
        "Modele non lineaire interpretable, fonde sur des coupures "
        "successives de l'espace des variables."
    )
    hypothese = (
        "Un arbre unique devrait mieux capturer la non-linearite que les "
        "baselines lineaires, mais restera plus instable qu'une foret aleatoire."
    )
    utiliser_pca_par_defaut = False
    necessite_scaling = False
    supports_probabilities = False

    supported_diagnostics = (
        "validation_curve_max_depth",
        "feature_importance",
        "arbre_tronque",
        "learning_curve",
        "confusion_classes_difficiles",
    )
    core_figures = (
        "validation_curve_max_depth",
        "feature_importance",
        "arbre_tronque",
        "learning_curve",
    )
    advanced_figures = ("confusion_classes_difficiles",)

    def creer_estimateur(self):
        return DecisionTreeClassifier(random_state=RANDOM_SEED)

    def grille_tuning(self) -> dict:
        return {
            "modele__max_depth": [None, 5, 10, 20],
            "modele__min_samples_leaf": [1, 2, 4, 8],
            "modele__criterion": ["gini", "entropy"],
            "modele__ccp_alpha": [0.0, 0.001, 0.005, 0.01],
        }
