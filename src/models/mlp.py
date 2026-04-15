"""Modèle MLP (Multi-Layer Perceptron)."""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from ..config import PCA_EXPLAINED_VARIANCE, RANDOM_SEED
from .base import ModeleBase


class ModeleMLP(ModeleBase):
    nom = "mlp"
    nom_affiche = "MLP"
    famille = "réseau_de_neurones"
    description = "Réseau dense multicouche adapté aux frontières très non linéaires."
    hypothese = (
        "Le MLP devrait profiter fortement de la mise à l'échelle ; "
        "le gain de l'ajustement doit être confirmé empiriquement."
    )
    utiliser_pca_par_defaut = True
    necessite_scaling = True
    supports_probabilities = True

    supported_diagnostics = (
        "loss_curve",
        "learning_curve",
        "heatmap_hidden_alpha",
        "surconfiance_erreurs",
        "permutation_importance",
        "tsne_representations",
    )
    core_figures = (
        "loss_curve",
        "learning_curve",
        "heatmap_hidden_alpha",
        "surconfiance_erreurs",
        "permutation_importance",
    )
    advanced_figures = ("tsne_representations",)

    def creer_estimateur(self):
        return MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=300,
            early_stopping=True,
            random_state=RANDOM_SEED,
        )

    def grille_tuning(self) -> dict:
        return {
            "pca": [
                "passthrough",
                PCA(n_components=PCA_EXPLAINED_VARIANCE, random_state=RANDOM_SEED),
            ],
            "modele__hidden_layer_sizes": [(64,), (128,)],
            "modele__alpha": [0.0001, 0.001],
        }
