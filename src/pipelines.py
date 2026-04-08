"""Fabrique de pipelines sklearn pour les modeles du projet."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PipelineFactory:
    """Construit les pipelines de pretraitement selon la famille du modele."""

    @staticmethod
    def creer_pipeline_lineaire(modele: BaseEstimator) -> Pipeline:
        """Cree un pipeline pour les modeles sensibles a l'echelle des variables.

        Args:
            modele: Estimateur sklearn a placer en fin de pipeline.

        Returns:
            Pipeline `StandardScaler -> PCA(0.99) -> classifier`.
        """
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.99, random_state=42)),
                ("classifier", modele),
            ]
        )

    @staticmethod
    def creer_pipeline_arbre(modele: BaseEstimator) -> Pipeline:
        """Cree un pipeline minimal pour les modeles a base d'arbres.

        Les arbres de decision et les ensembles bases sur des arbres n'ont pas besoin
        d'une standardisation des variables, car ils partitionnent l'espace par seuils
        et ne reposent ni sur une distance euclidienne ni sur une descente de gradient
        sensible a l'echelle des variables.

        Args:
            modele: Estimateur sklearn a placer en fin de pipeline.

        Returns:
            Pipeline ne contenant que l'estimateur final.
        """
        return Pipeline(steps=[("classifier", modele)])
