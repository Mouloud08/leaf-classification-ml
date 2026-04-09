"""Construction des pipelines baseline et preprocesses."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PCA_COMPONENTS

MODELES_AVEC_SCALING = {"perceptron", "regression_logistique", "svm", "mlp"}
MODELES_AVEC_PCA = {"perceptron", "regression_logistique", "svm", "mlp"}


def _pipeline_depuis_estimateur(
    nom_modele: str,
    estimateur: BaseEstimator,
    utiliser_pca: bool = False,
) -> Pipeline:
    """Construit le pipeline ameliore selon la famille du modele."""
    etapes: list[tuple[str, BaseEstimator]] = []

    if nom_modele in MODELES_AVEC_SCALING:
        etapes.append(("scaler", StandardScaler()))

    if utiliser_pca and nom_modele in MODELES_AVEC_PCA:
        etapes.append(("pca", PCA(n_components=PCA_COMPONENTS, random_state=42)))

    etapes.append(("modele", estimateur))
    return Pipeline(etapes)


def creer_pipeline_ameliore(
    nom_modele: str,
    utiliser_pca: bool = False,
) -> Pipeline:
    """Cree le pipeline preprocessse recommande pour un modele."""
    from .modeles import creer_modele_simple

    cle = nom_modele.lower()
    estimateur = creer_modele_simple(cle)
    return _pipeline_depuis_estimateur(
        nom_modele=cle,
        estimateur=estimateur,
        utiliser_pca=utiliser_pca,
    )


class PipelineFactory:
    """Compatibilite avec l'ancienne interface du projet."""

    @staticmethod
    def creer_pipeline_lineaire(
        modele: BaseEstimator,
        utiliser_pca: bool = True,
    ) -> Pipeline:
        return _pipeline_depuis_estimateur(
            nom_modele="perceptron",
            estimateur=modele,
            utiliser_pca=utiliser_pca,
        )

    @staticmethod
    def creer_pipeline_arbre(modele: BaseEstimator) -> Pipeline:
        return Pipeline([("modele", modele)])

