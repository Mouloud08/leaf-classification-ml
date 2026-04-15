"""Classe abstraite définissant l'interface commune à tous les modèles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import PCA_EXPLAINED_VARIANCE, RANDOM_SEED


class ModeleBase(ABC):
    """Interface OOP commune pour tous les modèles du projet."""

    nom: str
    nom_affiche: str
    famille: str
    description: str
    hypothese: str
    utiliser_pca_par_defaut: bool = False
    necessite_scaling: bool = False
    supports_probabilities: bool = False

    # Diagnostics supportés par le modèle
    supported_diagnostics: tuple[str, ...] = ()
    # Figures principales générées automatiquement
    core_figures: tuple[str, ...] = ()
    # Figures supplémentaires activables à la demande
    advanced_figures: tuple[str, ...] = ()

    @abstractmethod
    def creer_estimateur(self) -> BaseEstimator:
        """Instancie l'estimateur sklearn brut avec ses hyperparamètres par défaut."""

    @abstractmethod
    def grille_tuning(self) -> dict:
        """Retourne la grille de tuning avec clés préfixées ``modele__``."""

    def creer_pipeline(self, utiliser_pca: bool | None = None) -> Pipeline:
        """Construit un pipeline sklearn adapté au modèle.

        Le standardiseur est ajouté si ``necessite_scaling`` est vrai.
        Le PCA est ajouté si *utiliser_pca* est vrai (par défaut :
        ``self.utiliser_pca_par_defaut``).
        """
        if utiliser_pca is None:
            utiliser_pca = self.utiliser_pca_par_defaut

        if utiliser_pca and not self.necessite_scaling:
            raise ValueError(
                f"Le modèle '{self.nom}' ne nécessite pas de mise à l'échelle ; "
                "l'ajout de PCA sans standardisation en amont n'est pas supporté. "
                "Passez utiliser_pca=False ou activez necessite_scaling."
            )

        etapes: list[tuple[str, BaseEstimator]] = []

        if self.necessite_scaling:
            etapes.append(("scaler", StandardScaler()))

        if utiliser_pca:
            etapes.append(
                (
                    "pca",
                    PCA(n_components=PCA_EXPLAINED_VARIANCE, random_state=RANDOM_SEED),
                )
            )

        etapes.append(("modele", self.creer_estimateur()))
        return Pipeline(etapes)

    def definition(self) -> dict:
        """Retourne les métadonnées du modèle (compatibles avec l'ancien format)."""
        return {
            "nom_affiche": self.nom_affiche,
            "famille": self.famille,
            "description": self.description,
            "hypothese": self.hypothese,
            "utiliser_pca_par_defaut": self.utiliser_pca_par_defaut,
            "grille_tuning": deepcopy(self.grille_tuning()),
        }

    def supports_diagnostic(self, name: str) -> bool:
        """Vérifie si le modèle supporte un diagnostic spécifique."""
        return name in self.supported_diagnostics

    def figure_category(self, name: str) -> str | None:
        """Retourne la catégorie d'une figure (`core`, `advanced`) ou `None`."""
        if name in self.core_figures:
            return "core"
        if name in self.advanced_figures:
            return "advanced"
        return None
