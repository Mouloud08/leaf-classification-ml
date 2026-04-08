"""Outils d'evaluation et de visualisation pour les modeles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class ModelEvaluator:
    """Classe utilitaire pour calculer et comparer les performances des modeles."""

    @staticmethod
    def calculer_metriques(
        y_vrai: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Calcule les metriques principales du projet.

        Args:
            y_vrai: Labels de reference.
            y_pred: Predictions du modele.

        Returns:
            Dictionnaire des metriques calculees.
        """
        # TODO: utiliser accuracy_score et f1_score(average="macro")
        # TODO: retourner un dictionnaire standardise pour tous les modeles
        raise NotImplementedError(
            "La methode 'calculer_metriques' doit etre implemente pour finaliser le projet."
        )

    @staticmethod
    def comparer_modeles(resultats: dict[str, dict[str, float]]) -> pd.DataFrame:
        """Construit un tableau comparatif trie des performances.

        Args:
            resultats: Dictionnaire imbrique contenant les metriques par modele.

        Returns:
            DataFrame comparatif trie.
        """
        # TODO: convertir le dictionnaire en DataFrame pandas
        # TODO: trier le tableau sur `f1_macro` de facon decroissante
        raise NotImplementedError(
            "La methode 'comparer_modeles' doit etre implemente pour finaliser le projet."
        )

    @staticmethod
    def afficher_top_erreurs_confusion(
        y_vrai: np.ndarray,
        y_pred: np.ndarray,
        label_encoder: Any,
        top_n: int = 10,
    ) -> None:
        """Affiche les paires de classes les plus confondues.

        Args:
            y_vrai: Labels de reference.
            y_pred: Predictions du modele.
            label_encoder: Encodeur permettant de decoder les labels.
            top_n: Nombre de confusions a afficher.
        """
        # TODO: calculer la matrice de confusion sans afficher 99 x 99 cases
        # TODO: extraire les pires paires (classe vraie, classe predite)
        # TODO: tracer un barplot horizontal avec les noms d'especes
        raise NotImplementedError(
            "La methode 'afficher_top_erreurs_confusion' doit etre implemente."
        )

    @staticmethod
    def tracer_comparaison_modeles(resultats: dict[str, dict[str, float]]) -> None:
        """Trace un graphique comparatif des modeles.

        Args:
            resultats: Dictionnaire contenant les metriques par modele.
        """
        # TODO: convertir les resultats dans un format facilement tracable
        # TODO: produire un barplot de comparaison du F1-macro
        raise NotImplementedError(
            "La methode 'tracer_comparaison_modeles' doit etre implemente."
        )
