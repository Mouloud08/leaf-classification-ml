"""Definitions des modeles de classification du projet."""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .pipelines import PipelineFactory


class BaseClassifier(ABC):
    """Classe de base pour les modeles du projet.

    Cette classe centralise les attributs communs a tous les modeles et sert de point
    d'extension pour l'entrainement, la prediction et la persistance.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        grille_params: dict[str, list[Any]],
        nom: str,
    ) -> None:
        """Initialise les attributs communs du classifieur.

        Args:
            pipeline: Pipeline sklearn complet du modele.
            grille_params: Grille d'hyperparametres pour `GridSearchCV`.
            nom: Nom lisible du modele.
        """
        self.pipeline = pipeline
        self.grille_params = grille_params
        self.nom = nom
        self.meilleur_modele: Pipeline | None = None
        self.meilleurs_params: dict[str, Any] | None = None
        self.meilleur_score: float | None = None

    def entrainer(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> None:
        """Entraine le modele via validation croisee et recherche de grille.

        Args:
            X: Matrice de variables explicatives.
            y: Labels encodes.

        Raises:
            NotImplementedError: Methode volontairement laissee en squelette.
        """
        # TODO: configurer StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # TODO: instancier GridSearchCV avec scoring="f1_macro"
        # TODO: ajuster la recherche sur l'ensemble d'entrainement
        # TODO: sauvegarder best_estimator_, best_params_ et best_score_
        raise NotImplementedError(
            "La methode 'entrainer' doit etre implemente pour finaliser le projet."
        )

    def predire(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Produit des predictions a partir du meilleur modele entraine.

        Args:
            X: Matrice des exemples a predire.

        Returns:
            Tableau de predictions.

        Raises:
            NotImplementedError: Methode volontairement laissee en squelette.
        """
        # TODO: verifier que `self.meilleur_modele` n'est pas None
        # TODO: deleguer a `self.meilleur_modele.predict(X)`
        raise NotImplementedError(
            "La methode 'predire' doit etre implemente pour finaliser le projet."
        )

    def sauvegarder(self, chemin: Path) -> None:
        """Sauvegarde le meilleur modele sur disque avec joblib.

        Args:
            chemin: Emplacement du fichier `.pkl`.
        """
        # TODO: creer le dossier parent si necessaire
        # TODO: verifier qu'un modele entraine est disponible
        # TODO: utiliser `joblib.dump(...)`
        raise NotImplementedError(
            "La methode 'sauvegarder' doit etre implemente pour finaliser le projet."
        )

    @classmethod
    def charger(cls, chemin: Path) -> Any:
        """Charge un modele prealablement sauvegarde.

        Args:
            chemin: Emplacement du fichier `.pkl`.

        Returns:
            Objet charge depuis le disque.
        """
        # TODO: definir si l'on charge uniquement le pipeline ou une instance complete
        # TODO: harmoniser le format de serialisation pour tous les modeles
        raise NotImplementedError(
            "La methode 'charger' doit etre implemente pour finaliser le projet."
        )


class PerceptronModele(BaseClassifier):
    """Squelette du modele Perceptron."""

    def __init__(self) -> None:
        modele = Perceptron(random_state=42, max_iter=1000)
        pipeline = PipelineFactory.creer_pipeline_lineaire(modele)
        grille_params = {
            "classifier__alpha": [0.0001, 0.001, 0.01],
            "classifier__penalty": [None, "l2", "l1", "elasticnet"],
        }
        super().__init__(pipeline=pipeline, grille_params=grille_params, nom="Perceptron")


class RegressionLogistiqueModele(BaseClassifier):
    """Squelette du modele de regression logistique."""

    def __init__(self) -> None:
        modele = LogisticRegression(
            random_state=42,
            max_iter=2000,
            multi_class="auto",
        )
        pipeline = PipelineFactory.creer_pipeline_lineaire(modele)
        grille_params = {
            "classifier__C": [0.1, 1.0, 10.0, 100.0],
            "classifier__solver": ["lbfgs", "saga"],
        }
        super().__init__(
            pipeline=pipeline,
            grille_params=grille_params,
            nom="RegressionLogistique",
        )


class SVMModele(BaseClassifier):
    """Squelette du modele SVM a noyau RBF."""

    def __init__(self) -> None:
        modele = SVC(kernel="rbf", random_state=42)
        pipeline = PipelineFactory.creer_pipeline_lineaire(modele)
        grille_params = {
            "classifier__C": [0.1, 1.0, 10.0, 100.0],
            "classifier__gamma": ["scale", 0.01, 0.001],
        }
        super().__init__(pipeline=pipeline, grille_params=grille_params, nom="SVM_RBF")


class ForetAleatoireModele(BaseClassifier):
    """Squelette du modele Random Forest."""

    def __init__(self) -> None:
        modele = RandomForestClassifier(random_state=42)
        pipeline = PipelineFactory.creer_pipeline_arbre(modele)
        grille_params = {
            "classifier__n_estimators": [100, 300, 500],
            "classifier__max_depth": [None, 10, 20],
        }
        super().__init__(
            pipeline=pipeline,
            grille_params=grille_params,
            nom="ForetAleatoire",
        )


class AdaBoostModele(BaseClassifier):
    """Squelette du modele AdaBoost."""

    def __init__(self) -> None:
        modele = AdaBoostClassifier(random_state=42)
        pipeline = PipelineFactory.creer_pipeline_arbre(modele)
        grille_params = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.01, 0.1, 1.0],
        }
        super().__init__(pipeline=pipeline, grille_params=grille_params, nom="AdaBoost")


class MLPModele(BaseClassifier):
    """Squelette du modele MLP."""

    def __init__(self) -> None:
        modele = MLPClassifier(random_state=42, max_iter=500)
        pipeline = PipelineFactory.creer_pipeline_lineaire(modele)
        grille_params = {
            "classifier__hidden_layer_sizes": [(100,), (100, 50), (200,)],
            "classifier__alpha": [0.0001, 0.001, 0.01],
        }
        super().__init__(pipeline=pipeline, grille_params=grille_params, nom="MLP")


# TODO: choisir si l'on conserve la logique de persistance dans la classe de base
# TODO: ajouter possiblement une methode commune pour exposer les resultats d'entrainement
