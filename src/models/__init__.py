"""Package OOP des modeles de classification."""

from .arbre_decision import ModeleArbreDecision
from .base import ModeleBase
from .foret_aleatoire import ModeleForetAleatoire
from .mlp import ModeleMLP
from .perceptron import ModelePerceptron
from .pipelines import creer_estimateur, creer_pipeline, obtenir_grille_tuning
from .regression_logistique import ModeleRegressionLogistique
from .svm import ModeleSVM

MODELES: dict[str, type[ModeleBase]] = {
    "perceptron": ModelePerceptron,
    "regression_logistique": ModeleRegressionLogistique,
    "svm": ModeleSVM,
    "foret_aleatoire": ModeleForetAleatoire,
    "arbre_decision": ModeleArbreDecision,
    "mlp": ModeleMLP,
}

__all__ = [
    "MODELES",
    "ModeleArbreDecision",
    "ModeleBase",
    "ModeleForetAleatoire",
    "ModeleMLP",
    "ModelePerceptron",
    "ModeleRegressionLogistique",
    "ModeleSVM",
    "creer_estimateur",
    "creer_pipeline",
    "obtenir_grille_tuning",
]
