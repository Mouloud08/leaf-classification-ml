"""Paquet principal du projet Leaf Classification pour IFT 712."""

from .data_manager import LeafDataLoader
from .evaluator import ModelEvaluator
from .models import (
    AdaBoostModele,
    BaseClassifier,
    ForetAleatoireModele,
    MLPModele,
    PerceptronModele,
    RegressionLogistiqueModele,
    SVMModele,
)
from .pipelines import PipelineFactory

__all__ = [
    "LeafDataLoader",
    "PipelineFactory",
    "BaseClassifier",
    "PerceptronModele",
    "RegressionLogistiqueModele",
    "SVMModele",
    "ForetAleatoireModele",
    "AdaBoostModele",
    "MLPModele",
    "ModelEvaluator",
]
