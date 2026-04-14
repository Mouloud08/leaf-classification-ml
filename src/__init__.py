"""Public entry points for the Leaf Classification project."""

from .artifacts import charger_tableau_mesures_valides
from .config import DEFAULT_TRAIN_CSV, N_SPLITS, RANDOM_SEED, RESULTS_DIR
from .data import charger_donnees
from .evaluation import evaluer_modele_cv, evaluer_modele_tuned_nested_cv
from .experiments import ALL_MODEL_NAMES, StudySpec, executer_etude, obtenir_model_study_spec
from .models import MODELES
from .models.pipelines import creer_estimateur, creer_pipeline
from .notebook_support import NOTEBOOK_MODEL_SPECS, creer_notebook_context

__all__ = [
    "ALL_MODEL_NAMES",
    "DEFAULT_TRAIN_CSV",
    "MODELES",
    "NOTEBOOK_MODEL_SPECS",
    "N_SPLITS",
    "RANDOM_SEED",
    "RESULTS_DIR",
    "StudySpec",
    "charger_donnees",
    "charger_tableau_mesures_valides",
    "creer_estimateur",
    "creer_notebook_context",
    "creer_pipeline",
    "evaluer_modele_cv",
    "evaluer_modele_tuned_nested_cv",
    "executer_etude",
    "obtenir_model_study_spec",
]
