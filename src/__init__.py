"""Points d'entrée publics du projet de classification des feuilles."""

from .artifacts import charger_tableau_mesures_valides
from .config import DEFAULT_TRAIN_CSV, N_SPLITS, RANDOM_SEED, RESULTS_DIR
from .data import charger_donnees
from .evaluation import (
    calculer_top_k_accuracies,
    construire_matrice_confusion,
    evaluer_modele_cv,
    evaluer_modele_tuned_nested_cv,
)
from .experiments import (
    ALL_MODEL_NAMES,
    StudySpec,
    executer_etude,
    obtenir_model_study_spec,
)
from .models import MODELES
from .models.pipelines import creer_estimateur, creer_pipeline
from .notebook_support import (
    NOTEBOOK_MODEL_SPECS,
    SPECIFICATIONS_MODELES_NOTEBOOK,
    calculer_importance_permutation_exploratoire,
    calculer_metriques_classes_depuis_predictions,
    construire_tableau_gaps,
    construire_tableau_plis_externes,
    construire_tableau_variantes,
    creer_contexte_notebook,
    creer_notebook_context,
    evaluer_variantes_untuned,
    executer_tuning_confirmatoire,
    extraire_images_notebook,
)
from .plots import (
    tracer_courbes_precision_rappel_ovr,
    tracer_heatmap_tuning,
    tracer_histogramme_confiance,
    tracer_matrice_confusion,
    tracer_metrique_par_classe,
    tracer_reliability_diagram,
    tracer_top_confusions,
)

__all__ = [
    "ALL_MODEL_NAMES",
    "DEFAULT_TRAIN_CSV",
    "MODELES",
    "NOTEBOOK_MODEL_SPECS",
    "SPECIFICATIONS_MODELES_NOTEBOOK",
    "N_SPLITS",
    "RANDOM_SEED",
    "RESULTS_DIR",
    "StudySpec",
    "calculer_importance_permutation_exploratoire",
    "calculer_metriques_classes_depuis_predictions",
    "calculer_top_k_accuracies",
    "charger_donnees",
    "charger_tableau_mesures_valides",
    "construire_matrice_confusion",
    "construire_tableau_gaps",
    "construire_tableau_plis_externes",
    "construire_tableau_variantes",
    "creer_contexte_notebook",
    "creer_estimateur",
    "creer_notebook_context",
    "creer_pipeline",
    "extraire_images_notebook",
    "evaluer_modele_cv",
    "evaluer_modele_tuned_nested_cv",
    "evaluer_variantes_untuned",
    "executer_etude",
    "executer_tuning_confirmatoire",
    "obtenir_model_study_spec",
    "tracer_courbes_precision_rappel_ovr",
    "tracer_heatmap_tuning",
    "tracer_histogramme_confiance",
    "tracer_matrice_confusion",
    "tracer_metrique_par_classe",
    "tracer_reliability_diagram",
    "tracer_top_confusions",
]
