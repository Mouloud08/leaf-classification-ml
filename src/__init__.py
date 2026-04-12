"""Exports du projet Leaf Classification.

API publique minimale — les packages specialises sont accessibles
directement (src.data, src.models, src.evaluation, src.artifacts,
src.plots, src.experiments, src.cli).
"""

# --- Config ---
from .config import (
    DEFAULT_TRAIN_CSV,
    FIGURES_DIR,
    GRID_SEARCH_DIR,
    METRICS_DIR,
    N_SPLITS,
    PCA_COMPONENTS,
    PREDICTIONS_DIR,
    PRIMARY_METRIC,
    RANDOM_SEED,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
)

# --- Data ---
from .data import LeafDataLoader, charger_donnees

# --- OOP Models ---
from .models import (
    MODELES,
    ModeleArbreDecision,
    ModeleBase,
    ModeleForetAleatoire,
    ModeleMLP,
    ModelePerceptron,
    ModeleRegressionLogistique,
    ModeleSVM,
)

# --- Pipelines ---
from .models.pipelines import creer_estimateur, creer_pipeline, obtenir_grille_tuning

# --- Notebook support ---
from .notebook_support import (
    NOTEBOOK_MODEL_SPECS,
    NotebookContext,
    NotebookModelSpec,
    UntunedVariantSpec,
    calculer_importance_permutation_exploratoire,
    calculer_metriques_classes_depuis_predictions,
    construire_tableau_gaps,
    construire_tableau_plis_externes,
    construire_tableau_variantes,
    creer_notebook_context,
    evaluer_variantes_untuned,
    executer_tuning_confirmatoire,
    obtenir_notebook_model_spec,
)

# --- Evaluation ---
from .evaluation import (
    calculer_metriques_par_classe,
    calculer_top_k_accuracies,
    construire_matrice_confusion,
    evaluer_modele_cv,
    evaluer_modele_tuned_nested_cv,
    probabilites_oof,
)

# --- Results I/O ---
from .artifacts import (
    MetricsArtifactValidationError,
    REQUIRED_METRIC_KEYS,
    charger_tableau_mesures_valides,
    filtrer_legacy,
    normaliser_stades,
    sauvegarder_mesures,
    sauvegarder_probabilites_oof,
    valider_artefacts_mesures,
)

# --- Visualization ---
from .plots import (
    tracer_comparaison_variantes,
    tracer_courbes_precision_rappel_ovr,
    tracer_courbes_roc_ovr,
    tracer_heatmap_tuning,
    tracer_histogramme_confiance,
    tracer_hyperparametre_vs_performance,
    tracer_matrice_confusion,
    tracer_metrique_par_classe,
    tracer_reliability_diagram,
    tracer_top_confusions,
)

__all__ = [
    # Config
    "DEFAULT_TRAIN_CSV",
    "FIGURES_DIR",
    "GRID_SEARCH_DIR",
    "METRICS_DIR",
    "N_SPLITS",
    "PCA_COMPONENTS",
    "PREDICTIONS_DIR",
    "PRIMARY_METRIC",
    "RANDOM_SEED",
    "RESULTS_DIR",
    "SAVED_MODELS_DIR",
    # Data
    "LeafDataLoader",
    "charger_donnees",
    # OOP Models
    "MODELES",
    "ModeleArbreDecision",
    "ModeleBase",
    "ModeleForetAleatoire",
    "ModeleMLP",
    "ModelePerceptron",
    "ModeleRegressionLogistique",
    "ModeleSVM",
    # Pipelines
    "creer_estimateur",
    "creer_pipeline",
    "obtenir_grille_tuning",
    # Notebook support
    "calculer_importance_permutation_exploratoire",
    "calculer_metriques_classes_depuis_predictions",
    "calculer_metriques_par_classe",
    "calculer_top_k_accuracies",
    "construire_matrice_confusion",
    "construire_tableau_gaps",
    "construire_tableau_plis_externes",
    "construire_tableau_variantes",
    "creer_notebook_context",
    "evaluer_modele_cv",
    "evaluer_modele_tuned_nested_cv",
    "evaluer_variantes_untuned",
    "executer_tuning_confirmatoire",
    "NOTEBOOK_MODEL_SPECS",
    "NotebookContext",
    "NotebookModelSpec",
    "UntunedVariantSpec",
    "obtenir_notebook_model_spec",
    "probabilites_oof",
    # Results I/O
    "MetricsArtifactValidationError",
    "REQUIRED_METRIC_KEYS",
    "charger_tableau_mesures_valides",
    "filtrer_legacy",
    "normaliser_stades",
    "sauvegarder_mesures",
    "sauvegarder_probabilites_oof",
    "valider_artefacts_mesures",
    # Visualization
    "tracer_comparaison_variantes",
    "tracer_courbes_precision_rappel_ovr",
    "tracer_courbes_roc_ovr",
    "tracer_heatmap_tuning",
    "tracer_histogramme_confiance",
    "tracer_hyperparametre_vs_performance",
    "tracer_matrice_confusion",
    "tracer_metrique_par_classe",
    "tracer_reliability_diagram",
    "tracer_top_confusions",
]
