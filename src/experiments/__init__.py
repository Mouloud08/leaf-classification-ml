"""Package d'orchestration des experiences."""

from .figure_runner import generer_figures_generiques, generer_figures_specifiques
from .model_specs import (
    ALL_MODEL_NAMES,
    MODEL_STUDY_SPECS,
    obtenir_model_study_spec,
)
from .report_bundle import generer_comparaison_globale, generer_manifeste
from .runner import (
    charger_predictions_tuned_si_disponibles,
    executer_etude,
    executer_modele,
)
from .shared import (
    construire_estimateur_variante,
    creer_cv,
    selectionner_meilleure_variante_untuned,
)
from .study_spec import ModelStudySpec, StudySpec, UntunedVariantSpec

__all__ = [
    "ALL_MODEL_NAMES",
    "MODEL_STUDY_SPECS",
    "ModelStudySpec",
    "StudySpec",
    "UntunedVariantSpec",
    "charger_predictions_tuned_si_disponibles",
    "construire_estimateur_variante",
    "creer_cv",
    "executer_etude",
    "executer_modele",
    "generer_comparaison_globale",
    "generer_figures_generiques",
    "generer_figures_specifiques",
    "generer_manifeste",
    "obtenir_model_study_spec",
    "selectionner_meilleure_variante_untuned",
]
