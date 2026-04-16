"""Package de visualisation : graphiques génériques et spécifiques."""

# Ré-export des graphiques génériques depuis les deux emplacements
# (ancien : fichiers à plat, nouveau : sous-package generiques/)
from .generiques import (
    tracer_barres_comparaison,
    tracer_comparaison_variantes,
    tracer_comparaison_variantes_rapport,
    tracer_corroboration_globale,
    tracer_corroboration_holdout_global,
    tracer_courbes_precision_rappel_ovr,
    tracer_courbes_roc_ovr,
    tracer_heatmap_tuning,
    tracer_histogramme_confiance,
    tracer_hyperparametre_vs_performance,
    tracer_matrice_confusion,
    tracer_metrique_par_classe,
    tracer_reliability_diagram,
    tracer_temps_vs_performance,
    tracer_top_confusions,
)

__all__ = [
    "tracer_barres_comparaison",
    "tracer_corroboration_globale",
    "tracer_corroboration_holdout_global",
    "tracer_comparaison_variantes",
    "tracer_comparaison_variantes_rapport",
    "tracer_courbes_precision_rappel_ovr",
    "tracer_courbes_roc_ovr",
    "tracer_heatmap_tuning",
    "tracer_histogramme_confiance",
    "tracer_hyperparametre_vs_performance",
    "tracer_matrice_confusion",
    "tracer_metrique_par_classe",
    "tracer_reliability_diagram",
    "tracer_temps_vs_performance",
    "tracer_top_confusions",
]
