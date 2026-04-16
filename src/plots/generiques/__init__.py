"""Graphiques génériques : comparaison, confusion, courbes et ajustement."""

from .comparison import (
    tracer_barres_comparaison,
    tracer_corroboration_holdout_global,
    tracer_comparaison_variantes,
    tracer_comparaison_variantes_rapport,
    tracer_temps_vs_performance,
)
from .confusion import (
    tracer_matrice_confusion,
    tracer_top_confusions,
)
from .curves import (
    tracer_courbes_precision_rappel_ovr,
    tracer_courbes_roc_ovr,
    tracer_histogramme_confiance,
    tracer_reliability_diagram,
)
from .tuning import (
    tracer_heatmap_tuning,
    tracer_hyperparametre_vs_performance,
    tracer_metrique_par_classe,
)

__all__ = [
    "tracer_barres_comparaison",
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
