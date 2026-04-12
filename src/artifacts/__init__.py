"""Package de gestion des artefacts — schema, chemins, stores."""

from .figures_store import sauvegarder_figure, sauvegarder_figure_globale
from .metrics_store import (
    charger_mesures,
    charger_tableau_mesures,
    charger_tableau_mesures_valides,
    sauvegarder_mesures,
)
from .paths import (
    GlobalArtifactPaths,
    ModelArtifactPaths,
    global_paths,
    model_paths,
)
from .predictions_store import (
    charger_predictions,
    charger_probabilites,
    sauvegarder_predictions,
    sauvegarder_probabilites_oof,
)
from .schema import (
    METRICS_SCHEMA_VERSION,
    FigureBundleResult,
    FigureRequest,
    NestedEvaluationResult,
    StudyRunResult,
    TuningRunResult,
    UntunedVariantResult,
)
from .tuning_store import (
    charger_resultats_grid_search,
    sauvegarder_resultats_grid_search,
)
from .validation import (
    LEGACY_KEYS,
    REQUIRED_METRIC_KEYS,
    STAGE_MAP,
    MetricsArtifactValidationError,
    derive_selection_protocol,
    derive_stage,
    filtrer_legacy,
    normaliser_mesures_pour_schema,
    normaliser_stades,
    valider_artefacts_mesures,
)

__all__ = [
    "GlobalArtifactPaths",
    "LEGACY_KEYS",
    "METRICS_SCHEMA_VERSION",
    "MetricsArtifactValidationError",
    "ModelArtifactPaths",
    "FigureBundleResult",
    "FigureRequest",
    "NestedEvaluationResult",
    "STAGE_MAP",
    "StudyRunResult",
    "TuningRunResult",
    "UntunedVariantResult",
    "charger_mesures",
    "charger_predictions",
    "charger_probabilites",
    "charger_resultats_grid_search",
    "charger_tableau_mesures",
    "charger_tableau_mesures_valides",
    "derive_selection_protocol",
    "derive_stage",
    "filtrer_legacy",
    "global_paths",
    "model_paths",
    "normaliser_mesures_pour_schema",
    "normaliser_stades",
    "REQUIRED_METRIC_KEYS",
    "sauvegarder_figure",
    "sauvegarder_figure_globale",
    "sauvegarder_mesures",
    "sauvegarder_predictions",
    "sauvegarder_probabilites_oof",
    "sauvegarder_resultats_grid_search",
    "valider_artefacts_mesures",
]
