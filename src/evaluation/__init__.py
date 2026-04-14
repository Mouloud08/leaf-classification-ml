"""Package d'evaluation — metriques pures, validation croisee, diagnostics."""

from .evaluateur import (  # noqa: F401
    ModelEvaluator,
    comparer_variantes,
    evaluer_dummy_classifier,
    evaluer_modele_cv,
    evaluer_modele_tuned_nested_cv,
    predictions_oof,
    probabilites_oof,
)
from .metriques import (  # noqa: F401
    CHANCE_LEVEL,
    calculer_brier_score_multiclasse,
    calculer_metriques,
    calculer_metriques_par_classe,
    calculer_points_precision_rappel_ovr,
    calculer_points_roc_ovr,
    calculer_reliability_curve,
    calculer_statistiques_confiance,
    calculer_top_k_accuracies,
    construire_matrice_confusion,
    extraire_scores_probabilites,
    extraire_top_confusions,
    resumer_metriques_par_classe,
)
from .diagnostics_generiques import (  # noqa: F401
    classes_difficiles,
    comparaison_baseline_best_tuned,
    matrice_confusion_normalisee,
    tableau_gaps,
    tableau_plis_externes,
    tableau_variantes,
    top_confusions,
)
from .learning_curves import compute_learning_curve  # noqa: F401
from .sanity_checks import (  # noqa: F401
    verifier_bruit_aleatoire,
    verifier_overfit_petit_echantillon,
)

__all__ = [
    "CHANCE_LEVEL",
    "calculer_brier_score_multiclasse",
    "ModelEvaluator",
    "calculer_metriques",
    "calculer_metriques_par_classe",
    "calculer_points_precision_rappel_ovr",
    "calculer_points_roc_ovr",
    "calculer_reliability_curve",
    "calculer_statistiques_confiance",
    "calculer_top_k_accuracies",
    "classes_difficiles",
    "comparer_variantes",
    "comparaison_baseline_best_tuned",
    "compute_learning_curve",
    "construire_matrice_confusion",
    "evaluer_dummy_classifier",
    "evaluer_modele_cv",
    "evaluer_modele_tuned_nested_cv",
    "extraire_scores_probabilites",
    "extraire_top_confusions",
    "matrice_confusion_normalisee",
    "predictions_oof",
    "probabilites_oof",
    "resumer_metriques_par_classe",
    "tableau_gaps",
    "tableau_plis_externes",
    "tableau_variantes",
    "top_confusions",
    "verifier_bruit_aleatoire",
    "verifier_overfit_petit_echantillon",
]
