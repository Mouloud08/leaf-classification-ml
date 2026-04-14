"""Generation des figures pour un modele apres execution."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
import pandas as pd

def _is_notebook() -> bool:
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # type: ignore[name-defined]
    except NameError:
        return False

if not _is_notebook():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..artifacts import (
    FigureBundleResult,
    model_paths,
    sauvegarder_figure,
)
from ..data import charger_donnees
from ..evaluation import (
    construire_matrice_confusion,
    extraire_top_confusions,
)
from ..evaluation.diagnostics_generiques import (
    classes_difficiles,
    matrice_confusion_normalisee,
    tableau_variantes,
)
from ..evaluation.diagnostics_specifiques import (
    analyse_support_vectors,
    coefficients_absolus,
    compute_learning_curve,
    decision_function_histogram,
    importance_oob_foret,
    loss_curve_mlp,
    oob_vs_n_estimators,
    regularization_path,
    validation_curve_max_depth,
)
from ..models import MODELES
from ..models.pipelines import creer_estimateur, creer_pipeline
from ..plots.generiques import (
    tracer_comparaison_variantes,
    tracer_matrice_confusion,
    tracer_metrique_par_classe,
    tracer_top_confusions,
)
from ..plots.specifiques import arbre_decision as plots_arbre
from ..plots.specifiques import foret_aleatoire as plots_foret
from ..plots.specifiques import mlp as plots_mlp
from ..plots.specifiques import perceptron as plots_perceptron
from ..plots.specifiques import regression_logistique as plots_reglog
from ..plots.specifiques import svm as plots_svm
from .model_specs import obtenir_model_study_spec
from .study_spec import StudySpec

logger = logging.getLogger(__name__)

EXPLORATORY_FIGURES: frozenset[str] = frozenset(
    {
        "arbre_tronque",
        "feature_importance",
        "oob_vs_n_estimators",
        "importance_dispersion",
        "loss_curve",
        "coefficients",
        "support_vectors",
        "decision_function_histogram",
    }
)


def _preparer_diagnostic(model_name: str, X: pd.DataFrame):
    """Prepare donnees scalees + estimateur pour diagnostics exploratoires.

    NOTE: le fit du preprocessing se fait sur X complet. C'est intentionnel —
    ces diagnostics servent a la visualisation, pas a l'evaluation.
    """
    pipeline = creer_pipeline(model_name, utiliser_pca=False)
    preprocessing = pipeline[:-1]
    X_prep = pd.DataFrame(preprocessing.fit_transform(X), columns=X.columns)
    est = pipeline.named_steps["modele"]
    return X_prep, est


def _save(
    fig: plt.Figure,
    name: str,
    paths,
    category: str = "core",
    exploratory: bool = False,
) -> str:
    """Sauvegarde et ferme une figure."""
    output_name = f"exploratory__{name}" if exploratory else name
    sauvegarder_figure(fig, output_name, paths, category=category)
    return f"{name} [exploratory]" if exploratory else name


def generer_figures_generiques(
    model_name: str,
    untuned_dicts: list[dict[str, Any]],
    tuned_dict: dict[str, Any] | None,
    y_true: Any,
    y_pred: Any,
    label_encoder: Any,
    study: StudySpec,
) -> list[str]:
    """Genere les figures generiques core pour un modele."""
    paths = model_paths(model_name, root=study.output_root)
    paths.ensure_dirs()
    generated = []

    # Comparaison des variantes
    tab = tableau_variantes(untuned_dicts, tuned_dict)
    cols = [
        "val_accuracy_mean",
        "val_f1_macro_mean",
        "train_accuracy_mean",
        "train_f1_macro_mean",
    ]
    cols_present = [c for c in cols if c in tab.columns]
    if cols_present:
        ax = tracer_comparaison_variantes(tab, cols_present, titre=f"{model_name} - Variants")
        generated.append(_save(ax.figure, "comparaison_variantes", paths))

    # Matrice de confusion
    mat = matrice_confusion_normalisee(y_true, y_pred, label_encoder=label_encoder)
    ax = tracer_matrice_confusion(mat, titre=f"{model_name} - Confusion Matrix")
    generated.append(_save(ax.figure, "matrice_confusion", paths))

    # Top confusions
    top_conf = extraire_top_confusions(y_true, y_pred, label_encoder=label_encoder)
    if not top_conf.empty:
        ax = tracer_top_confusions(top_conf, titre=f"{model_name} - Top Confusions")
        generated.append(_save(ax.figure, "top_confusions", paths))

    # Classes difficiles
    diff = classes_difficiles(y_true, y_pred, label_encoder=label_encoder)
    if not diff.empty:
        ax = tracer_metrique_par_classe(diff, titre=f"{model_name} - Hardest Classes")
        generated.append(_save(ax.figure, "classes_difficiles", paths))

    return generated


def generer_figures_specifiques(
    model_name: str,
    study: StudySpec,
    grid_results: pd.DataFrame | None = None,
) -> FigureBundleResult:
    """Genere les figures specifiques pour un modele."""
    spec = obtenir_model_study_spec(model_name)
    modele_cls = MODELES[model_name]()
    paths = model_paths(model_name, root=study.output_root)
    paths.ensure_dirs()

    X, y, label_encoder = charger_donnees()
    bundle = FigureBundleResult(model_name=model_name)

    # Determiner quelles figures generer
    figures_to_generate = list(spec.core_figures)
    if study.figures_mode == "all":
        figures_to_generate.extend(spec.advanced_figures)
    elif model_name in study.advanced_figures:
        figures_to_generate.extend(study.advanced_figures[model_name])

    if grid_results is None:
        chemin_grid = paths.grid_search_file()
        if chemin_grid.exists():
            grid_results = pd.read_csv(chemin_grid, encoding="utf-8")

    for fig_name in figures_to_generate:
        category = modele_cls.figure_category(fig_name)
        if category is None:
            bundle.skipped.append(fig_name)
            continue

        try:
            fig = _dispatch_figure(
                fig_name,
                model_name,
                X,
                y,
                label_encoder,
                grid_results,
            )
            if fig is not None:
                saved_name = _save(
                    fig,
                    fig_name,
                    paths,
                    category=category,
                    exploratory=fig_name in EXPLORATORY_FIGURES,
                )
                if category == "core":
                    bundle.core_generated.append(saved_name)
                else:
                    bundle.advanced_generated.append(saved_name)
            else:
                bundle.skipped.append(fig_name)
        except Exception as e:
            logger.warning("Figure %s echouee pour %s: %s", fig_name, model_name, e)
            bundle.errors[fig_name] = str(e)

    return bundle


def _dispatch_figure(
    fig_name: str,
    model_name: str,
    X: pd.DataFrame,
    y: Any,
    label_encoder: Any,
    grid_results: pd.DataFrame | None,
) -> plt.Figure | None:
    """Dispatch vers le bon generateur de figure."""
    estimateur = creer_estimateur(model_name)

    # Arbre de decision
    if fig_name == "validation_curve_max_depth" and model_name == "arbre_decision":
        data = validation_curve_max_depth(estimateur, X, y)
        return plots_arbre.tracer_validation_curve_max_depth(data)

    if fig_name == "arbre_tronque" and model_name == "arbre_decision":
        from sklearn.base import clone
        est = clone(estimateur)
        est.fit(X, y)
        class_names = (
            label_encoder.classes_.tolist() if label_encoder is not None else None
        )
        return plots_arbre.tracer_arbre_tronque(
            est, list(X.columns), class_names=class_names
        )

    if fig_name == "learning_curve" and model_name in ("arbre_decision", "svm", "mlp"):
        if model_name == "mlp":
            X_prep, est = _preparer_diagnostic(model_name, X)
            data = compute_learning_curve(est, X_prep, y)
            return plots_mlp.tracer_learning_curve_mlp(data)
        data = compute_learning_curve(estimateur, X, y)
        if model_name == "arbre_decision":
            return plots_arbre.tracer_learning_curve(data)
        return plots_svm.tracer_learning_curve_svm(data)

    # Foret aleatoire
    if fig_name == "oob_vs_n_estimators" and model_name == "foret_aleatoire":
        data = oob_vs_n_estimators(estimateur, X, y)
        return plots_foret.tracer_oob_vs_n_estimators(data)

    if fig_name == "importance_dispersion" and model_name == "foret_aleatoire":
        data = importance_oob_foret(estimateur, X, y)
        return plots_foret.tracer_importance_dispersion(data)

    # MLP
    if fig_name == "loss_curve" and model_name == "mlp":
        X_prep, est = _preparer_diagnostic(model_name, X)
        data = loss_curve_mlp(est, X_prep, y)
        return plots_mlp.tracer_loss_curve(data)

    if fig_name == "heatmap_hidden_alpha" and model_name == "mlp":
        if grid_results is not None and "param_modele__hidden_layer_sizes" in grid_results.columns:
            return plots_mlp.tracer_heatmap_hidden_alpha(grid_results)
        return None

    # Regression logistique
    if fig_name == "regularization_path" and model_name == "regression_logistique":
        X_prep, est = _preparer_diagnostic(model_name, X)
        data = regularization_path(est, X_prep, y)
        return plots_reglog.tracer_regularization_path(data)

    if fig_name == "coefficients" and model_name == "regression_logistique":
        X_prep, est = _preparer_diagnostic(model_name, X)
        data = coefficients_absolus(est, X_prep, y)
        return plots_reglog.tracer_coefficients_absolus(data)

    # SVM
    if fig_name == "heatmap_C_gamma" and model_name == "svm":
        if grid_results is not None and "param_modele__C" in grid_results.columns:
            spec = obtenir_model_study_spec(model_name)
            return plots_svm.tracer_heatmap_C_gamma(
                grid_results, facet_column=spec.tuning_facet_column
            )
        return None

    if fig_name == "support_vectors" and model_name == "svm":
        X_prep, est = _preparer_diagnostic(model_name, X)
        info = analyse_support_vectors(est, X_prep, y)
        return plots_svm.tracer_support_vectors_par_classe(info)

    # Perceptron
    if fig_name == "decision_function_histogram" and model_name == "perceptron":
        X_prep, est = _preparer_diagnostic(model_name, X)
        data = decision_function_histogram(est, X_prep, y)
        return plots_perceptron.tracer_decision_function_histogram(data)

    if fig_name == "coefficients" and model_name == "perceptron":
        X_prep, est = _preparer_diagnostic(model_name, X)
        data = coefficients_absolus(est, X_prep, y)
        return plots_perceptron.tracer_coefficients_perceptron(data)

    return None
