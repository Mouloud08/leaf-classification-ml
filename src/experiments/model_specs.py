"""Table explicite des specs d'etude pour les 6 modeles."""

from __future__ import annotations

from ..models import (
    ModeleArbreDecision,
    ModeleForetAleatoire,
    ModeleMLP,
    ModelePerceptron,
    ModeleRegressionLogistique,
    ModeleSVM,
)
from .study_spec import ModelStudySpec, UntunedVariantSpec


def _spec_perceptron() -> ModelStudySpec:
    m = ModelePerceptron()
    return ModelStudySpec(
        model_name="perceptron",
        title="Perceptron",
        objective="Mesurer le plafond d'une baseline lineaire minimale.",
        untuned_variants=(
            UntunedVariantSpec(name="default", kind="simple"),
            UntunedVariantSpec(name="scaled_only", kind="pipeline_no_pca"),
            UntunedVariantSpec(name="scaled_pca", kind="pipeline_with_pca"),
        ),
        tuning_uses_pipeline=True,
        supports_probabilities=m.supports_probabilities,
        tuning_heatmap_axes=("param_modele__alpha", "param_modele__penalty"),
        tuning_facet_column="param_pca",
        core_figures=m.core_figures,
        advanced_figures=m.advanced_figures,
    )


def _spec_regression_logistique() -> ModelStudySpec:
    m = ModeleRegressionLogistique()
    return ModelStudySpec(
        model_name="regression_logistique",
        title="Regression Logistique",
        objective="Etablir la baseline probabiliste principale.",
        untuned_variants=(
            UntunedVariantSpec(name="default", kind="simple"),
            UntunedVariantSpec(name="scaled_only", kind="pipeline_no_pca"),
            UntunedVariantSpec(name="scaled_pca", kind="pipeline_with_pca"),
        ),
        tuning_uses_pipeline=True,
        supports_probabilities=m.supports_probabilities,
        tuning_heatmap_axes=("param_modele__C", "param_pca"),
        core_figures=m.core_figures,
        advanced_figures=m.advanced_figures,
    )


def _spec_svm() -> ModelStudySpec:
    m = ModeleSVM()
    return ModelStudySpec(
        model_name="svm",
        title="SVM RBF",
        objective="Mesurer la valeur ajoutee d'un noyau RBF.",
        untuned_variants=(
            UntunedVariantSpec(name="default", kind="simple"),
            UntunedVariantSpec(name="scaled_only", kind="pipeline_no_pca"),
            UntunedVariantSpec(name="scaled_pca", kind="pipeline_with_pca"),
        ),
        tuning_uses_pipeline=True,
        supports_probabilities=m.supports_probabilities,
        tuning_heatmap_axes=("param_modele__C", "param_modele__gamma"),
        tuning_facet_column="param_pca",
        core_figures=m.core_figures,
        advanced_figures=m.advanced_figures,
    )


def _spec_foret_aleatoire() -> ModelStudySpec:
    m = ModeleForetAleatoire()
    return ModelStudySpec(
        model_name="foret_aleatoire",
        title="Foret Aleatoire",
        objective="Quantifier la robustesse d'un ensemble d'arbres sans preprocessing.",
        untuned_variants=(
            UntunedVariantSpec(name="default", kind="simple"),
            UntunedVariantSpec(
                name="restricted_depth",
                kind="simple",
                params={"max_depth": 10},
            ),
        ),
        tuning_uses_pipeline=False,
        supports_probabilities=m.supports_probabilities,
        tuning_heatmap_axes=(
            "param_modele__n_estimators",
            "param_modele__max_depth",
        ),
        tuning_facet_column="param_modele__min_samples_leaf",
        core_figures=m.core_figures,
        advanced_figures=m.advanced_figures,
    )


def _spec_arbre_decision() -> ModelStudySpec:
    m = ModeleArbreDecision()
    return ModelStudySpec(
        model_name="arbre_decision",
        title="Arbre de Decision",
        objective="Documenter le compromis performance/complexite d'un arbre unique.",
        untuned_variants=(
            UntunedVariantSpec(name="default", kind="simple"),
            UntunedVariantSpec(
                name="restricted_depth",
                kind="simple",
                params={"max_depth": 10},
            ),
        ),
        tuning_uses_pipeline=False,
        supports_probabilities=m.supports_probabilities,
        secondary_holdout_enabled=False,
        tuning_heatmap_axes=(
            "param_modele__max_depth",
            "param_modele__min_samples_leaf",
        ),
        tuning_facet_column="param_modele__ccp_alpha",
        core_figures=m.core_figures,
        advanced_figures=m.advanced_figures,
    )


def _spec_mlp() -> ModelStudySpec:
    m = ModeleMLP()
    return ModelStudySpec(
        model_name="mlp",
        title="MLP",
        objective="Tester la sensibilite d'un reseau dense au preprocessing et au tuning.",
        untuned_variants=(
            UntunedVariantSpec(name="default", kind="simple"),
            UntunedVariantSpec(name="scaled_only", kind="pipeline_no_pca"),
            UntunedVariantSpec(name="scaled_pca", kind="pipeline_with_pca"),
        ),
        tuning_uses_pipeline=True,
        supports_probabilities=m.supports_probabilities,
        tuning_heatmap_axes=(
            "param_modele__hidden_layer_sizes",
            "param_modele__alpha",
        ),
        tuning_facet_column="param_pca",
        core_figures=m.core_figures,
        advanced_figures=m.advanced_figures,
    )


MODEL_STUDY_SPECS: dict[str, ModelStudySpec] = {
    "perceptron": _spec_perceptron(),
    "regression_logistique": _spec_regression_logistique(),
    "svm": _spec_svm(),
    "foret_aleatoire": _spec_foret_aleatoire(),
    "arbre_decision": _spec_arbre_decision(),
    "mlp": _spec_mlp(),
}

ALL_MODEL_NAMES: list[str] = list(MODEL_STUDY_SPECS.keys())


def obtenir_model_study_spec(model_name: str) -> ModelStudySpec:
    """Retourne la spec d'etude pour un modele."""
    key = model_name.lower()
    if key not in MODEL_STUDY_SPECS:
        raise KeyError(f"Spec d'etude inconnue: {model_name}")
    return MODEL_STUDY_SPECS[key]
