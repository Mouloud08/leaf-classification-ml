"""Dataclasses typees pour tous les artefacts du projet."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UntunedVariantResult:
    """Résultat d'une variante non ajustée."""

    variante: str
    selection_protocol: str
    evaluation_stage: str
    val_accuracy_mean: float
    val_accuracy_std: float
    val_f1_macro_mean: float
    val_f1_macro_std: float
    train_accuracy_mean: float
    train_f1_macro_mean: float
    generalization_gap_accuracy: float
    generalization_gap_f1_macro: float
    elapsed_seconds: float
    timing_policy: str
    n_splits: int
    fit_time_mean: float = 0.0
    score_time_mean: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "variante": self.variante,
            "selection_protocol": self.selection_protocol,
            "evaluation_stage": self.evaluation_stage,
            "val_accuracy_mean": self.val_accuracy_mean,
            "val_accuracy_std": self.val_accuracy_std,
            "val_f1_macro_mean": self.val_f1_macro_mean,
            "val_f1_macro_std": self.val_f1_macro_std,
            "train_accuracy_mean": self.train_accuracy_mean,
            "train_f1_macro_mean": self.train_f1_macro_mean,
            "generalization_gap_accuracy": self.generalization_gap_accuracy,
            "generalization_gap_f1_macro": self.generalization_gap_f1_macro,
            "elapsed_seconds": self.elapsed_seconds,
            "timing_policy": self.timing_policy,
            "n_splits": self.n_splits,
            "fit_time_mean": self.fit_time_mean,
            "score_time_mean": self.score_time_mean,
        }


@dataclass(frozen=True)
class TuningRunResult:
    """Résultat complet du tuning exploratoire.

    Inclut aussi la validation imbriquée confirmatoire.
    """

    val_accuracy_mean: float
    val_accuracy_std: float
    val_f1_macro_mean: float
    val_f1_macro_std: float
    elapsed_seconds: float
    timing_policy: str
    n_splits: int
    inner_splits: int
    selection_protocol: str = "nested_cv"
    evaluation_stage: str = "tuned"
    best_params_exploratory: dict[str, Any] = field(default_factory=dict)
    best_params_per_fold: list[dict[str, Any]] = field(default_factory=list)
    outer_fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    exploratory_tuning_seconds: float = 0.0
    fit_time_mean: float = 0.0
    score_time_mean: float = 0.0
    secondary_holdout: dict[str, Any] = field(default_factory=dict)
    descriptive_holdout_all_variants: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "variante": "tuned",
            "selection_protocol": self.selection_protocol,
            "evaluation_stage": self.evaluation_stage,
            "val_accuracy_mean": self.val_accuracy_mean,
            "val_accuracy_std": self.val_accuracy_std,
            "val_f1_macro_mean": self.val_f1_macro_mean,
            "val_f1_macro_std": self.val_f1_macro_std,
            "train_accuracy_mean": float("nan"),
            "train_f1_macro_mean": float("nan"),
            "generalization_gap_accuracy": float("nan"),
            "generalization_gap_f1_macro": float("nan"),
            "elapsed_seconds": self.elapsed_seconds,
            "timing_policy": self.timing_policy,
            "n_splits": self.n_splits,
            "inner_splits": self.inner_splits,
            "best_params_exploratory": self.best_params_exploratory,
            "best_params_per_fold": self.best_params_per_fold,
            "outer_fold_metrics": self.outer_fold_metrics,
            "exploratory_tuning_seconds": self.exploratory_tuning_seconds,
            "fit_time_mean": self.fit_time_mean,
            "score_time_mean": self.score_time_mean,
        }
        if self.secondary_holdout:
            result["secondary_holdout"] = self.secondary_holdout
        if self.descriptive_holdout_all_variants:
            result["descriptive_holdout_all_variants"] = (
                self.descriptive_holdout_all_variants
            )
        return result


@dataclass(frozen=True)
class FigureRequest:
    """Description d'une figure a generer."""

    name: str
    category: str  # "core" ou "advanced"
    generator: str  # nom de la fonction de generation
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class FigureBundleResult:
    """Bilan de generation de figures pour un modele."""

    model_name: str
    core_generated: list[str] = field(default_factory=list)
    advanced_generated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class StudyRunResult:
    """Resultat complet de l'etude pour un modele."""

    model_name: str
    untuned_results: list[UntunedVariantResult] = field(default_factory=list)
    tuning_result: TuningRunResult | None = None
    figures: FigureBundleResult | None = None
    success: bool = True
    error: str | None = None


METRICS_SCHEMA_VERSION = 2

REQUIRED_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "modele",
        "variante",
        "schema_version",
        "selection_protocol",
        "evaluation_stage",
        "timing_policy",
        "val_f1_macro_mean",
        "val_f1_macro_std",
        "val_accuracy_mean",
        "val_accuracy_std",
        "elapsed_seconds",
    }
)

VALID_SELECTION_PROTOCOLS: frozenset[str] = frozenset({"simple_cv", "nested_cv"})

VALID_EVALUATION_STAGES: frozenset[str] = frozenset(
    {"baseline", "improved_untuned", "tuned", "autre"}
)

NESTED_CV_INVALID_FIELDS: tuple[str, ...] = (
    "train_accuracy_mean",
    "train_f1_macro_mean",
    "generalization_gap_accuracy",
    "generalization_gap_f1_macro",
)
