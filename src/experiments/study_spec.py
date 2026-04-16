"""Spécification globale d'une étude et d'un modèle dans l'étude."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import (
    EXPLORATORY_TUNING_SPLITS,
    INNER_TUNING_SPLITS,
    N_SPLITS,
    PRIMARY_METRIC,
    RANDOM_SEED,
    RESULTS_DIR,
)


@dataclass(frozen=True)
class UntunedVariantSpec:
    """Description d'une variante non ajustée."""

    name: str
    kind: str  # "simple", "pipeline_no_pca", "pipeline_with_pca"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelStudySpec:
    """Définition exécutable d'un modèle dans l'étude."""

    model_name: str
    title: str
    objective: str
    untuned_variants: tuple[UntunedVariantSpec, ...]
    tuning_uses_pipeline: bool
    supports_probabilities: bool = False
    secondary_holdout_enabled: bool = True
    tuning_heatmap_axes: tuple[str, str] | None = None
    tuning_facet_column: str | None = None
    core_figures: tuple[str, ...] = ()
    advanced_figures: tuple[str, ...] = ()


@dataclass(frozen=True)
class StudySpec:
    """Paramètres globaux de l'étude."""

    models: list[str]
    output_root: Path = RESULTS_DIR
    n_splits: int = N_SPLITS
    inner_tuning_splits: int = INNER_TUNING_SPLITS
    exploratory_tuning_splits: int = EXPLORATORY_TUNING_SPLITS
    random_seed: int = RANDOM_SEED
    primary_metric: str = PRIMARY_METRIC
    figures_mode: str = "core"  # "core", "all", "none"
    advanced_figures: dict[str, list[str]] = field(default_factory=dict)
    skip_existing: bool = False
    force: bool = False
    no_tuning: bool = False
    no_figures: bool = False
