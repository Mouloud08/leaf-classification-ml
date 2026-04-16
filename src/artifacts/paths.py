"""Regles de chemins pour l'arborescence results/<modele>/..."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import RESULTS_DIR


@dataclass(frozen=True)
class ModelArtifactPaths:
    """Chemins d'artefacts pour un modele donne."""

    model_name: str
    root: Path

    @property
    def metrics_dir(self) -> Path:
        return self.root / self.model_name / "metrics"

    @property
    def predictions_dir(self) -> Path:
        return self.root / self.model_name / "predictions"

    @property
    def tuning_dir(self) -> Path:
        return self.root / self.model_name / "tuning"

    @property
    def figures_core_dir(self) -> Path:
        return self.root / self.model_name / "figures" / "core"

    @property
    def figures_advanced_dir(self) -> Path:
        return self.root / self.model_name / "figures" / "advanced"

    def metrics_file(self, variante: str) -> Path:
        return self.metrics_dir / f"{variante}.json"

    def predictions_file(self, variante: str) -> Path:
        return self.predictions_dir / f"{variante}__predictions.csv"

    def probabilities_file(self, variante: str) -> Path:
        return self.predictions_dir / f"{variante}__probabilities.csv"

    def grid_search_file(self) -> Path:
        return self.tuning_dir / "grid_search.csv"

    def figure_file(self, name: str, category: str = "core") -> Path:
        base = (
            self.figures_core_dir
            if category == "core"
            else self.figures_advanced_dir
        )
        return base / f"{name}.png"

    def ensure_dirs(self) -> None:
        """Cree tous les sous-dossiers necessaires."""
        for d in (
            self.metrics_dir,
            self.predictions_dir,
            self.tuning_dir,
            self.figures_core_dir,
            self.figures_advanced_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class GlobalArtifactPaths:
    """Chemins des artefacts globaux de l'etude."""

    root: Path

    @property
    def global_dir(self) -> Path:
        return self.root / "global"

    @property
    def comparison_dir(self) -> Path:
        return self.global_dir / "comparison"

    @property
    def figures_dir(self) -> Path:
        return self.global_dir / "figures"

    @property
    def manifest_file(self) -> Path:
        return self.global_dir / "study_manifest.json"

    @property
    def metrics_summary_file(self) -> Path:
        return self.comparison_dir / "metrics_summary.csv"

    @property
    def ranked_models_file(self) -> Path:
        return self.comparison_dir / "ranked_models.csv"

    @property
    def frozen_holdout_split_file(self) -> Path:
        return self.comparison_dir / "frozen_holdout_split.json"

    def ensure_dirs(self) -> None:
        for d in (self.comparison_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)


def model_paths(model_name: str, root: Path | None = None) -> ModelArtifactPaths:
    """Construit les chemins d'artefacts pour un modele."""
    return ModelArtifactPaths(model_name=model_name, root=root or RESULTS_DIR)


def global_paths(root: Path | None = None) -> GlobalArtifactPaths:
    """Construit les chemins des artefacts globaux."""
    return GlobalArtifactPaths(root=root or RESULTS_DIR)
