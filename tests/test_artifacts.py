"""Tests pour le package artifacts — schema, paths, validation."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from src.artifacts import (
    ModelArtifactPaths,
    GlobalArtifactPaths,
    artifact_integrity_table,
    canonical_artifact_completeness,
    charger_mesures,
    model_paths,
    recompute_metrics_from_predictions,
    sauvegarder_mesures,
    sauvegarder_predictions,
    global_paths,
)
from src.artifacts.schema import (
    FigureBundleResult,
    FigureRequest,
    StudyRunResult,
    TuningRunResult,
    UntunedVariantResult,
)
from src.artifacts.validation import (
    MetricsArtifactValidationError,
    derive_stage,
    derive_selection_protocol,
    normaliser_mesures_pour_schema,
    valider_artefacts_mesures,
)

ROOT = Path(__file__).resolve().parents[1]


def _temp_dir() -> Path:
    p = ROOT / f".tmp_test_{uuid.uuid4().hex}"
    p.mkdir(parents=True, exist_ok=False)
    return p


class TestArtifactPaths(unittest.TestCase):
    def test_model_paths_structure(self) -> None:
        paths = model_paths("perceptron", root=Path("/tmp/test"))
        self.assertEqual(paths.metrics_dir, Path("/tmp/test/perceptron/metrics"))
        self.assertEqual(paths.predictions_dir, Path("/tmp/test/perceptron/predictions"))
        self.assertEqual(paths.tuning_dir, Path("/tmp/test/perceptron/tuning"))
        self.assertEqual(paths.figures_core_dir, Path("/tmp/test/perceptron/figures/core"))
        self.assertEqual(paths.figures_advanced_dir, Path("/tmp/test/perceptron/figures/advanced"))

    def test_model_paths_files(self) -> None:
        paths = model_paths("svm", root=Path("/tmp/test"))
        self.assertEqual(paths.metrics_file("default"), Path("/tmp/test/svm/metrics/default.json"))
        self.assertEqual(
            paths.predictions_file("tuned"),
            Path("/tmp/test/svm/predictions/tuned__predictions.csv"),
        )
        self.assertEqual(paths.grid_search_file(), Path("/tmp/test/svm/tuning/grid_search.csv"))
        self.assertEqual(
            paths.figure_file("confusion", category="core"),
            Path("/tmp/test/svm/figures/core/confusion.png"),
        )
        self.assertEqual(
            paths.figure_file("tsne", category="advanced"),
            Path("/tmp/test/svm/figures/advanced/tsne.png"),
        )

    def test_global_paths_structure(self) -> None:
        paths = global_paths(root=Path("/tmp/test"))
        self.assertEqual(paths.manifest_file, Path("/tmp/test/global/study_manifest.json"))
        self.assertEqual(
            paths.metrics_summary_file, Path("/tmp/test/global/comparison/metrics_summary.csv")
        )

    def test_ensure_dirs(self) -> None:
        tmp = _temp_dir()
        try:
            paths = model_paths("test_model", root=tmp)
            paths.ensure_dirs()
            self.assertTrue(paths.metrics_dir.exists())
            self.assertTrue(paths.predictions_dir.exists())
            self.assertTrue(paths.tuning_dir.exists())
            self.assertTrue(paths.figures_core_dir.exists())
            self.assertTrue(paths.figures_advanced_dir.exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestArtifactSchema(unittest.TestCase):
    def test_untuned_variant_result_roundtrip(self) -> None:
        result = UntunedVariantResult(
            variante="default",
            selection_protocol="simple_cv",
            evaluation_stage="baseline",
            val_accuracy_mean=0.8,
            val_accuracy_std=0.01,
            val_f1_macro_mean=0.7,
            val_f1_macro_std=0.02,
            train_accuracy_mean=0.9,
            train_f1_macro_mean=0.85,
            generalization_gap_accuracy=0.1,
            generalization_gap_f1_macro=0.15,
            elapsed_seconds=1.0,
            timing_policy="cv_n_jobs=1",
            n_splits=5,
        )
        d = result.to_dict()
        self.assertEqual(d["variante"], "default")
        self.assertEqual(d["val_f1_macro_mean"], 0.7)
        self.assertEqual(d["n_splits"], 5)

    def test_tuning_run_result_has_nan(self) -> None:
        result = TuningRunResult(
            val_accuracy_mean=0.9,
            val_accuracy_std=0.01,
            val_f1_macro_mean=0.88,
            val_f1_macro_std=0.02,
            elapsed_seconds=10.0,
            timing_policy="cv_n_jobs=1",
            n_splits=5,
            inner_splits=5,
        )
        d = result.to_dict()
        import math
        self.assertTrue(math.isnan(d["train_accuracy_mean"]))
        self.assertTrue(math.isnan(d["generalization_gap_f1_macro"]))

    def test_study_run_result_default(self) -> None:
        result = StudyRunResult(model_name="test")
        self.assertTrue(result.success)
        self.assertEqual(result.untuned_results, [])
        self.assertIsNone(result.tuning_result)

    def test_figure_bundle_result(self) -> None:
        bundle = FigureBundleResult(model_name="svm")
        bundle.core_generated.append("confusion")
        bundle.errors["tsne"] = "no module"
        self.assertEqual(len(bundle.core_generated), 1)
        self.assertIn("tsne", bundle.errors)


class TestValidation(unittest.TestCase):
    def test_derive_stage(self) -> None:
        self.assertEqual(derive_stage("default"), "baseline")
        self.assertEqual(derive_stage("scaled_only"), "improved_untuned")
        self.assertEqual(derive_stage("tuned"), "tuned")
        self.assertEqual(derive_stage("unknown"), "autre")

    def test_derive_selection_protocol(self) -> None:
        self.assertEqual(derive_selection_protocol("tuned", {}), "nested_cv")
        self.assertEqual(derive_selection_protocol("default", {}), "simple_cv")
        self.assertEqual(
            derive_selection_protocol("default", {"selection_protocol": "nested_cv"}),
            "nested_cv",
        )

    def test_normaliser_mesures(self) -> None:
        mesures = {
            "val_accuracy_mean": 0.8,
            "val_accuracy_std": 0.01,
            "val_f1_macro_mean": 0.7,
            "val_f1_macro_std": 0.02,
            "train_accuracy_mean": 0.9,
            "train_f1_macro_mean": 0.85,
            "generalization_gap_accuracy": 0.1,
            "generalization_gap_f1_macro": 0.15,
            "elapsed_seconds": 1.0,
            "n_splits": 5,
        }
        contenu = normaliser_mesures_pour_schema("perceptron", "default", mesures)
        self.assertEqual(contenu["modele"], "perceptron")
        self.assertEqual(contenu["variante"], "default")
        self.assertEqual(contenu["evaluation_stage"], "baseline")
        self.assertEqual(contenu["selection_protocol"], "simple_cv")

    def test_valider_artefacts_rejects_missing_keys(self) -> None:
        erreurs = valider_artefacts_mesures([{"modele": "x"}])
        self.assertTrue(len(erreurs) > 0)

    def test_valid_artifact_passes(self) -> None:
        artefact = {
            "modele": "svm",
            "variante": "default",
            "schema_version": 2,
            "selection_protocol": "simple_cv",
            "evaluation_stage": "baseline",
            "timing_policy": "cv_n_jobs=1",
            "val_f1_macro_mean": 0.9,
            "val_f1_macro_std": 0.01,
            "val_accuracy_mean": 0.9,
            "val_accuracy_std": 0.01,
            "elapsed_seconds": 1.0,
            "n_splits": 5,
        }
        erreurs = valider_artefacts_mesures([artefact])
        self.assertEqual(len(erreurs), 0)


class TestCanonicalArtifactLoading(unittest.TestCase):
    def test_charger_mesures_prefers_canonical_tree(self) -> None:
        tmp = _temp_dir()
        try:
            paths = model_paths("modele_demo", root=tmp)
            sauvegarder_mesures(
                "modele_demo",
                "default",
                {
                    "selection_protocol": "simple_cv",
                    "evaluation_stage": "baseline",
                    "timing_policy": "cv_n_jobs=1",
                    "val_accuracy_mean": 0.8,
                    "val_accuracy_std": 0.01,
                    "val_f1_macro_mean": 0.7,
                    "val_f1_macro_std": 0.02,
                    "train_accuracy_mean": 0.9,
                    "train_f1_macro_mean": 0.85,
                    "generalization_gap_accuracy": 0.1,
                    "generalization_gap_f1_macro": 0.15,
                    "elapsed_seconds": 1.0,
                    "n_splits": 5,
                },
                paths=paths,
            )
            (tmp / "flat_extra.json").write_text("{}", encoding="utf-8")

            records = charger_mesures(metrics_dir=tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["modele"], "modele_demo")
        self.assertEqual(records[0]["variante"], "default")

    def test_recompute_metrics_from_predictions(self) -> None:
        tableau = pd.DataFrame(
            {
                "y_true": [0, 1, 1, 2],
                "y_pred": [0, 1, 0, 2],
            }
        )
        metrics = recompute_metrics_from_predictions(tableau)
        self.assertAlmostEqual(metrics["val_accuracy_mean"], 0.75)
        self.assertAlmostEqual(metrics["val_f1_macro_mean"], 0.7777777777777777)

    def test_canonical_completeness_and_integrity_table(self) -> None:
        tmp = _temp_dir()
        try:
            paths = model_paths("modele_integrite", root=tmp)
            sauvegarder_mesures(
                "modele_integrite",
                "tuned",
                {
                    "selection_protocol": "nested_cv",
                    "evaluation_stage": "tuned",
                    "timing_policy": "cv_n_jobs=1",
                    "val_accuracy_mean": 1.0,
                    "val_accuracy_std": 0.0,
                    "val_f1_macro_mean": 1.0,
                    "val_f1_macro_std": 0.0,
                    "train_accuracy_mean": float("nan"),
                    "train_f1_macro_mean": float("nan"),
                    "generalization_gap_accuracy": float("nan"),
                    "generalization_gap_f1_macro": float("nan"),
                    "elapsed_seconds": 1.0,
                    "n_splits": 5,
                    "inner_splits": 5,
                },
                paths=paths,
            )
            sauvegarder_predictions(
                "modele_integrite",
                "tuned",
                pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1]}),
                paths=paths,
            )

            completeness = canonical_artifact_completeness(
                ["modele_integrite", "modele_manquant"],
                root=tmp,
            )
            integrity = artifact_integrity_table(["modele_integrite"], root=tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        self.assertEqual(
            completeness.loc[completeness["modele"] == "modele_integrite", "canonical_complete"].item(),
            True,
        )
        self.assertEqual(
            completeness.loc[completeness["modele"] == "modele_manquant", "canonical_complete"].item(),
            False,
        )
        self.assertEqual(integrity.iloc[0]["status"], "ok")
        self.assertEqual(integrity.iloc[0]["metrics_match_predictions"], True)


if __name__ == "__main__":
    unittest.main()
