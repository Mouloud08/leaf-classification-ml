"""Tests pour les diagnostics generiques et specifiques."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.diagnostics_generiques import (
    classes_difficiles,
    comparaison_baseline_best_tuned,
    matrice_confusion_normalisee,
    tableau_gaps,
    tableau_variantes,
    top_confusions,
)
from src.models import MODELES


class TestDiagnosticsGeneriques(unittest.TestCase):
    def test_tableau_variantes(self) -> None:
        untuned = [
            {
                "variante": "default",
                "selection_protocol": "simple_cv",
                "evaluation_stage": "baseline",
                "val_accuracy_mean": 0.5,
                "val_accuracy_std": 0.1,
                "val_f1_macro_mean": 0.4,
                "val_f1_macro_std": 0.1,
                "train_accuracy_mean": 0.7,
                "train_f1_macro_mean": 0.6,
                "generalization_gap_accuracy": 0.2,
                "generalization_gap_f1_macro": 0.2,
                "elapsed_seconds": 1.0,
                "timing_policy": "cv_n_jobs=1",
            },
        ]
        tuned = {
            "selection_protocol": "nested_cv",
            "evaluation_stage": "tuned",
            "val_accuracy_mean": 0.8,
            "val_accuracy_std": 0.05,
            "val_f1_macro_mean": 0.75,
            "val_f1_macro_std": 0.04,
            "train_accuracy_mean": float("nan"),
            "train_f1_macro_mean": float("nan"),
            "generalization_gap_accuracy": float("nan"),
            "generalization_gap_f1_macro": float("nan"),
            "elapsed_seconds": 2.0,
            "timing_policy": "cv_n_jobs=1",
        }
        tab = tableau_variantes(untuned, tuned)
        self.assertEqual(len(tab), 2)
        self.assertEqual(tab.iloc[0]["variante"], "tuned")

    def test_tableau_gaps(self) -> None:
        untuned = [
            {
                "variante": "default",
                "generalization_gap_accuracy": 0.2,
                "generalization_gap_f1_macro": 0.15,
            }
        ]
        gaps = tableau_gaps(untuned)
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps.iloc[0]["generalization_gap_accuracy"], 0.2)

    def test_top_confusions(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 0])
        result = top_confusions(y_true, y_pred)
        self.assertIn("classe_vraie", result.columns)
        self.assertTrue(len(result) > 0)

    def test_classes_difficiles(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        result = classes_difficiles(y_true, y_pred, top_n=3)
        self.assertTrue(len(result) <= 3)

    def test_matrice_confusion_normalisee(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        mat = matrice_confusion_normalisee(y_true, y_pred)
        self.assertEqual(mat.shape, (3, 3))
        self.assertTrue(np.allclose(mat.sum(axis=1).to_numpy(), np.ones(3)))

    def test_comparaison_baseline_best_tuned(self) -> None:
        untuned = [
            {
                "variante": "default",
                "evaluation_stage": "baseline",
                "val_f1_macro_mean": 0.4,
                "val_f1_macro_std": 0.1,
                "val_accuracy_mean": 0.5,
                "elapsed_seconds": 1.0,
            },
            {
                "variante": "scaled_only",
                "evaluation_stage": "improved_untuned",
                "val_f1_macro_mean": 0.6,
                "val_f1_macro_std": 0.08,
                "val_accuracy_mean": 0.7,
                "elapsed_seconds": 1.5,
            },
        ]
        tuned = {
            "val_f1_macro_mean": 0.75,
            "val_f1_macro_std": 0.04,
            "val_accuracy_mean": 0.8,
            "elapsed_seconds": 5.0,
        }
        result = comparaison_baseline_best_tuned(untuned, tuned)
        self.assertEqual(len(result), 3)


class TestModelDiagnosticCapabilities(unittest.TestCase):
    def test_all_models_have_diagnostics(self) -> None:
        for name, cls in MODELES.items():
            with self.subTest(model=name):
                modele = cls()
                self.assertTrue(len(modele.supported_diagnostics) > 0)
                self.assertTrue(len(modele.core_figures) > 0)

    def test_figure_category(self) -> None:
        from src.models import ModeleMLP
        mlp = ModeleMLP()
        self.assertEqual(mlp.figure_category("loss_curve"), "core")
        self.assertEqual(mlp.figure_category("tsne_representations"), "advanced")
        self.assertIsNone(mlp.figure_category("nonexistent"))

    def test_supports_diagnostic(self) -> None:
        from src.models import ModeleSVM
        svm = ModeleSVM()
        self.assertTrue(svm.supports_diagnostic("heatmap_C_gamma"))
        self.assertFalse(svm.supports_diagnostic("loss_curve"))


if __name__ == "__main__":
    unittest.main()
