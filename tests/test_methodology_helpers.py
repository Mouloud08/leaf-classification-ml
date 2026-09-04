from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.diagnostics_specifiques import importance_oob_foret
from src.evaluation.metriques import calculer_reliability_curve
from src.notebook_support import (
    construire_tableau_complementarite_erreurs,
    construire_tableau_plis_tuned_global,
)


class TestMethodologyHelpers(unittest.TestCase):
    def test_importance_oob_foret_returns_ranked_table(self) -> None:
        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target.to_numpy()
        foret = RandomForestClassifier(
            n_estimators=60,
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=1,
        )

        tableau = importance_oob_foret(foret, X, y, n_repeats=2)

        self.assertEqual(len(tableau), X.shape[1])
        self.assertEqual(
            list(tableau.columns),
            ["feature", "importance_mean", "importance_std"],
        )
        self.assertGreaterEqual(float(tableau.iloc[0]["importance_mean"]), 0.0)
        self.assertTrue(
            tableau["importance_mean"].is_monotonic_decreasing,
            "Le tableau doit rester trie par importance decroissante.",
        )

    def test_reliability_curve_uses_quantile_bins_by_default(self) -> None:
        tableau = pd.DataFrame(
            {
                "y_true": [0, 0, 1, 1, 0, 1],
                "proba__0": [0.91, 0.88, 0.47, 0.45, 0.83, 0.41],
                "proba__1": [0.09, 0.12, 0.53, 0.55, 0.17, 0.59],
            }
        )

        donnees = calculer_reliability_curve(tableau, n_bins=3)

        self.assertGreaterEqual(len(donnees), 2)
        self.assertIn("confiance_moyenne", donnees.columns)
        self.assertIn("precision_empirique", donnees.columns)
        self.assertTrue(donnees["confiance_moyenne"].between(0, 1).all())
        self.assertTrue(donnees["precision_empirique"].between(0, 1).all())

    def test_construire_tableau_plis_tuned_global(self) -> None:
        X = pd.DataFrame({"x1": range(10), "x2": range(10, 20)})
        y = np.array([0, 1] * 5)
        predictions = {
            "modele_a": pd.DataFrame({"y_true": y, "y_pred": y}),
            "modele_b": pd.DataFrame({"y_true": y, "y_pred": np.array([0] * 10)}),
        }

        tableau = construire_tableau_plis_tuned_global(
            X,
            y,
            predictions,
            n_splits=5,
            random_state=42,
        )

        self.assertEqual(set(tableau["modele"]), {"modele_a", "modele_b"})
        self.assertEqual(set(tableau["fold"]), {1, 2, 3, 4, 5})
        self.assertIn("val_f1_macro", tableau.columns)
        self.assertIn("val_accuracy", tableau.columns)

    def test_construire_tableau_complementarite_erreurs(self) -> None:
        predictions = {
            "modele_a": pd.DataFrame({"y_true": [0, 1, 0, 1], "y_pred": [0, 1, 1, 1]}),
            "modele_b": pd.DataFrame({"y_true": [0, 1, 0, 1], "y_pred": [0, 0, 0, 1]}),
            "modele_c": pd.DataFrame({"y_true": [0, 1, 0, 1], "y_pred": [1, 1, 0, 1]}),
        }

        tableau = construire_tableau_complementarite_erreurs(predictions)

        self.assertEqual(len(tableau), 3)
        self.assertIn("disagreement_rate", tableau.columns)
        self.assertIn("error_overlap_rate", tableau.columns)
        self.assertIn("corriges_par_a_seul", tableau.columns)
        self.assertIn("corriges_par_b_seul", tableau.columns)


if __name__ == "__main__":
    unittest.main()
