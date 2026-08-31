"""Tests pour les sanity checks (Regle 5)."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.evaluation.sanity_checks import (
    verifier_bruit_aleatoire,
    verifier_overfit_petit_echantillon,
)
from src.evaluation import evaluer_dummy_classifier


class TestDummyClassifierBaseline(unittest.TestCase):
    def test_dummy_near_chance(self) -> None:
        rng = np.random.RandomState(42)
        X_fake = rng.rand(200, 10)
        y_fake = np.tile(np.arange(10), 20)
        result = evaluer_dummy_classifier(
            X_fake, y_fake, strategie="most_frequent", n_splits=3,
        )
        self.assertLess(result["val_accuracy_mean"], 0.20)


class TestBruitAleatoire(unittest.TestCase):
    def test_arbre_sur_bruit_proche_du_hasard(self) -> None:
        y = np.tile(np.arange(5), 40)
        result = verifier_bruit_aleatoire(
            DecisionTreeClassifier(random_state=42),
            y,
            n_features=20,
        )
        self.assertLess(result["val_accuracy_mean"], result["chance_level"] + 0.10)
        self.assertAlmostEqual(result["chance_level"], 0.2)


class TestOverfitPetitEchantillon(unittest.TestCase):
    def test_mlp_overfit_micro_batch(self) -> None:
        result = verifier_overfit_petit_echantillon(n_samples=10, max_iter=2000)
        self.assertTrue(result["overfit_ok"])
        self.assertGreaterEqual(result["train_accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
