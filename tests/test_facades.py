"""Tests de non-regression: les imports publics du projet fonctionnent."""

import unittest


class TestImportsPackages(unittest.TestCase):
    """Verifie que les imports depuis les vrais packages sont accessibles."""

    def test_import_data(self) -> None:
        from src.data import charger_donnees
        self.assertTrue(callable(charger_donnees))

    def test_import_models(self) -> None:
        from src.models import MODELES
        from src.models.pipelines import creer_estimateur, creer_pipeline, obtenir_grille_tuning
        self.assertTrue(len(MODELES) >= 6)
        self.assertTrue(callable(creer_estimateur))
        self.assertTrue(callable(creer_pipeline))
        self.assertTrue(callable(obtenir_grille_tuning))

    def test_import_evaluation(self) -> None:
        from src.evaluation import (
            CHANCE_LEVEL,
            calculer_metriques_par_classe,
            construire_matrice_confusion,
            evaluer_modele_cv,
            evaluer_modele_tuned_nested_cv,
            extraire_top_confusions,
            predictions_oof,
            probabilites_oof,
            resumer_metriques_par_classe,
        )
        self.assertIsInstance(CHANCE_LEVEL, float)
        self.assertTrue(callable(evaluer_modele_cv))
        self.assertTrue(callable(evaluer_modele_tuned_nested_cv))

    def test_import_artifacts(self) -> None:
        from src.artifacts import (
            charger_tableau_mesures,
            sauvegarder_mesures,
            sauvegarder_predictions,
            sauvegarder_resultats_grid_search,
        )
        self.assertTrue(callable(sauvegarder_mesures))
        self.assertTrue(callable(charger_tableau_mesures))

    def test_import_plots(self) -> None:
        from src.plots import (
            tracer_comparaison_variantes,
            tracer_heatmap_tuning,
            tracer_matrice_confusion,
            tracer_metrique_par_classe,
            tracer_top_confusions,
        )
        self.assertTrue(callable(tracer_matrice_confusion))

    def test_import_config(self) -> None:
        from src.config import N_SPLITS, RANDOM_SEED
        self.assertIsInstance(N_SPLITS, int)
        self.assertIsInstance(RANDOM_SEED, int)

    def test_creer_estimateur(self) -> None:
        """Verifie que creer_estimateur retourne un estimateur valide."""
        from sklearn.base import BaseEstimator
        from src.models.pipelines import creer_estimateur
        estimateur = creer_estimateur("perceptron")
        self.assertIsInstance(estimateur, BaseEstimator)

    def test_creer_pipeline(self) -> None:
        """Verifie que creer_pipeline retourne un pipeline valide."""
        from sklearn.pipeline import Pipeline
        from src.models.pipelines import creer_pipeline
        pipeline = creer_pipeline("perceptron", utiliser_pca=True)
        self.assertIsInstance(pipeline, Pipeline)

    def test_pipeline_honore_default_modele(self) -> None:
        """Verifie que l'absence d'override respecte le defaut du modele."""
        from src.models.pipelines import creer_pipeline

        pipeline_perceptron = creer_pipeline("perceptron")
        self.assertIn("pca", pipeline_perceptron.named_steps)

        pipeline_arbre = creer_pipeline("arbre_decision")
        self.assertNotIn("pca", pipeline_arbre.named_steps)

    def test_pipeline_override_false(self) -> None:
        """Verifie qu'un override False supprime le PCA meme si le modele le recommande."""
        from src.models.pipelines import creer_pipeline

        pipeline = creer_pipeline("perceptron", utiliser_pca=False)
        self.assertNotIn("pca", pipeline.named_steps)

    def test_obtenir_grille_sans_prefixe(self) -> None:
        """Verifie que prefixer_modele=False retire le prefixe."""
        from src.models.pipelines import obtenir_grille_tuning
        grille = obtenir_grille_tuning("perceptron", prefixer_modele=False)
        for cle in grille:
            self.assertFalse(cle.startswith("modele__"), f"Cle '{cle}' ne devrait pas avoir de prefixe")

if __name__ == "__main__":
    unittest.main()
