"""Tests pour la hierarchie OOP des modeles."""

import unittest

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.models import (
    MODELES,
    ModeleArbreDecision,
    ModeleBase,
    ModeleForetAleatoire,
    ModeleMLP,
    ModelePerceptron,
    ModeleRegressionLogistique,
    ModeleSVM,
)


class TestModeleBase(unittest.TestCase):
    """Verifie que toutes les classes concretes respectent le contrat."""

    def test_modeles_dict_complet(self) -> None:
        self.assertEqual(len(MODELES), 6)
        noms_attendus = {
            "perceptron",
            "regression_logistique",
            "svm",
            "foret_aleatoire",
            "arbre_decision",
            "mlp",
        }
        self.assertEqual(set(MODELES.keys()), noms_attendus)

    def test_toutes_classes_heritent_de_modele_base(self) -> None:
        for nom, cls in MODELES.items():
            with self.subTest(modele=nom):
                self.assertTrue(
                    issubclass(cls, ModeleBase),
                    f"{cls.__name__} doit heriter de ModeleBase",
                )

    def test_creer_estimateur(self) -> None:
        for nom, cls in MODELES.items():
            with self.subTest(modele=nom):
                modele = cls()
                estimateur = modele.creer_estimateur()
                self.assertIsInstance(estimateur, BaseEstimator)

    def test_creer_pipeline(self) -> None:
        for nom, cls in MODELES.items():
            with self.subTest(modele=nom):
                modele = cls()
                pipeline = modele.creer_pipeline()
                self.assertIsInstance(pipeline, Pipeline)
                # Le dernier step doit etre 'modele'
                self.assertEqual(pipeline.steps[-1][0], "modele")

    def test_pipeline_scaling_pour_modeles_lineaires(self) -> None:
        modeles_avec_scaling = [
            ModelePerceptron,
            ModeleRegressionLogistique,
            ModeleSVM,
            ModeleMLP,
        ]
        for cls in modeles_avec_scaling:
            with self.subTest(modele=cls.__name__):
                modele = cls()
                pipeline = modele.creer_pipeline(utiliser_pca=False)
                noms_steps = [nom for nom, _ in pipeline.steps]
                self.assertIn("scaler", noms_steps)

    def test_pipeline_pas_de_scaling_pour_arbres(self) -> None:
        modeles_sans_scaling = [ModeleForetAleatoire, ModeleArbreDecision]
        for cls in modeles_sans_scaling:
            with self.subTest(modele=cls.__name__):
                modele = cls()
                pipeline = modele.creer_pipeline()
                noms_steps = [nom for nom, _ in pipeline.steps]
                self.assertNotIn("scaler", noms_steps)

    def test_pipeline_avec_pca(self) -> None:
        modele = ModelePerceptron()
        pipeline = modele.creer_pipeline(utiliser_pca=True)
        noms_steps = [nom for nom, _ in pipeline.steps]
        self.assertIn("pca", noms_steps)

    def test_pipeline_pca_interdit_sans_scaling(self) -> None:
        """Les modeles sans scaling (arbres) doivent lever ValueError si on force PCA."""
        modeles_sans_scaling = [ModeleForetAleatoire, ModeleArbreDecision]
        for cls in modeles_sans_scaling:
            with self.subTest(modele=cls.__name__):
                modele = cls()
                with self.assertRaises(ValueError, msg="PCA sans scaling doit lever ValueError"):
                    modele.creer_pipeline(utiliser_pca=True)

    def test_modeles_avec_pca_exposent_hyperparametre_pca(self) -> None:
        modeles_avec_pca = [
            ModelePerceptron,
            ModeleRegressionLogistique,
            ModeleSVM,
            ModeleMLP,
        ]
        for cls in modeles_avec_pca:
            with self.subTest(modele=cls.__name__):
                grille = cls().grille_tuning()
                self.assertIn("pca", grille)

    def test_grille_tuning_format(self) -> None:
        for nom, cls in MODELES.items():
            with self.subTest(modele=nom):
                modele = cls()
                grille = modele.grille_tuning()
                self.assertIsInstance(grille, dict)
                for cle in grille:
                    self.assertIsInstance(cle, str)
                    self.assertTrue(
                        cle.startswith("modele__") or cle == "pca",
                        f"Cle '{cle}' doit commencer par 'modele__' ou etre 'pca'",
                    )

    def test_definition_format(self) -> None:
        cles_attendues = {
            "nom_affiche",
            "famille",
            "description",
            "hypothese",
            "utiliser_pca_par_defaut",
            "grille_tuning",
        }
        for nom, cls in MODELES.items():
            with self.subTest(modele=nom):
                modele = cls()
                definition = modele.definition()
                self.assertEqual(set(definition.keys()), cles_attendues)


if __name__ == "__main__":
    unittest.main()
