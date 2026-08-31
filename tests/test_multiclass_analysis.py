import unittest

import matplotlib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")

from src.evaluation import (  # noqa: E402
    calculer_reliability_curve,
    calculer_points_precision_rappel_ovr,
    calculer_points_roc_ovr,
    calculer_top_k_accuracies,
    calculer_metriques_par_classe,
    construire_matrice_confusion,
    evaluer_modele_tuned_nested_cv,
    probabilites_oof,
)
from src.evaluation.diagnostics_specifiques import coefficients_absolus, regularization_path  # noqa: E402
from src.models.pipelines import creer_pipeline  # noqa: E402
from src.plots import (  # noqa: E402
    tracer_hyperparametre_vs_performance,
    tracer_matrice_confusion,
    tracer_metrique_par_classe,
)


class TestMulticlassAnalysis(unittest.TestCase):
    def test_calculer_metriques_par_classe(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])

        tableau = calculer_metriques_par_classe(y_true, y_pred)
        classe_0 = tableau.loc[tableau["classe"] == 0].iloc[0]

        self.assertAlmostEqual(classe_0["precision"], 0.5)
        self.assertAlmostEqual(classe_0["recall"], 0.5)
        self.assertAlmostEqual(classe_0["f1"], 0.5)
        self.assertEqual(classe_0["support"], 2)
        self.assertAlmostEqual(classe_0["fpr"], 0.25)
        self.assertAlmostEqual(classe_0["fnr"], 0.5)
        self.assertAlmostEqual(classe_0["specificite"], 0.75)

    def test_construire_matrice_confusion_normalisee(self) -> None:
        encodeur = LabelEncoder().fit(["a", "b", "c"])
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])

        matrice = construire_matrice_confusion(
            y_true,
            y_pred,
            label_encoder=encodeur,
            normalize="true",
        )

        self.assertEqual(matrice.shape, (3, 3))
        self.assertEqual(list(matrice.index), ["a", "b", "c"])
        self.assertTrue(np.allclose(matrice.sum(axis=1).to_numpy(), np.ones(3)))
        self.assertAlmostEqual(float(matrice.loc["b", "b"]), 1.0)

    def test_probabilites_oof(self) -> None:
        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target.to_numpy()
        noms = np.array(["setosa", "versicolor", "virginica"])
        encodeur = LabelEncoder().fit(noms)

        modele = LogisticRegression(max_iter=500)
        tableau = probabilites_oof(
            modele,
            X,
            y,
            label_encoder=encodeur,
            n_splits=3,
            random_state=42,
        )

        colonnes_proba = [colonne for colonne in tableau.columns if colonne.startswith("proba__")]
        self.assertEqual(len(colonnes_proba), 3)
        self.assertTrue(np.allclose(tableau[colonnes_proba].sum(axis=1).to_numpy(), np.ones(len(tableau))))
        argmax = tableau[colonnes_proba].to_numpy().argmax(axis=1)
        self.assertTrue(np.array_equal(argmax, tableau["y_pred"].to_numpy()))

    def test_probabilite_metrics_accept_string_labels(self) -> None:
        tableau = pd.DataFrame(
            {
                "y_true": ["a", "b", "c", "a", "b", "c"],
                "proba__a": [0.9, 0.1, 0.0, 0.8, 0.2, 0.1],
                "proba__b": [0.1, 0.8, 0.2, 0.1, 0.7, 0.2],
                "proba__c": [0.0, 0.1, 0.8, 0.1, 0.1, 0.7],
            }
        )

        top_k = calculer_top_k_accuracies(tableau, ks=(1, 2))
        roc = calculer_points_roc_ovr(tableau)
        pr = calculer_points_precision_rappel_ovr(tableau)

        self.assertAlmostEqual(top_k["top_1_accuracy"], 1.0)
        self.assertIn("auc_micro", roc)
        self.assertIn("ap_micro", pr)

    def test_reliability_curve_quantile_strategy_handles_compressed_confidence(self) -> None:
        tableau = pd.DataFrame(
            {
                "y_true": ["a"] * 6 + ["b"] * 6,
                "y_pred": ["a", "a", "a", "a", "b", "b", "b", "b", "b", "a", "b", "a"],
                "proba__a": [0.11, 0.12, 0.12, 0.13, 0.10, 0.09, 0.08, 0.08, 0.07, 0.11, 0.06, 0.10],
                "proba__b": [0.89, 0.88, 0.88, 0.87, 0.90, 0.91, 0.92, 0.92, 0.93, 0.89, 0.94, 0.90],
            }
        )

        donnees = calculer_reliability_curve(tableau, n_bins=4)

        self.assertGreaterEqual(len(donnees), 3)
        self.assertTrue(donnees["confiance_moyenne"].between(0, 1).all())
        self.assertTrue(donnees["precision_empirique"].between(0, 1).all())

    def test_reliability_curve_rejects_invalid_strategy(self) -> None:
        tableau = pd.DataFrame(
            {
                "y_true": ["a", "b"],
                "proba__a": [0.7, 0.2],
                "proba__b": [0.3, 0.8],
            }
        )

        with self.assertRaises(ValueError):
            calculer_reliability_curve(tableau, strategy="invalid")

    def test_evaluer_modele_tuned_nested_cv(self) -> None:
        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target.to_numpy()
        modele = LogisticRegression(max_iter=500)

        resultat = evaluer_modele_tuned_nested_cv(
            estimateur_ou_pipeline=modele,
            param_grid={"C": [0.1, 1.0]},
            X=X,
            y=y,
            outer_splits=3,
            inner_splits=2,
            random_state=42,
            return_probabilities=True,
            n_jobs=1,
        )

        mesures = resultat["metrics"]
        self.assertEqual(mesures["selection_protocol"], "nested_cv")
        self.assertEqual(mesures["n_splits"], 3)
        self.assertEqual(mesures["inner_splits"], 2)
        self.assertEqual(len(resultat["oof_predictions"]), len(X))
        self.assertEqual(len(resultat["outer_fold_metrics"]), 3)
        self.assertEqual(len(resultat["best_params_per_fold"]), 3)
        self.assertIsNotNone(resultat["oof_probabilities"])

    def test_nested_cv_returns_nan_for_train_and_gap(self) -> None:
        """Train scores and generalization gaps must be NaN for nested CV results."""
        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target.to_numpy()
        modele = LogisticRegression(max_iter=500)

        resultat = evaluer_modele_tuned_nested_cv(
            estimateur_ou_pipeline=modele,
            param_grid={"C": [0.1, 1.0]},
            X=X,
            y=y,
            outer_splits=3,
            inner_splits=2,
            random_state=42,
            n_jobs=1,
        )

        mesures = resultat["metrics"]
        nan_fields = [
            "train_accuracy_mean",
            "train_f1_macro_mean",
            "generalization_gap_accuracy",
            "generalization_gap_f1_macro",
        ]
        for champ in nan_fields:
            with self.subTest(field=champ):
                self.assertIn(champ, mesures)
                self.assertTrue(
                    np.isnan(mesures[champ]),
                    f"{champ} should be NaN but got {mesures[champ]}",
                )

    def test_regularization_path_supports_pipeline_estimators(self) -> None:
        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target.to_numpy()

        donnees = regularization_path(
            creer_pipeline("regression_logistique", utiliser_pca=False),
            X,
            y,
            C_values=[0.1, 1.0],
        )

        self.assertEqual(list(donnees["C"]), [0.1, 1.0])
        self.assertIn("val_f1_macro_mean", donnees.columns)
        self.assertIn("train_f1_macro_mean", donnees.columns)

    def test_coefficients_absolus_supports_pipeline_estimators(self) -> None:
        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target.to_numpy()

        donnees = coefficients_absolus(
            creer_pipeline("regression_logistique", utiliser_pca=False),
            X,
            y,
            top_n=3,
        )

        self.assertEqual(len(donnees), 3)
        self.assertEqual(list(donnees.columns), ["feature", "coef_abs_mean"])
        self.assertTrue((donnees["coef_abs_mean"] >= 0).all())

    def test_heatmap_tuning_rejects_duplicates_by_default(self) -> None:
        """tracer_heatmap_tuning must raise on duplicate (x, y) pairs by default."""
        from src.plots import tracer_heatmap_tuning

        tableau_doublons = pd.DataFrame({
            "param_C": [0.1, 0.1, 1.0],
            "param_gamma": [0.01, 0.01, 0.1],
            "mean_test_f1_macro": [0.7, 0.8, 0.9],
        })
        with self.assertRaises(ValueError, msg="Duplicate x/y should raise"):
            tracer_heatmap_tuning(
                tableau_doublons,
                x="param_C",
                y="param_gamma",
                valeur="mean_test_f1_macro",
            )

    def test_visualisation_smoke(self) -> None:
        matrice = pd.DataFrame(
            [[0.8, 0.2], [0.1, 0.9]],
            index=["a", "b"],
            columns=["a", "b"],
        )
        tableau_classes = pd.DataFrame(
            {
                "classe": ["a", "b"],
                "recall": [0.8, 0.6],
                "f1": [0.75, 0.55],
                "fpr": [0.2, 0.1],
            }
        )
        tableau_grid = pd.DataFrame(
            {
                "param_n_estimators": [50, 100, 200],
                "mean_test_f1_macro": [0.6, 0.7, 0.72],
                "param_learning_rate": [0.1, 0.1, 0.1],
            }
        )

        axe_matrice = tracer_matrice_confusion(matrice)
        axe_classes = tracer_metrique_par_classe(tableau_classes, metrique="recall", top_n=2)
        axe_hyper = tracer_hyperparametre_vs_performance(
            tableau_grid,
            x="param_n_estimators",
            hue="param_learning_rate",
        )

        self.assertIsNotNone(axe_matrice)
        self.assertIsNotNone(axe_classes)
        self.assertIsNotNone(axe_hyper)


if __name__ == "__main__":
    unittest.main()
