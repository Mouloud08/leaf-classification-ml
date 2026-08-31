"""Tests pour le package experiments — specs, runner, CLI."""

from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from src.experiments import (
    ALL_MODEL_NAMES,
    MODEL_STUDY_SPECS,
    ModelStudySpec,
    StudySpec,
    UntunedVariantSpec,
    construire_estimateur_variante,
    creer_cv,
    executer_modele,
    obtenir_model_study_spec,
    selectionner_meilleure_variante_untuned,
)
from src.experiments.report_bundle import generer_comparaison_globale
from src.experiments.shared import ajouter_gaps
from src.notebook_support import (
    construire_tableau_complementarite_erreurs,
    construire_tableau_corroboration_secondaire,
)
from src.artifacts import (
    StudyRunResult,
    TuningRunResult,
    UntunedVariantResult,
    global_paths,
    model_paths,
    sauvegarder_mesures,
    sauvegarder_predictions,
    valider_artefacts_mesures,
)
from src.data import charger_donnees_modelisation
from src.cli.main import build_parser, _parse_advanced_figures


ROOT = Path(__file__).resolve().parents[1]


def _workspace_temp_dir() -> Path:
    path = ROOT / f".tmp_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class TestModelStudySpecs(unittest.TestCase):
    def test_all_6_models_have_specs(self) -> None:
        self.assertEqual(len(ALL_MODEL_NAMES), 6)
        expected = {
            "perceptron",
            "regression_logistique",
            "svm",
            "foret_aleatoire",
            "arbre_decision",
            "mlp",
        }
        self.assertEqual(set(ALL_MODEL_NAMES), expected)

    def test_spec_types(self) -> None:
        for name in ALL_MODEL_NAMES:
            with self.subTest(model=name):
                spec = obtenir_model_study_spec(name)
                self.assertIsInstance(spec, ModelStudySpec)
                self.assertEqual(spec.model_name, name)
                self.assertTrue(len(spec.untuned_variants) >= 1)

    def test_spec_unknown_model_raises(self) -> None:
        with self.assertRaises(KeyError):
            obtenir_model_study_spec("imaginary_model")

    def test_untuned_variant_spec_kinds(self) -> None:
        valid_kinds = {"simple", "pipeline_no_pca", "pipeline_with_pca"}
        for name in ALL_MODEL_NAMES:
            spec = obtenir_model_study_spec(name)
            for variant in spec.untuned_variants:
                with self.subTest(model=name, variant=variant.name):
                    self.assertIn(variant.kind, valid_kinds)


class TestStudySpec(unittest.TestCase):
    def test_default_study_spec(self) -> None:
        study = StudySpec(models=["perceptron"])
        self.assertEqual(study.models, ["perceptron"])
        self.assertEqual(study.figures_mode, "core")
        self.assertFalse(study.no_tuning)

    def test_study_spec_all_models(self) -> None:
        study = StudySpec(models=ALL_MODEL_NAMES)
        self.assertEqual(len(study.models), 6)


class TestRunnerReuse(unittest.TestCase):
    def test_skip_existing_reuses_saved_artifacts(self) -> None:
        chemin_temp = _workspace_temp_dir()
        try:
            paths = model_paths("perceptron", root=chemin_temp)
            sauvegarder_mesures(
                "perceptron",
                "default",
                {
                    "selection_protocol": "simple_cv",
                    "evaluation_stage": "baseline",
                    "timing_policy": "cv_n_jobs=1",
                    "val_accuracy_mean": 0.5,
                    "val_accuracy_std": 0.01,
                    "val_f1_macro_mean": 0.4,
                    "val_f1_macro_std": 0.01,
                    "train_accuracy_mean": 0.6,
                    "train_f1_macro_mean": 0.5,
                    "generalization_gap_accuracy": 0.1,
                    "generalization_gap_f1_macro": 0.1,
                    "elapsed_seconds": 1.0,
                    "n_splits": 5,
                },
                paths=paths,
            )
            sauvegarder_mesures(
                "perceptron",
                "scaled_only",
                {
                    "selection_protocol": "simple_cv",
                    "evaluation_stage": "improved_untuned",
                    "timing_policy": "cv_n_jobs=1",
                    "val_accuracy_mean": 0.6,
                    "val_accuracy_std": 0.01,
                    "val_f1_macro_mean": 0.55,
                    "val_f1_macro_std": 0.01,
                    "train_accuracy_mean": 0.7,
                    "train_f1_macro_mean": 0.65,
                    "generalization_gap_accuracy": 0.1,
                    "generalization_gap_f1_macro": 0.1,
                    "elapsed_seconds": 1.0,
                    "n_splits": 5,
                },
                paths=paths,
            )
            sauvegarder_mesures(
                "perceptron",
                "scaled_pca",
                {
                    "selection_protocol": "simple_cv",
                    "evaluation_stage": "improved_untuned",
                    "timing_policy": "cv_n_jobs=1",
                    "val_accuracy_mean": 0.62,
                    "val_accuracy_std": 0.01,
                    "val_f1_macro_mean": 0.58,
                    "val_f1_macro_std": 0.01,
                    "train_accuracy_mean": 0.72,
                    "train_f1_macro_mean": 0.67,
                    "generalization_gap_accuracy": 0.1,
                    "generalization_gap_f1_macro": 0.09,
                    "elapsed_seconds": 1.0,
                    "n_splits": 5,
                },
                paths=paths,
            )
            sauvegarder_mesures(
                "perceptron",
                "tuned",
                {
                    "selection_protocol": "nested_cv",
                    "evaluation_stage": "tuned",
                    "timing_policy": "cv_n_jobs=1",
                    "val_accuracy_mean": 0.7,
                    "val_accuracy_std": 0.01,
                    "val_f1_macro_mean": 0.66,
                    "val_f1_macro_std": 0.01,
                    "train_accuracy_mean": float("nan"),
                    "train_f1_macro_mean": float("nan"),
                    "generalization_gap_accuracy": float("nan"),
                    "generalization_gap_f1_macro": float("nan"),
                    "elapsed_seconds": 2.0,
                    "n_splits": 5,
                    "inner_splits": 5,
                    "secondary_holdout": {
                        "enabled": True,
                        "corroboration_scope": "corroboration_only",
                        "ranking_eligible": False,
                        "origin": "exploratory_refit",
                        "reference_variant": "scaled_pca",
                        "n_dev": 10,
                        "n_holdout": 2,
                        "tuned": {"f1_macro": 0.7, "accuracy": 0.7, "best_params": {}},
                        "reference": {"variant": "scaled_pca", "f1_macro": 0.6, "accuracy": 0.6},
                    },
                    "descriptive_holdout_all_variants": [
                        {"variante": "default", "f1_macro": 0.4, "accuracy": 0.45, "role": "baseline"},
                        {"variante": "scaled_only", "f1_macro": 0.55, "accuracy": 0.56, "role": "untuned"},
                        {"variante": "scaled_pca", "f1_macro": 0.6, "accuracy": 0.6, "role": "reference"},
                        {"variante": "tuned", "f1_macro": 0.7, "accuracy": 0.7, "role": "tuned"},
                    ],
                },
                paths=paths,
            )
            sauvegarder_predictions(
                "perceptron",
                "tuned",
                pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1]}),
                paths=paths,
            )
            sauvegarder_predictions(
                "perceptron",
                "tuned_holdout",
                pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1]}),
                paths=paths,
            )
            sauvegarder_predictions(
                "perceptron",
                "scaled_pca_holdout",
                pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1]}),
                paths=paths,
            )

            study = StudySpec(
                models=["perceptron"],
                output_root=chemin_temp,
                skip_existing=True,
            )
            with patch(
                "src.experiments.runner.charger_donnees_modelisation",
                side_effect=AssertionError("charger_donnees_modelisation ne devrait pas etre appele"),
            ):
                resultat = executer_modele("perceptron", study)
        finally:
            import shutil

            shutil.rmtree(chemin_temp, ignore_errors=True)

        self.assertTrue(resultat.success)
        self.assertEqual(len(resultat.untuned_results), 3)
        self.assertIsNotNone(resultat.tuning_result)
        self.assertAlmostEqual(resultat.tuning_result.val_f1_macro_mean, 0.66)
        self.assertEqual(
            len(resultat.tuning_result.descriptive_holdout_all_variants),
            4,
        )
        self.assertEqual(
            resultat.tuning_result.secondary_holdout["corroboration_scope"],
            "corroboration_only",
        )
        self.assertFalse(resultat.tuning_result.secondary_holdout["ranking_eligible"])
        self.assertEqual(
            resultat.tuning_result.secondary_holdout["origin"],
            "exploratory_refit",
        )


class TestSharedUtilities(unittest.TestCase):
    """Unit tests for experiments.shared — the single source of truth for both paths."""

    def test_creer_cv_returns_stratified_kfold(self) -> None:
        cv = creer_cv(n_splits=3, random_seed=42)
        self.assertIsInstance(cv, StratifiedKFold)
        self.assertEqual(cv.n_splits, 3)
        self.assertTrue(cv.shuffle)
        self.assertEqual(cv.random_state, 42)

    def test_ajouter_gaps_computes_differences(self) -> None:
        mesures = {
            "train_accuracy_mean": 0.9,
            "val_accuracy_mean": 0.8,
            "train_f1_macro_mean": 0.85,
            "val_f1_macro_mean": 0.7,
        }
        result = ajouter_gaps(mesures)
        self.assertAlmostEqual(result["generalization_gap_accuracy"], 0.1)
        self.assertAlmostEqual(result["generalization_gap_f1_macro"], 0.15)
        # Original dict must not be mutated
        self.assertNotIn("generalization_gap_accuracy", mesures)

    def test_construire_estimateur_variante_simple(self) -> None:
        spec = UntunedVariantSpec(name="default", kind="simple")
        est = construire_estimateur_variante("perceptron", spec)
        self.assertTrue(hasattr(est, "fit"))

    def test_construire_estimateur_variante_pipeline(self) -> None:
        spec = UntunedVariantSpec(name="scaled_only", kind="pipeline_no_pca")
        est = construire_estimateur_variante("perceptron", spec)
        # A pipeline should have named_steps
        self.assertTrue(hasattr(est, "named_steps"))

    def test_construire_estimateur_variante_with_params(self) -> None:
        spec = UntunedVariantSpec(
            name="restricted", kind="simple", params={"max_depth": 5},
        )
        est = construire_estimateur_variante("arbre_decision", spec)
        self.assertEqual(est.get_params()["max_depth"], 5)

    def test_construire_estimateur_variante_invalid_kind_raises(self) -> None:
        spec = UntunedVariantSpec(name="bad", kind="unknown_kind")
        with self.assertRaises(ValueError):
            construire_estimateur_variante("perceptron", spec)

    def test_selectionner_meilleure_variante_untuned(self) -> None:
        spec = obtenir_model_study_spec("perceptron")
        results = [
            UntunedVariantResult(
                variante="default",
                selection_protocol="simple_cv",
                evaluation_stage="baseline",
                val_accuracy_mean=0.5,
                val_accuracy_std=0.01,
                val_f1_macro_mean=0.3,
                val_f1_macro_std=0.01,
                train_accuracy_mean=0.6,
                train_f1_macro_mean=0.5,
                generalization_gap_accuracy=0.1,
                generalization_gap_f1_macro=0.2,
                elapsed_seconds=1.0,
                timing_policy="cv_n_jobs=1",
                n_splits=5,
            ),
            UntunedVariantResult(
                variante="scaled_only",
                selection_protocol="simple_cv",
                evaluation_stage="improved_untuned",
                val_accuracy_mean=0.7,
                val_accuracy_std=0.01,
                val_f1_macro_mean=0.65,
                val_f1_macro_std=0.01,
                train_accuracy_mean=0.8,
                train_f1_macro_mean=0.75,
                generalization_gap_accuracy=0.1,
                generalization_gap_f1_macro=0.1,
                elapsed_seconds=1.0,
                timing_policy="cv_n_jobs=1",
                n_splits=5,
            ),
        ]
        best = selectionner_meilleure_variante_untuned(spec, results)
        self.assertEqual(best.name, "scaled_only")

    def test_construire_tableau_complementarite_erreurs_uses_set_overlap(self) -> None:
        base = ROOT / "results"
        predictions = {
            "regression_logistique": pd.read_csv(
                base / "regression_logistique" / "predictions" / "tuned__predictions.csv"
            ),
            "svm": pd.read_csv(base / "svm" / "predictions" / "tuned__predictions.csv"),
            "foret_aleatoire": pd.read_csv(
                base / "foret_aleatoire" / "predictions" / "tuned__predictions.csv"
            ),
        }

        tableau = construire_tableau_complementarite_erreurs(predictions)
        index = tableau.set_index(["modele_a", "modele_b"])

        self.assertAlmostEqual(
            index.loc[("regression_logistique", "svm"), "error_overlap_rate"],
            0.625,
        )
        self.assertAlmostEqual(
            index.loc[("regression_logistique", "foret_aleatoire"), "error_overlap_rate"],
            5 / 27,
        )
        self.assertAlmostEqual(
            index.loc[("svm", "foret_aleatoire"), "error_overlap_rate"],
            5 / 29,
        )
        self.assertEqual(
            index.loc[("regression_logistique", "svm"), "erreurs_communes"],
            10,
        )
        self.assertEqual(index.loc[("regression_logistique", "svm"), "corriges_par_a_seul"], 4)
        self.assertEqual(index.loc[("regression_logistique", "svm"), "corriges_par_b_seul"], 2)

    def test_valider_artefacts_exige_la_semantique_secondary_holdout(self) -> None:
        artefact_valide = {
            "modele": "svm",
            "variante": "tuned",
            "schema_version": 2,
            "selection_protocol": "nested_cv",
            "evaluation_stage": "tuned",
            "timing_policy": "cv_n_jobs=1",
            "val_f1_macro_mean": 0.9,
            "val_f1_macro_std": 0.01,
            "val_accuracy_mean": 0.9,
            "val_accuracy_std": 0.01,
            "elapsed_seconds": 1.0,
            "n_splits": 5,
            "inner_splits": 5,
            "train_accuracy_mean": float("nan"),
            "train_f1_macro_mean": float("nan"),
            "generalization_gap_accuracy": float("nan"),
            "generalization_gap_f1_macro": float("nan"),
            "secondary_holdout": {
                "enabled": True,
                "corroboration_scope": "corroboration_only",
                "ranking_eligible": False,
                "origin": "exploratory_refit",
                "reference_variant": "scaled_only",
                "tuned": {"f1_macro": 0.9, "accuracy": 0.9, "best_params": {}},
                "reference": {"variant": "scaled_only", "f1_macro": 0.8, "accuracy": 0.8},
            },
        }
        self.assertEqual(valider_artefacts_mesures([artefact_valide]), [])

        artefact_invalide = {
            **artefact_valide,
            "secondary_holdout": {
                **artefact_valide["secondary_holdout"],
                "ranking_eligible": True,
            },
        }
        erreurs = valider_artefacts_mesures([artefact_invalide])
        self.assertTrue(
            any("ranking_eligible" in erreur for erreur in erreurs),
            msg=f"Erreurs inattendues: {erreurs}",
        )

    def test_construire_tableau_corroboration_secondaire_expose_la_semantique(self) -> None:
        mesures_tuned = {
            "secondary_holdout": {
                "enabled": True,
                "corroboration_scope": "corroboration_only",
                "ranking_eligible": False,
                "origin": "exploratory_refit",
                "reference_variant": "scaled_only",
                "tuned": {"f1_macro": 0.9, "accuracy": 0.9, "best_params": {}},
                "reference": {"variant": "scaled_only", "f1_macro": 0.8, "accuracy": 0.8},
            }
        }

        tableau = construire_tableau_corroboration_secondaire(mesures_tuned)

        self.assertEqual(list(tableau["corroboration_scope"].unique()), ["corroboration_only"])
        self.assertEqual(list(tableau["origin"].unique()), ["exploratory_refit"])
        self.assertFalse(tableau["ranking_eligible"].any())


class TestFrozenSplit(unittest.TestCase):
    def test_charger_donnees_modelisation_persiste_un_split_gelee(self) -> None:
        chemin_temp = _workspace_temp_dir()
        csv_path = chemin_temp / "train.csv"
        try:
            lignes = []
            for classe in ("a", "b", "c"):
                for idx in range(10):
                    lignes.append({"id": f"{classe}_{idx}", "species": classe, "x1": idx, "x2": idx + 1})
            pd.DataFrame(lignes).to_csv(csv_path, index=False)

            split = charger_donnees_modelisation(csv_path=csv_path, output_root=chemin_temp)
            split_recharge = charger_donnees_modelisation(csv_path=csv_path, output_root=chemin_temp)
        finally:
            import shutil

            shutil.rmtree(chemin_temp, ignore_errors=True)

        self.assertEqual(len(split.X_dev), 24)
        self.assertEqual(len(split.X_holdout), 6)
        self.assertEqual(split.metadata["n_dev"], 24)
        self.assertEqual(split.metadata["n_holdout"], 6)
        self.assertTrue(Path(split.metadata["split_artifact"]).name.endswith("frozen_holdout_split.json"))
        self.assertTrue(split.X_dev.equals(split_recharge.X_dev))
        self.assertTrue(split.X_holdout.equals(split_recharge.X_holdout))


class TestReportBundle(unittest.TestCase):
    def test_ranked_models_uses_tuned_scores_only(self) -> None:
        chemin_temp = _workspace_temp_dir()
        try:
            study = StudySpec(
                models=["perceptron", "svm"],
                output_root=chemin_temp,
            )
            resultats = [
                StudyRunResult(
                    model_name="perceptron",
                    untuned_results=[
                        UntunedVariantResult(
                            variante="scaled_pca",
                            selection_protocol="simple_cv",
                            evaluation_stage="improved_untuned",
                            val_accuracy_mean=0.91,
                            val_accuracy_std=0.01,
                            val_f1_macro_mean=0.90,
                            val_f1_macro_std=0.01,
                            train_accuracy_mean=0.95,
                            train_f1_macro_mean=0.94,
                            generalization_gap_accuracy=0.04,
                            generalization_gap_f1_macro=0.04,
                            elapsed_seconds=1.0,
                            timing_policy="cv_n_jobs=1",
                            n_splits=5,
                        )
                    ],
                    tuning_result=TuningRunResult(
                        val_accuracy_mean=0.62,
                        val_accuracy_std=0.01,
                        val_f1_macro_mean=0.60,
                        val_f1_macro_std=0.01,
                        elapsed_seconds=2.0,
                        timing_policy="cv_n_jobs=1",
                        n_splits=5,
                        inner_splits=3,
                    ),
                ),
                StudyRunResult(
                    model_name="svm",
                    untuned_results=[
                        UntunedVariantResult(
                            variante="scaled_only",
                            selection_protocol="simple_cv",
                            evaluation_stage="improved_untuned",
                            val_accuracy_mean=0.76,
                            val_accuracy_std=0.01,
                            val_f1_macro_mean=0.74,
                            val_f1_macro_std=0.01,
                            train_accuracy_mean=0.80,
                            train_f1_macro_mean=0.79,
                            generalization_gap_accuracy=0.04,
                            generalization_gap_f1_macro=0.05,
                            elapsed_seconds=1.0,
                            timing_policy="cv_n_jobs=1",
                            n_splits=5,
                        )
                    ],
                    tuning_result=TuningRunResult(
                        val_accuracy_mean=0.83,
                        val_accuracy_std=0.01,
                        val_f1_macro_mean=0.80,
                        val_f1_macro_std=0.01,
                        elapsed_seconds=2.0,
                        timing_policy="cv_n_jobs=1",
                        n_splits=5,
                        inner_splits=3,
                    ),
                ),
            ]

            with patch(
                "src.experiments.report_bundle.canonical_artifact_completeness",
                return_value=pd.DataFrame({"canonical_complete": [True, True]}),
            ):
                tableau = generer_comparaison_globale(resultats, study)

            ranked = pd.read_csv(
                global_paths(root=chemin_temp).ranked_models_file,
                encoding="utf-8",
            )
        finally:
            import shutil

            shutil.rmtree(chemin_temp, ignore_errors=True)

        self.assertIn(
            "meilleure_variante_non_ajustee (scaled_pca)",
            tableau["variante"].tolist(),
        )
        self.assertEqual(ranked.loc[0, "modele"], "svm")
        self.assertAlmostEqual(ranked.loc[0, "best_f1_macro"], 0.80)
        self.assertEqual(ranked.loc[1, "modele"], "perceptron")
        self.assertAlmostEqual(ranked.loc[1, "best_f1_macro"], 0.60)


class TestCLI(unittest.TestCase):
    def test_parse_all_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--all"])
        self.assertTrue(args.all_models)

    def test_parse_specific_models(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--models", "perceptron", "svm"])
        self.assertEqual(args.models, ["perceptron", "svm"])

    def test_parse_figures_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--all", "--figures", "all"])
        self.assertEqual(args.figures, "all")

    def test_parse_no_tuning(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--all", "--no-tuning"])
        self.assertTrue(args.no_tuning)

    def test_parse_advanced_figures(self) -> None:
        result = _parse_advanced_figures(["mlp:tsne", "svm:learning_curve"])
        self.assertEqual(result, {"mlp": ["tsne"], "svm": ["learning_curve"]})

    def test_parse_advanced_figures_empty(self) -> None:
        self.assertEqual(_parse_advanced_figures(None), {})
        self.assertEqual(_parse_advanced_figures([]), {})


if __name__ == "__main__":
    unittest.main()
