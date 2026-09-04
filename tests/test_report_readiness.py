"""Tests des artefacts prêts pour le rapport et des garde-fous de notebooks."""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from src.artifacts import (  # noqa: E402  # noqa: E402
    REQUIRED_METRIC_KEYS,
    MetricsArtifactValidationError,
    UntunedVariantResult,
    charger_mesures,
    charger_tableau_mesures_valides,
    model_paths,
    sauvegarder_mesures,
    valider_artefacts_mesures,
)
from src.experiments.model_specs import obtenir_model_study_spec  # noqa: E402
from src.notebook_support import (  # noqa: E402
    NotebookModelSpec,
    construire_tableau_gaps,
    construire_tableau_holdout_descriptif_variantes,
    construire_tableau_variantes,
    creer_notebook_context,
    obtenir_notebook_model_spec,
)
from src.plots.generiques.curves import tracer_histogramme_confiance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_NOTEBOOKS = (
    "02_Modele_Perceptron.ipynb",
    "03_Modele_Regression_Logistique.ipynb",
    "04_Modele_SVM.ipynb",
    "05_Modele_Foret_Aleatoire.ipynb",
    "06_Modele_Arbre_Decision.ipynb",
    "07_Modele_MLP.ipynb",
)
CLASSIFIER_HEADINGS = (
    "## 1. Objectif et protocole",
    "## 2. Signal attendu et design experimental",
    "## 3. Résultats exploratoires non tunés",
    "## 4. Résultat confirmatoire ajusté",
    "## 5. Vérifications de validité scientifique",
    "## 6. Diagnostics du modele",
    "## 7. Interpretation",
)
HOLDOUT_NOTEBOOKS = (
    "02_Modele_Perceptron.ipynb",
    "03_Modele_Regression_Logistique.ipynb",
    "04_Modele_SVM.ipynb",
    "05_Modele_Foret_Aleatoire.ipynb",
    "07_Modele_MLP.ipynb",
)
COMPARISON_HEADINGS = (
    "## 1. Protocole de synthese et garde-fous",
    "## 2. Modèle de référence",
    "## 3. Résultats confirmatoires par modèle",
    "## 4. Variantes exploratoires et coût",
    "## 5. Diagnostics multiclasses du meilleur modele",
    "## 6. Interpretation",
    "## 7. Corroboration secondaire sur le jeu de corroboration",
)


def _workspace_temp_dir() -> Path:
    path = ROOT / f".tmp_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class TestReportReadiness(unittest.TestCase):
    def test_obtenir_notebook_model_spec(self) -> None:
        spec = obtenir_notebook_model_spec("mlp")
        self.assertIsInstance(spec, NotebookModelSpec)
        self.assertEqual(spec.model_name, "mlp")
        self.assertTrue(spec.tuning_uses_pipeline)
        self.assertTrue(len(spec.expected_signal) >= 1)

    def test_notebook_spec_reuses_executable_study_spec(self) -> None:
        notebook_spec = obtenir_notebook_model_spec("svm")
        study_spec = obtenir_model_study_spec("svm")
        self.assertEqual(notebook_spec.untuned_variants, study_spec.untuned_variants)
        self.assertEqual(
            notebook_spec.tuning_uses_pipeline,
            study_spec.tuning_uses_pipeline,
        )
        self.assertEqual(
            notebook_spec.supports_probabilities,
            study_spec.supports_probabilities,
        )
        self.assertEqual(
            notebook_spec.tuning_heatmap_axes,
            study_spec.tuning_heatmap_axes,
        )

    def test_construire_tableaux_notebook(self) -> None:
        untuned = [
            UntunedVariantResult(
                variante="default",
                selection_protocol="simple_cv",
                evaluation_stage="baseline",
                val_accuracy_mean=0.5,
                val_accuracy_std=0.1,
                val_f1_macro_mean=0.4,
                val_f1_macro_std=0.1,
                train_accuracy_mean=0.7,
                train_f1_macro_mean=0.6,
                generalization_gap_accuracy=0.2,
                generalization_gap_f1_macro=0.2,
                elapsed_seconds=1.0,
                timing_policy="cv_n_jobs=1",
                n_splits=5,
            ),
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

        tableau = construire_tableau_variantes(untuned, tuned)
        gaps = construire_tableau_gaps(untuned)

        self.assertEqual(list(tableau["variante"]), ["tuned", "default"])
        self.assertEqual(list(gaps["variante"]), ["default"])

    def test_construire_tableau_holdout_descriptif_variantes(self) -> None:
        mesures_tuned = {
            "descriptive_holdout_all_variants": [
                {
                    "variante": "scaled_only",
                    "f1_macro": 0.7,
                    "accuracy": 0.71,
                    "role": "reference",
                },
                {
                    "variante": "default",
                    "f1_macro": 0.5,
                    "accuracy": 0.52,
                    "role": "baseline",
                },
                {
                    "variante": "scaled_pca",
                    "f1_macro": 0.68,
                    "accuracy": 0.69,
                    "role": "untuned",
                },
                {
                    "variante": "tuned",
                    "f1_macro": 0.72,
                    "accuracy": 0.73,
                    "role": "tuned",
                },
            ]
        }

        tableau = construire_tableau_holdout_descriptif_variantes(mesures_tuned)

        self.assertEqual(
            list(tableau["variante"]),
            ["default", "scaled_only", "scaled_pca", "tuned"],
        )
        self.assertEqual(tableau.loc[1, "role"], "reference")
        self.assertAlmostEqual(tableau.loc[0, "delta_f1_vs_reference"], -0.2)
        self.assertAlmostEqual(tableau.loc[3, "delta_f1_vs_reference"], 0.02)

    def test_sauvegarder_mesures_injecte_schema(self) -> None:
        chemin_temp = _workspace_temp_dir()
        try:
            chemin = sauvegarder_mesures(
                "demo",
                "default",
                {
                    "val_accuracy_mean": 0.8,
                    "val_accuracy_std": 0.01,
                    "val_f1_macro_mean": 0.7,
                    "val_f1_macro_std": 0.02,
                    "train_accuracy_mean": 0.9,
                    "train_f1_macro_mean": 0.85,
                    "generalization_gap_accuracy": 0.1,
                    "generalization_gap_f1_macro": 0.15,
                    "elapsed_seconds": 3.0,
                    "n_splits": 5,
                },
                paths=model_paths("demo", root=chemin_temp),
            )
            contenu = json.loads(chemin.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(chemin_temp, ignore_errors=True)

        self.assertTrue(REQUIRED_METRIC_KEYS.issubset(contenu.keys()))
        self.assertEqual(contenu["selection_protocol"], "simple_cv")
        self.assertEqual(contenu["evaluation_stage"], "baseline")

    def test_charger_mesures_reads_canonical_layout(self) -> None:
        chemin_temp = _workspace_temp_dir()
        try:
            chemins = model_paths("demo", root=chemin_temp)
            sauvegarder_mesures(
                "demo",
                "tuned",
                {
                    "val_accuracy_mean": 0.8,
                    "val_accuracy_std": 0.01,
                    "val_f1_macro_mean": 0.7,
                    "val_f1_macro_std": 0.02,
                    "elapsed_seconds": 2.5,
                    "inner_splits": 3,
                },
                paths=chemins,
            )
            records = charger_mesures(metrics_dir=chemin_temp, modeles=["demo"])
        finally:
            shutil.rmtree(chemin_temp, ignore_errors=True)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["modele"], "demo")
        self.assertEqual(records[0]["variante"], "tuned")

    def test_valider_artefacts_rejette_tuned_non_nested(self) -> None:
        artefact = {
            "modele": "svm",
            "variante": "tuned",
            "schema_version": 2,
            "selection_protocol": "simple_cv",
            "evaluation_stage": "tuned",
            "timing_policy": "cv_n_jobs=1",
            "val_f1_macro_mean": 0.9,
            "val_f1_macro_std": 0.01,
            "val_accuracy_mean": 0.9,
            "val_accuracy_std": 0.01,
            "elapsed_seconds": 1.0,
            "n_splits": 5,
        }
        erreurs = valider_artefacts_mesures([artefact])
        self.assertTrue(
            any("selection_protocol='nested_cv'" in erreur for erreur in erreurs)
        )

    def test_charger_tableau_mesures_valides_rejette_artefact_melange(self) -> None:
        chemin_temp = _workspace_temp_dir()
        try:
            artefact = {
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
                "train_accuracy_mean": 1.0,
                "train_f1_macro_mean": 1.0,
                "generalization_gap_accuracy": 0.1,
                "generalization_gap_f1_macro": 0.1,
            }
            (chemin_temp / "svm__tuned.json").write_text(
                json.dumps(artefact, indent=2),
                encoding="utf-8",
            )
            with self.assertRaises(MetricsArtifactValidationError):
                charger_tableau_mesures_valides(
                    metrics_dir=chemin_temp,
                )
        finally:
            shutil.rmtree(chemin_temp, ignore_errors=True)

    def test_tracer_histogramme_confiance_supporte_labels_non_numeriques(self) -> None:
        tableau = pd.DataFrame(
            {
                "y_true": ["oak", "maple", "oak", "maple"],
                "y_pred": ["oak", "oak", "oak", "maple"],
                "proba__oak": [0.9, 0.6, 0.8, 0.2],
                "proba__maple": [0.1, 0.4, 0.2, 0.8],
            }
        )
        axe = tracer_histogramme_confiance(tableau)
        self.assertIsNotNone(axe)

    def test_notebooks_classifier_template_sections(self) -> None:
        for notebook_name in CLASSIFIER_NOTEBOOKS:
            contenu = json.loads(
                (ROOT / "notebooks" / notebook_name).read_text(encoding="utf-8")
            )
            texte = "\n".join(
                "".join(cell.get("source", [])) for cell in contenu["cells"]
            )
            for heading in CLASSIFIER_HEADINGS:
                self.assertIn(heading, texte, msg=f"{notebook_name} missing {heading}")
            self.assertNotIn(
                "## Conclusion",
                texte,
                msg=f"{notebook_name} should not contain conclusions",
            )
            self.assertIn(
                "executer_verifications_validite(",
                texte,
                msg=f"{notebook_name} missing validity helper",
            )
            self.assertNotIn(
                "obsolete",
                texte.lower(),
                msg=f"{notebook_name} should not mention cleanup internals",
            )
            self.assertNotIn(
                "model_paths(",
                texte,
                msg=f"{notebook_name} should not manipulate artifact paths directly",
            )

    def test_notebook_comparison_template_sections(self) -> None:
        contenu = json.loads(
            (ROOT / "notebooks" / "08_Comparaison_Globale.ipynb").read_text(
                encoding="utf-8"
            )
        )
        texte = "\n".join("".join(cell.get("source", [])) for cell in contenu["cells"])
        for heading in COMPARISON_HEADINGS:
            self.assertIn(heading, texte)
        self.assertIn(
            "## 7. Corroboration secondaire sur le jeu de corroboration", texte
        )
        self.assertNotIn("## Conclusion", texte)
        self.assertNotIn("src.resultats", texte)
        self.assertNotIn("obsolete", texte.lower())
        self.assertNotIn("model_paths(", texte)

    def test_holdout_section_added_only_to_eligible_notebooks(self) -> None:
        for notebook_name in CLASSIFIER_NOTEBOOKS:
            contenu = json.loads(
                (ROOT / "notebooks" / notebook_name).read_text(encoding="utf-8")
            )
            texte = "\n".join(
                "".join(cell.get("source", [])) for cell in contenu["cells"]
            )
            if notebook_name in HOLDOUT_NOTEBOOKS:
                self.assertIn(
                    "## 8. Vérification secondaire sur le jeu de corroboration",
                    texte,
                )
            else:
                self.assertNotIn(
                    "## 8. Vérification secondaire sur le jeu de corroboration",
                    texte,
                )

    def test_holdout_notebooks_include_secondary_plot(self) -> None:
        for notebook_name in HOLDOUT_NOTEBOOKS:
            contenu = json.loads(
                (ROOT / "notebooks" / notebook_name).read_text(encoding="utf-8")
            )
            texte = "\n".join(
                "".join(cell.get("source", [])) for cell in contenu["cells"]
            )
            self.assertIn(
                "tableau_corroboration = construire_tableau_corroboration_secondaire(",
                texte,
            )
            self.assertIn("axe_corroboration = tracer_comparaison_variantes(", texte)
            self.assertIn("f1_macro_holdout", texte)
            self.assertIn("accuracy_holdout", texte)
            self.assertIn(
                "### Lecture descriptive de toutes les variantes sur le jeu de corroboration",
                texte,
            )
            self.assertIn(
                "construire_tableau_corroboration_descriptive_variantes", texte
            )
            self.assertIn(
                "axe_corroboration_descriptif = tracer_comparaison_variantes(", texte
            )
            self.assertIn("Ce graphique est purement descriptif.", texte)

    def test_notebooks_bootstrap_src_from_notebooks_directory(self) -> None:
        for notebook_name in CLASSIFIER_NOTEBOOKS + ("08_Comparaison_Globale.ipynb",):
            contenu = json.loads(
                (ROOT / "notebooks" / notebook_name).read_text(encoding="utf-8")
            )
            first_code = next(
                cell for cell in contenu["cells"] if cell["cell_type"] == "code"
            )
            texte = "".join(first_code.get("source", []))
            self.assertNotIn("sys.path.insert", texte)
            self.assertNotIn("Path.cwd()", texte)
            self.assertNotIn('ROOT / "src"', texte)

    def test_modeling_notebooks_pin_an_output_root(self) -> None:
        for notebook_name in CLASSIFIER_NOTEBOOKS + ("08_Comparaison_Globale.ipynb",):
            contenu = json.loads(
                (ROOT / "notebooks" / notebook_name).read_text(encoding="utf-8")
            )
            texte = "\n".join(
                "".join(cell.get("source", [])) for cell in contenu["cells"]
            )
            self.assertIn("OUTPUT_ROOT = RESULTS_DIR", texte)

    def test_notebook_context_smoke_builds_for_all_models(self) -> None:
        for model_name in (
            "perceptron",
            "regression_logistique",
            "svm",
            "foret_aleatoire",
            "arbre_decision",
            "mlp",
        ):
            contexte = creer_notebook_context(model_name)
            self.assertEqual(contexte.spec.model_name, model_name)
            self.assertFalse(contexte.X.empty)
            self.assertEqual(len(contexte.X), len(contexte.y))

    def test_notebook_context_respects_custom_output_root(self) -> None:
        chemin_temp = _workspace_temp_dir()
        try:
            contexte = creer_notebook_context("perceptron", output_root=chemin_temp)
        finally:
            shutil.rmtree(chemin_temp, ignore_errors=True)

        self.assertEqual(contexte.output_root, chemin_temp)
        self.assertIn(str(chemin_temp), contexte.split_metadata["split_artifact"])


if __name__ == "__main__":
    unittest.main()
