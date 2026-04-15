"""Orchestration layer for report-ready classifier notebooks.

Merges notebook-specific display fields (title, prose, diagnostics) with
executable study specs from ``experiments.model_specs``, and exposes
convenience functions that notebooks call cell-by-cell.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from .config import (
    CV_N_JOBS,
    N_SPLITS,
    RANDOM_SEED,
    RESULTS_DIR,
    TIMING_POLICY,
)
from .data import charger_donnees_modelisation
from .evaluation import (
    calculer_metriques_par_classe,
    evaluer_dummy_classifier,
    predictions_oof,
    top_confusions,
)
from .evaluation.diagnostics_specifiques import importance_oob_foret
from .evaluation.sanity_checks import verifier_bruit_aleatoire
from .models import MODELES
from .artifacts import (
    UntunedVariantResult,
    charger_predictions_modele,
)

# Re-export study specs from experiments
from .experiments.model_specs import obtenir_model_study_spec
from .experiments.shared import (
    construire_estimateur_variante,
    creer_cv,
    preparer_tuning_modele,
)
from .experiments.study_spec import UntunedVariantSpec


# ---------------------------------------------------------------------------
# Internal: notebook-only display fields
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _NotebookDisplaySpec:
    """Notebook-only fields — merged with ModelStudySpec at lookup time."""

    title: str
    objective: str
    experiment_design: tuple[str, ...]
    expected_signal: tuple[str, ...]
    exploratory_diagnostics: tuple[str, ...] = ()


_NOTEBOOK_DISPLAY_SPECS: dict[str, _NotebookDisplaySpec] = {
    "perceptron": _NotebookDisplaySpec(
        title="02 - Modele Perceptron",
        objective="Mesurer le plafond d'une baseline lineaire minimale et l'effet du scaling, de la PCA et du tuning.",
        experiment_design=(
            "Le Perceptron sert de test de separabilite lineaire rapide.",
            "Le scaling est traite comme un pre-requis experimental, pas comme un detail de confort.",
            "La PCA reste une variante exploratoire a confirmer uniquement via la selection nested CV.",
        ),
        expected_signal=(
            "Un grand saut entre default et variantes scalees confirme que l'echelle des variables dominait le comportement du modele.",
            "Un gain faible apres tuning ou PCA reste compatible avec une baseline lineaire a capacite limitee.",
        ),
        exploratory_diagnostics=("tuning_heatmap",),
    ),
    "regression_logistique": _NotebookDisplaySpec(
        title="03 - Modele Regression Logistique",
        objective="Etablir la baseline probabiliste principale et documenter ses performances confirmatoires.",
        experiment_design=(
            "La regression logistique est la reference lineaire probabiliste du projet.",
            "Le scaling est obligatoire pour un conditionnement numerique stable.",
            "Les probabilites out-of-fold servent a documenter calibration, top-k et confiance predictive.",
        ),
        expected_signal=(
            "Le scaling doit apporter un gain majeur par rapport a la variante brute.",
            "Une calibration exploitable doit etre visible uniquement sur les probabilites OOF de la nested CV.",
        ),
        exploratory_diagnostics=("tuning_heatmap", "probabilities"),
    ),
    "svm": _NotebookDisplaySpec(
        title="04 - Modele SVM RBF",
        objective="Mesurer la valeur ajoutee d'un noyau RBF sur des donnees correctement scalees.",
        experiment_design=(
            "Le SVM RBF reste sensible a l'echelle; toutes les variantes pretraitees passent par StandardScaler.",
            "La selection finale conserve probability=True pour que le cout temporel publie inclue l'estimation probabiliste.",
            "Le diagnostic principal du tuning reste la surface C x gamma, lue separement par etat PCA.",
        ),
        expected_signal=(
            "Le scaling est un pre-requis; sans lui, le SVM doit rester nettement moins performant.",
            "Le noyau RBF n'est justifie scientifiquement que si le gain confirme compense son cout en temps.",
        ),
        exploratory_diagnostics=("tuning_heatmap",),
    ),
    "foret_aleatoire": _NotebookDisplaySpec(
        title="05 - Modele Foret Aleatoire",
        objective="Quantifier la robustesse d'un ensemble d'arbres sans preprocessing et documenter son interpretabilite exploratoire.",
        experiment_design=(
            "La foret aleatoire est etudiee sans scaling ni PCA.",
            "La variante de profondeur restreinte sert de sanity check de complexite, pas de baseline amelioree officielle.",
            "Les importances publiees dans le notebook doivent venir d'un diagnostic OOB, et rester exploratoires.",
        ),
        expected_signal=(
            "Une forte performance sans preprocessing confirmerait la robustesse du modele sur ce tabulaire multiclasses.",
            "Les importances OOB ne doivent servir qu'a illustrer des tendances, pas a inferrer une causalite.",
        ),
        exploratory_diagnostics=("permutation_importance",),
    ),
    "arbre_decision": _NotebookDisplaySpec(
        title="06 - Modele Arbre de decision",
        objective="Documenter le compromis performance/complexite d'un arbre unique et ses limites de generalisation.",
        experiment_design=(
            "L'arbre de decision est analyse comme modele interpretable mais instable.",
            "La variante de profondeur restreinte reste un controle de complexite simple.",
            "Le tuning etudie explicitement la profondeur, la taille de feuille et l'elagage cost-complexity.",
        ),
        expected_signal=(
            "L'arbre unique devrait rester derriere la foret en performance et en stabilite inter-plis.",
            "Sa valeur scientifique principale est interpretative: profondeur, elagage et top erreurs doivent raconter la limite de generalisation.",
        ),
        exploratory_diagnostics=("complexity_curve",),
    ),
    "mlp": _NotebookDisplaySpec(
        title="07 - Modele MLP",
        objective="Tester la sensibilite d'un reseau dense au preprocessing, au tuning et a la variabilite inter-plis.",
        experiment_design=(
            "Le MLP requiert un scaling strict pour stabiliser l'optimisation.",
            "La PCA est laissee en competition interne mais ne doit etre interpretee qu'a travers la nested CV.",
            "La loss curve et les diagnostics de confiance restent exploratoires s'ils proviennent d'un refit global.",
        ),
        expected_signal=(
            "Sans scaling, le MLP doit s'effondrer ou devenir tres instable; avec scaling, il doit redevenir competitif.",
            "Le tuning ne merite une conclusion positive que si la nested CV confirme un gain au-dela des variantes exploratoires bien pretraitees.",
        ),
        exploratory_diagnostics=("tuning_heatmap", "loss_curve", "probabilities"),
    ),
}


# ---------------------------------------------------------------------------
# Public: merged spec, context, and lookup
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NotebookModelSpec:
    """Decision-complete notebook configuration for one classifier."""

    model_name: str
    title: str
    objective: str
    experiment_design: tuple[str, ...]
    expected_signal: tuple[str, ...]
    untuned_variants: tuple[UntunedVariantSpec, ...]
    tuning_uses_pipeline: bool
    supports_probabilities: bool = False
    tuning_heatmap_axes: tuple[str, str] | None = None
    tuning_facet_column: str | None = None
    exploratory_diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class NotebookContext:
    """Execution context shared by notebook cells."""

    spec: NotebookModelSpec
    definition: dict[str, Any]
    output_root: Path
    X: pd.DataFrame
    y: Any
    X_holdout: pd.DataFrame
    y_holdout: Any
    label_encoder: Any
    split_metadata: dict[str, Any]
    cv_eval: StratifiedKFold    # display only (n_splits) — evaluation functions create their own CV
    cv_tuning: StratifiedKFold  # outer CV used by the confirmatory tuning workflow
    timing_policy: str = TIMING_POLICY


def obtenir_notebook_model_spec(model_name: str) -> NotebookModelSpec:
    """Return the shared notebook configuration for a model."""
    key = model_name.lower()
    if key not in _NOTEBOOK_DISPLAY_SPECS:
        raise KeyError(f"Configuration notebook inconnue: {model_name}")
    display = _NOTEBOOK_DISPLAY_SPECS[key]
    study_spec = obtenir_model_study_spec(key)
    return NotebookModelSpec(
        model_name=study_spec.model_name,
        title=display.title,
        objective=display.objective,
        experiment_design=display.experiment_design,
        expected_signal=display.expected_signal,
        untuned_variants=study_spec.untuned_variants,
        tuning_uses_pipeline=study_spec.tuning_uses_pipeline,
        supports_probabilities=study_spec.supports_probabilities,
        tuning_heatmap_axes=study_spec.tuning_heatmap_axes,
        tuning_facet_column=study_spec.tuning_facet_column,
        exploratory_diagnostics=display.exploratory_diagnostics,
    )


# Computed public dict — backwards-compatible with scripts and notebooks
# that iterate over NOTEBOOK_MODEL_SPECS.
NOTEBOOK_MODEL_SPECS: dict[str, NotebookModelSpec] = {
    key: obtenir_notebook_model_spec(key) for key in _NOTEBOOK_DISPLAY_SPECS
}


def creer_notebook_context(
    model_name: str,
    *,
    output_root: str | Path | None = None,
) -> NotebookContext:
    """Build the shared execution context for a classifier notebook."""
    spec = obtenir_notebook_model_spec(model_name)
    definition = MODELES[model_name.lower()]().definition()
    racine_sortie = Path(output_root) if output_root is not None else RESULTS_DIR
    split = charger_donnees_modelisation(output_root=racine_sortie)
    cv = creer_cv(n_splits=N_SPLITS, random_seed=RANDOM_SEED)
    return NotebookContext(
        spec=spec,
        definition=definition,
        output_root=racine_sortie,
        X=split.X_dev,
        y=split.y_dev,
        X_holdout=split.X_holdout,
        y_holdout=split.y_holdout,
        label_encoder=split.label_encoder,
        split_metadata=split.metadata,
        cv_eval=cv,
        cv_tuning=cv,
    )


# ---------------------------------------------------------------------------
# Notebook execution helpers
# ---------------------------------------------------------------------------

def evaluer_variantes_untuned(
    contexte: NotebookContext,
) -> list[UntunedVariantResult]:
    """Evaluate and persist every untuned variant defined for the notebook."""
    from .experiments.runner import executer_variantes_untuned as _run_untuned
    from .experiments.study_spec import StudySpec

    study = StudySpec(
        models=[contexte.spec.model_name],
        output_root=contexte.output_root,
    )
    spec = obtenir_model_study_spec(contexte.spec.model_name)
    return _run_untuned(spec, contexte.X, contexte.y, study)


def executer_tuning_confirmatoire(
    contexte: NotebookContext,
    resultats_untuned: list[UntunedVariantResult] | None = None,
) -> dict[str, Any]:
    """Run the shared confirmatory tuning workflow for a notebook."""
    from .experiments.runner import executer_tuning_bundle
    from .experiments.study_spec import StudySpec

    model_name = contexte.spec.model_name
    study_spec = obtenir_model_study_spec(model_name)
    study = StudySpec(
        models=[model_name],
        n_splits=contexte.cv_tuning.n_splits,
        random_seed=contexte.cv_tuning.random_state,
        output_root=contexte.output_root,
    )

    bundle = executer_tuning_bundle(
        study_spec,
        X=contexte.X,
        y=contexte.y,
        label_encoder=contexte.label_encoder,
        untuned_results=resultats_untuned or evaluer_variantes_untuned(contexte),
        X_holdout=contexte.X_holdout,
        y_holdout=contexte.y_holdout,
        split_metadata=contexte.split_metadata,
        study=study,
    )
    return {
        key: value
        for key, value in bundle.items()
        if key != "tuning_result"
    }


def charger_ou_generer_predictions_variante(
    model_name: str,
    variante: str,
    X: pd.DataFrame,
    y: Any,
    label_encoder: Any | None = None,
    *,
    root: str | Path | None = None,
) -> pd.DataFrame:
    """Return a standard prediction table for a model variant.

    Confirmatory ``tuned`` predictions are loaded from canonical artifacts.
    Untuned variants are recomputed as OOF predictions with the shared simple-CV
    protocol so notebooks can compare confusion matrices across every variant.
    """
    racine = Path(root) if root is not None else None
    try:
        return charger_predictions_modele(model_name, variante, root=racine)
    except FileNotFoundError:
        pass

    spec = obtenir_model_study_spec(model_name)
    variante_norm = variante.lower()
    variante_spec = next(
        (v for v in spec.untuned_variants if v.name.lower() == variante_norm),
        None,
    )
    if variante_spec is None:
        raise FileNotFoundError(
            f"Aucune prediction disponible et variante inconnue: {model_name}/{variante}"
        )

    estimateur = construire_estimateur_variante(model_name, variante_spec)
    return predictions_oof(
        estimateur,
        X,
        y,
        label_encoder=label_encoder,
    )


# ---------------------------------------------------------------------------
# Notebook table builders
# ---------------------------------------------------------------------------

def construire_tableau_variantes(
    resultats_untuned: list[UntunedVariantResult],
    mesures_tuned: dict[str, Any],
) -> pd.DataFrame:
    """Build the report-safe summary table for notebook variants."""
    from .evaluation.diagnostics_generiques import tableau_variantes
    return tableau_variantes(
        [r.to_dict() for r in resultats_untuned], mesures_tuned,
    )


def construire_tableau_holdout_secondaire(
    mesures_tuned: dict[str, Any],
) -> pd.DataFrame:
    """Construit un tableau compact `tuned` vs reference sur holdout."""
    secondary_holdout = mesures_tuned.get("secondary_holdout", {})
    if not isinstance(secondary_holdout, dict) or not secondary_holdout.get("enabled"):
        return pd.DataFrame()

    reference = secondary_holdout.get("reference", {})
    tuned = secondary_holdout.get("tuned", {})
    tableau = pd.DataFrame(
        [
            {
                "variante": reference.get("variant", secondary_holdout.get("reference_variant")),
                "f1_macro_holdout": reference.get("f1_macro"),
                "accuracy_holdout": reference.get("accuracy"),
                "role": "reference",
            },
            {
                "variante": "tuned",
                "f1_macro_holdout": tuned.get("f1_macro"),
                "accuracy_holdout": tuned.get("accuracy"),
                "role": "tuned",
            },
        ]
    )
    tableau["delta_f1_vs_reference"] = (
        tableau["f1_macro_holdout"] - float(reference.get("f1_macro", np.nan))
    )
    return tableau


def ordonner_tableau_variantes(tableau: pd.DataFrame) -> pd.DataFrame:
    """Order notebook/report variants in a stable, presentation-friendly way."""
    if tableau.empty or "variante" not in tableau.columns:
        return tableau.copy()

    ordre_variantes = {
        "default": 0,
        "scaled_only": 1,
        "scaled_pca": 2,
        "restricted_depth": 3,
        "tuned": 99,
    }
    return (
        tableau.copy()
        .assign(
            _ordre=lambda df: df["variante"].astype("string").map(
                lambda v: ordre_variantes.get(str(v), 50)
            )
        )
        .sort_values(["_ordre", "variante"])
        .drop(columns="_ordre")
        .reset_index(drop=True)
    )


def charger_tableau_variantes_modele(
    nom_modele: str,
    root: Path | None = None,
) -> pd.DataFrame:
    """Load all canonical variant metrics for a model from results/<modele>/metrics."""
    metrics_dir = Path(root or RESULTS_DIR) / nom_modele / "metrics"
    lignes: list[dict[str, Any]] = []
    for chemin in sorted(metrics_dir.glob("*.json")):
        with open(chemin, encoding="utf-8") as fh:
            contenu = json.load(fh)
        lignes.append({"variante": chemin.stem, **contenu})
    tableau = pd.DataFrame(lignes)
    if tableau.empty:
        raise FileNotFoundError(f"Aucune metrique canonique trouvee pour {nom_modele}.")
    return ordonner_tableau_variantes(tableau)


def construire_tableau_variantes_rapport(tableau: pd.DataFrame) -> pd.DataFrame:
    """Return a compact report-ready variant table with explicit status labels."""
    if tableau.empty:
        return tableau.copy()

    statut = tableau.get("evaluation_stage", pd.Series(index=tableau.index, dtype="object")).map(
        {
            "baseline": "exploratoire",
            "improved_untuned": "exploratoire",
            "tuned": "confirmatoire",
        }
    )
    colonnes = [
        "variante",
        "val_f1_macro_mean",
        "val_accuracy_mean",
        "evaluation_stage",
    ]
    present = [c for c in colonnes if c in tableau.columns]
    retour = tableau.loc[:, present].copy()
    retour["statut"] = statut.fillna("")
    retour = retour.rename(
        columns={
            "val_f1_macro_mean": "f1_macro",
            "val_accuracy_mean": "accuracy",
            "evaluation_stage": "evaluation_stage",
        }
    )
    return retour


def construire_resume_confusions_variantes(
    model_name: str,
    summary_table: pd.DataFrame,
    X: pd.DataFrame,
    y: Any,
    label_encoder: Any,
) -> pd.DataFrame:
    """Build a compact per-variant summary before plotting confusion matrices."""
    ordre_variantes = {
        "default": 0,
        "scaled_only": 1,
        "scaled_pca": 2,
        "restricted_depth": 3,
        "tuned": 99,
    }
    variantes_str = summary_table["variante"].astype("string")
    catalogue_variantes = (
        summary_table[
            [
                "variante",
                "evaluation_stage",
                "val_f1_macro_mean",
                "val_accuracy_mean",
                "generalization_gap_f1_macro",
            ]
        ]
        .dropna(subset=["variante"])
        .drop_duplicates(subset=["variante", "evaluation_stage"])
        .assign(
            variante=variantes_str,
            _ordre=lambda df: df["variante"].map(lambda v: ordre_variantes.get(str(v), 50)),
        )
        .sort_values(["_ordre", "variante"])
        .reset_index(drop=True)
    )

    lignes: list[dict[str, Any]] = []
    for ligne in catalogue_variantes.to_dict("records"):
        variante = str(ligne["variante"])
        predictions_variante = charger_ou_generer_predictions_variante(
            model_name,
            variante,
            X,
            y,
            label_encoder=label_encoder,
        )
        confusion = top_confusions(
            predictions_variante["y_true"].to_numpy(),
            predictions_variante["y_pred"].to_numpy(),
            label_encoder=label_encoder,
            top_n=1,
        )
        if confusion.empty:
            confusion_dominante = "aucune"
            erreurs_dominantes = 0
        else:
            premiere = confusion.iloc[0]
            confusion_dominante = (
                f"{premiere['classe_vraie']} -> {premiere['classe_predite']}"
            )
            erreurs_dominantes = int(premiere["nombre_erreurs"])

        lignes.append(
            {
                "variante": variante,
                "evaluation_stage": ligne["evaluation_stage"],
                "val_f1_macro_mean": ligne["val_f1_macro_mean"],
                "val_accuracy_mean": ligne["val_accuracy_mean"],
                "generalization_gap_f1_macro": ligne["generalization_gap_f1_macro"],
                "confusion_dominante": confusion_dominante,
                "nombre_erreurs_dominantes": erreurs_dominantes,
            }
        )

    return pd.DataFrame(lignes)


def construire_tableau_gaps(
    resultats_untuned: list[UntunedVariantResult],
) -> pd.DataFrame:
    """Return the untuned-only generalization gap table."""
    from .evaluation.diagnostics_generiques import tableau_gaps
    return tableau_gaps([r.to_dict() for r in resultats_untuned])


def construire_tableau_plis_externes(nested_tuned: dict[str, Any]) -> pd.DataFrame:
    """Return the nested outer-fold metrics in a notebook-friendly format."""
    return nested_tuned["outer_fold_metrics"].copy()


def calculer_metriques_classes_depuis_predictions(
    predictions_df: pd.DataFrame,
    label_encoder: Any,
) -> pd.DataFrame:
    """Compute per-class metrics from a standard prediction table."""
    return calculer_metriques_par_classe(
        predictions_df["y_true"].to_numpy(),
        predictions_df["y_pred"].to_numpy(),
        label_encoder=label_encoder,
    )


def calculer_importance_permutation_exploratoire(
    estimateur: Any,
    X: pd.DataFrame,
    y: Any,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Fit an exploratory estimator on full data and compute permutation importance."""
    modele = clone(estimateur)
    modele.fit(X, y)
    importance = permutation_importance(
        modele,
        X,
        y,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        n_jobs=CV_N_JOBS,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def calculer_importance_permutation_oob_foret(
    estimateur: Any,
    X: pd.DataFrame,
    y: Any,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Expose le diagnostic d'importance OOB pour le notebook foret."""
    return importance_oob_foret(estimateur, X, y, n_repeats=n_repeats)


def calculer_importance_oob_foret_exploratoire(
    estimateur: Any,
    X: pd.DataFrame,
    y: Any,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Calcule une importance OOB exploratoire specifique a la foret aleatoire."""
    return importance_oob_foret(
        estimateur=estimateur,
        X=X,
        y=y,
        n_repeats=n_repeats,
    )


def construire_tableau_plis_tuned_global(
    X: pd.DataFrame,
    y: Any,
    predictions_par_modele: dict[str, pd.DataFrame],
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Reconstruit les metriques par pli externe a partir des predictions OOF tuned."""
    from sklearn.metrics import accuracy_score, f1_score

    cv = creer_cv(n_splits=n_splits, random_seed=random_state)
    y_array = y.to_numpy() if hasattr(y, "to_numpy") else pd.Series(y).to_numpy()

    lignes: list[dict[str, Any]] = []
    for index_pli, (_, test_idx) in enumerate(cv.split(X, y_array), start=1):
        y_true_pli = y_array[test_idx]
        for modele, predictions in predictions_par_modele.items():
            y_pred_pli = predictions["y_pred"].to_numpy()[test_idx]
            lignes.append(
                {
                    "fold": index_pli,
                    "modele": modele,
                    "val_accuracy": float(accuracy_score(y_true_pli, y_pred_pli)),
                    "val_f1_macro": float(
                        f1_score(y_true_pli, y_pred_pli, average="macro")
                    ),
                }
            )

    return pd.DataFrame(lignes)


def construire_tableau_holdout_global(
    modeles: list[str] | tuple[str, ...],
    root: Path | None = None,
) -> pd.DataFrame:
    """Agrège les corroborations holdout disponibles pour les modeles eligibles."""
    lignes: list[dict[str, Any]] = []
    for nom_modele in modeles:
        chemin = Path(root or RESULTS_DIR) / nom_modele / "metrics" / "tuned.json"
        if not chemin.exists():
            continue
        contenu = json.loads(chemin.read_text(encoding="utf-8"))
        secondary_holdout = contenu.get("secondary_holdout", {})
        if not isinstance(secondary_holdout, dict) or not secondary_holdout.get("enabled"):
            continue
        lignes.append(
            {
                "modele": nom_modele,
                "reference_variant": secondary_holdout.get("reference_variant"),
                "nested_cv_f1": contenu.get("val_f1_macro_mean"),
                "holdout_tuned_f1": secondary_holdout.get("tuned", {}).get("f1_macro"),
                "holdout_reference_f1": secondary_holdout.get("reference", {}).get("f1_macro"),
                "delta_holdout": (
                    secondary_holdout.get("tuned", {}).get("f1_macro", np.nan)
                    - secondary_holdout.get("reference", {}).get("f1_macro", np.nan)
                ),
            }
        )
    return pd.DataFrame(lignes)


def construire_tableau_complementarite_erreurs(
    predictions_par_modele: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare le recouvrement des erreurs entre modeles a partir des predictions OOF."""
    lignes: list[dict[str, Any]] = []
    modeles = list(predictions_par_modele.keys())
    for index_a, modele_a in enumerate(modeles):
        pred_a = predictions_par_modele[modele_a]
        vrai_a = pred_a["y_true"].to_numpy()
        correct_a = pred_a["y_pred"].to_numpy() == vrai_a

        for modele_b in modeles[index_a + 1 :]:
            pred_b = predictions_par_modele[modele_b]
            vrai_b = pred_b["y_true"].to_numpy()
            if not np.array_equal(vrai_a, vrai_b):
                raise ValueError(
                    "Les tableaux de predictions doivent partager le meme ordre de verite terrain."
                )
            correct_b = pred_b["y_pred"].to_numpy() == vrai_b
            erreurs_a = ~correct_a
            erreurs_b = ~correct_b
            desaccord = pred_a["y_pred"].to_numpy() != pred_b["y_pred"].to_numpy()
            lignes.append(
                {
                    "modele_a": modele_a,
                    "modele_b": modele_b,
                    "disagreement_rate": float(desaccord.mean()),
                    "error_overlap_rate": float((erreurs_a & erreurs_b).mean()),
                    "corriges_par_a_seul": int((correct_a & erreurs_b).sum()),
                    "corriges_par_b_seul": int((correct_b & erreurs_a).sum()),
                }
            )

    return pd.DataFrame(lignes)


def executer_verifications_validite(
    contexte: NotebookContext,
) -> dict[str, Any]:
    """Run the standard scientific validity checks for a model notebook."""
    study_spec = obtenir_model_study_spec(contexte.spec.model_name)
    estimateur_validite, _ = preparer_tuning_modele(study_spec)

    dummy = evaluer_dummy_classifier(
        contexte.X,
        contexte.y,
        n_splits=contexte.cv_eval.n_splits,
        random_state=contexte.cv_eval.random_state,
    )
    bruit = verifier_bruit_aleatoire(
        estimateur_validite,
        contexte.y,
        n_features=contexte.X.shape[1],
        random_state=contexte.cv_eval.random_state,
    )

    note = (
        "Les variantes non tunees restent exploratoires; les figures refittees "
        "sur le sous-ensemble de developpement (80 %) aussi. Les predictions "
        "OOF de la variante tuned sont des diagnostics descriptifs "
        "confirmatoires, mais seul le score "
        "tuned en nested CV sert au classement final inter-modeles."
    )

    return {
        "dummy": pd.DataFrame([dummy]).assign(reference="DummyClassifier"),
        "random_noise": pd.DataFrame([bruit]).assign(reference="Bruit aleatoire"),
        "note": note,
    }
