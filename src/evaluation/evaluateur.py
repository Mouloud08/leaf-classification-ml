"""Evaluation par validation croisee et predictions out-of-fold."""

from __future__ import annotations

import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate

from ..config import N_SPLITS, RANDOM_SEED
from .metriques import (
    _obtenir_labels_et_noms,
    calculer_metriques,
    calculer_metriques_par_classe,
    extraire_top_confusions,
)


def _get_cv_n_jobs() -> int:
    """Lit CV_N_JOBS depuis le namespace du package src.evaluation.

    Ceci permet aux notebooks de muter ``src.evaluation.CV_N_JOBS``
    et que la valeur soit prise en compte au moment de l'appel.
    """
    mod = sys.modules.get("src.evaluation")
    if mod and hasattr(mod, "CV_N_JOBS"):
        return mod.CV_N_JOBS
    from ..config import CV_N_JOBS
    return CV_N_JOBS


def _creer_cv(n_splits: int = N_SPLITS, random_state: int = RANDOM_SEED) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def _selectionner_lignes(X: Any, indices: np.ndarray) -> Any:
    """Selectionne des lignes sans supposer le type exact de ``X``."""
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    return X[indices]


def _construire_tableau_predictions(
    y_vrai: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: Any | None = None,
) -> pd.DataFrame:
    """Construit un DataFrame standardise de predictions."""
    dataframe = pd.DataFrame({"y_true": y_vrai, "y_pred": y_pred})
    if label_encoder is not None:
        dataframe["species_true"] = label_encoder.inverse_transform(y_vrai)
        dataframe["species_pred"] = label_encoder.inverse_transform(y_pred)
    return dataframe


def _construire_tableau_probabilites(
    y_vrai: np.ndarray,
    y_pred: np.ndarray,
    probabilites: np.ndarray,
    noms_classes: list[Any],
    label_encoder: Any | None = None,
) -> pd.DataFrame:
    """Construit un DataFrame standardise de probabilites OOF."""
    dataframe = _construire_tableau_predictions(
        y_vrai=y_vrai,
        y_pred=y_pred,
        label_encoder=label_encoder,
    )
    for index, nom_classe in enumerate(noms_classes):
        dataframe[f"proba__{nom_classe}"] = probabilites[:, index]
    return dataframe


def _aligner_probabilites(
    probabilites: np.ndarray,
    classes_modele: np.ndarray,
    labels_attendus: np.ndarray,
) -> np.ndarray:
    """Reordonne les probabilites d'un estimateur selon l'ordre de labels attendu."""
    positions = {label: index for index, label in enumerate(labels_attendus.tolist())}
    alignees = np.zeros((probabilites.shape[0], len(labels_attendus)), dtype=float)
    for index_source, classe in enumerate(classes_modele):
        if classe not in positions:
            continue
        alignees[:, positions[classe]] = probabilites[:, index_source]
    return alignees


def _scoring_standard() -> dict[str, str]:
    return {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
    }


def evaluer_modele_cv(
    modele_ou_pipeline: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_SEED,
) -> dict[str, float]:
    """Evalue proprement un modele avec validation croisee stratifiee."""
    scoring = _scoring_standard()
    cv = _creer_cv(n_splits=n_splits, random_state=random_state)
    debut = perf_counter()
    resultats = cross_validate(
        modele_ou_pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=_get_cv_n_jobs(),
    )
    duree = perf_counter() - debut

    return {
        "val_accuracy_mean": float(np.mean(resultats["test_accuracy"])),
        "val_accuracy_std": float(np.std(resultats["test_accuracy"])),
        "val_f1_macro_mean": float(np.mean(resultats["test_f1_macro"])),
        "val_f1_macro_std": float(np.std(resultats["test_f1_macro"])),
        "train_accuracy_mean": float(np.mean(resultats["train_accuracy"])),
        "train_f1_macro_mean": float(np.mean(resultats["train_f1_macro"])),
        "fit_time_mean": float(np.mean(resultats["fit_time"])),
        "score_time_mean": float(np.mean(resultats["score_time"])),
        "elapsed_seconds": float(duree),
        "n_splits": int(n_splits),
    }


def predictions_oof(
    modele_ou_pipeline: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: Any | None = None,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Retourne les predictions out-of-fold pour l'analyse d'erreurs."""
    cv = _creer_cv(n_splits=n_splits, random_state=random_state)
    predictions = cross_val_predict(
        modele_ou_pipeline,
        X,
        y,
        cv=cv,
        n_jobs=_get_cv_n_jobs(),
    )
    return _construire_tableau_predictions(
        y_vrai=y,
        y_pred=predictions,
        label_encoder=label_encoder,
    )


def probabilites_oof(
    modele_ou_pipeline: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: Any | None = None,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Retourne les probabilites out-of-fold pour les modeles probabilistes."""
    cv = _creer_cv(n_splits=n_splits, random_state=random_state)
    labels, noms_classes = _obtenir_labels_et_noms(y, label_encoder=label_encoder)
    probabilites = np.zeros((len(y), len(labels)), dtype=float)
    predictions = np.empty(len(y), dtype=labels.dtype)

    for train_idx, test_idx in cv.split(X, y):
        estimateur = clone(modele_ou_pipeline)
        X_train = _selectionner_lignes(X, train_idx)
        X_test = _selectionner_lignes(X, test_idx)
        y_train = y[train_idx]
        estimateur.fit(X_train, y_train)
        probas_test = estimateur.predict_proba(X_test)
        classes_modele = getattr(estimateur, "classes_", labels)
        probas_alignees = _aligner_probabilites(
            probabilites=probas_test,
            classes_modele=np.asarray(classes_modele),
            labels_attendus=labels,
        )
        probabilites[test_idx] = probas_alignees
        predictions[test_idx] = labels[np.argmax(probas_alignees, axis=1)]

    return _construire_tableau_probabilites(
        y_vrai=y,
        y_pred=predictions,
        probabilites=probabilites,
        noms_classes=noms_classes,
        label_encoder=label_encoder,
    )


def evaluer_modele_tuned_nested_cv(
    estimateur_ou_pipeline: Any,
    param_grid: dict[str, list[Any]],
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: Any | None = None,
    outer_splits: int = N_SPLITS,
    inner_splits: int = N_SPLITS,
    random_state: int = RANDOM_SEED,
    refit: str = "f1_macro",
    scoring: dict[str, str] | None = None,
    return_probabilities: bool = False,
) -> dict[str, Any]:
    """Evalue un modele tune via une validation croisee imbriquee.

    Le tuning est effectue uniquement sur les donnees d'entrainement de chaque
    fold externe. Les metriques retournees sont donc des estimations non
    biaisees de la performance du modele tune.
    """
    scoring = scoring or _scoring_standard()
    labels, noms_classes = _obtenir_labels_et_noms(y, label_encoder=label_encoder)
    outer_cv = _creer_cv(n_splits=outer_splits, random_state=random_state)

    y_pred_oof = np.empty(len(y), dtype=labels.dtype)
    probabilites_oof = (
        np.full((len(y), len(labels)), np.nan, dtype=float)
        if return_probabilities
        else None
    )
    probabilites_disponibles = return_probabilities

    mesures_plis: list[dict[str, Any]] = []
    meilleurs_parametres: list[dict[str, Any]] = []
    debut_total = perf_counter()

    for index_pli, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train = _selectionner_lignes(X, train_idx)
        X_test = _selectionner_lignes(X, test_idx)
        y_train = y[train_idx]
        y_test = y[test_idx]

        inner_cv = _creer_cv(n_splits=inner_splits, random_state=random_state + index_pli)
        grille = GridSearchCV(
            estimator=clone(estimateur_ou_pipeline),
            param_grid=param_grid,
            scoring=scoring,
            refit=refit,
            cv=inner_cv,
            n_jobs=_get_cv_n_jobs(),
            return_train_score=True,
        )

        debut_fit = perf_counter()
        grille.fit(X_train, y_train)
        fit_time = perf_counter() - debut_fit

        meilleur_modele = grille.best_estimator_
        debut_score = perf_counter()
        y_pred_test = meilleur_modele.predict(X_test)
        score_time = perf_counter() - debut_score
        y_pred_train = meilleur_modele.predict(X_train)
        y_pred_oof[test_idx] = y_pred_test

        mesures_plis.append(
            {
                "fold": index_pli,
                "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
                "train_f1_macro": float(f1_score(y_train, y_pred_train, average="macro")),
                "val_accuracy": float(accuracy_score(y_test, y_pred_test)),
                "val_f1_macro": float(f1_score(y_test, y_pred_test, average="macro")),
                "fit_time": float(fit_time),
                "score_time": float(score_time),
                "best_score_inner": float(grille.best_score_),
            }
        )
        meilleurs_parametres.append({"fold": index_pli, **grille.best_params_})

        if probabilites_disponibles and probabilites_oof is not None:
            try:
                probas_test = meilleur_modele.predict_proba(X_test)
                classes_modele = getattr(meilleur_modele, "classes_", labels)
                probabilites_oof[test_idx] = _aligner_probabilites(
                    probabilites=probas_test,
                    classes_modele=np.asarray(classes_modele),
                    labels_attendus=labels,
                )
            except AttributeError:
                import warnings
                warnings.warn(
                    f"predict_proba non disponible sur le fold {index_pli} — "
                    "les probabilites OOF ne seront pas retournees.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                probabilites_disponibles = False
                probabilites_oof = None

    tableau_plis = pd.DataFrame(mesures_plis)
    mesures = {
        "val_accuracy_mean": float(tableau_plis["val_accuracy"].mean()),
        "val_accuracy_std": float(tableau_plis["val_accuracy"].std(ddof=0)),
        "val_f1_macro_mean": float(tableau_plis["val_f1_macro"].mean()),
        "val_f1_macro_std": float(tableau_plis["val_f1_macro"].std(ddof=0)),
        # Train scores and generalization gaps are not scientifically valid
        # for nested CV tuned results.  Per-fold train scores remain available
        # in ``outer_fold_metrics`` for diagnostic purposes.
        "train_accuracy_mean": float("nan"),
        "train_f1_macro_mean": float("nan"),
        "generalization_gap_accuracy": float("nan"),
        "generalization_gap_f1_macro": float("nan"),
        "fit_time_mean": float(tableau_plis["fit_time"].mean()),
        "score_time_mean": float(tableau_plis["score_time"].mean()),
        "elapsed_seconds": float(perf_counter() - debut_total),
        "n_splits": int(outer_splits),
        "inner_splits": int(inner_splits),
        "selection_protocol": "nested_cv",
    }

    resultat: dict[str, Any] = {
        "metrics": mesures,
        "outer_fold_metrics": tableau_plis,
        "best_params_per_fold": pd.DataFrame(meilleurs_parametres),
        "oof_predictions": _construire_tableau_predictions(
            y_vrai=y,
            y_pred=y_pred_oof,
            label_encoder=label_encoder,
        ),
    }

    if probabilites_disponibles and probabilites_oof is not None:
        resultat["oof_probabilities"] = _construire_tableau_probabilites(
            y_vrai=y,
            y_pred=y_pred_oof,
            probabilites=probabilites_oof,
            noms_classes=noms_classes,
            label_encoder=label_encoder,
        )
    else:
        resultat["oof_probabilities"] = None

    return resultat


def comparer_variantes(resultats: list[dict[str, Any]]) -> pd.DataFrame:
    """Construit un tableau comparatif trie par F1-macro."""
    dataframe = pd.DataFrame(resultats)
    if dataframe.empty:
        return dataframe
    colonnes_prioritaires = [
        "modele",
        "variante",
        "val_f1_macro_mean",
        "val_accuracy_mean",
        "elapsed_seconds",
    ]
    colonnes_existantes = [
        colonne for colonne in colonnes_prioritaires if colonne in dataframe.columns
    ]
    autres_colonnes = [
        colonne for colonne in dataframe.columns if colonne not in colonnes_existantes
    ]
    dataframe = dataframe[colonnes_existantes + autres_colonnes]
    return dataframe.sort_values(
        by="val_f1_macro_mean",
        ascending=False,
    ).reset_index(drop=True)


def evaluer_dummy_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    strategie: str = "most_frequent",
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_SEED,
) -> dict[str, float]:
    """Evalue un DummyClassifier avec la meme CV stratifiee que les autres modeles."""
    dummy = DummyClassifier(strategy=strategie, random_state=random_state)
    mesures = evaluer_modele_cv(dummy, X, y, n_splits=n_splits, random_state=random_state)
    mesures["modele"] = "dummy"
    mesures["variante"] = strategie
    return mesures


class ModelEvaluator:
    """Facade de compatibilite pour l'ancien code du projet."""

    @staticmethod
    def calculer_metriques(y_vrai: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return calculer_metriques(y_vrai, y_pred)

    @staticmethod
    def calculer_metriques_par_classe(
        y_vrai: np.ndarray,
        y_pred: np.ndarray,
        label_encoder: Any | None = None,
    ) -> pd.DataFrame:
        return calculer_metriques_par_classe(y_vrai, y_pred, label_encoder=label_encoder)

    @staticmethod
    def evaluer_modele_tuned_nested_cv(
        estimateur_ou_pipeline: Any,
        param_grid: dict[str, list[Any]],
        X: pd.DataFrame,
        y: np.ndarray,
        label_encoder: Any | None = None,
        outer_splits: int = N_SPLITS,
        inner_splits: int = N_SPLITS,
        random_state: int = RANDOM_SEED,
        refit: str = "f1_macro",
        return_probabilities: bool = False,
    ) -> dict[str, Any]:
        return evaluer_modele_tuned_nested_cv(
            estimateur_ou_pipeline=estimateur_ou_pipeline,
            param_grid=param_grid,
            X=X,
            y=y,
            label_encoder=label_encoder,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            random_state=random_state,
            refit=refit,
            return_probabilities=return_probabilities,
        )

    @staticmethod
    def comparer_modeles(resultats: dict[str, dict[str, float]]) -> pd.DataFrame:
        lignes = []
        for nom_modele, mesures in resultats.items():
            ligne = {"modele": nom_modele}
            ligne.update(mesures)
            lignes.append(ligne)
        tableau = pd.DataFrame(lignes)
        if "f1_macro" in tableau.columns:
            tableau = tableau.sort_values("f1_macro", ascending=False)
        return tableau.reset_index(drop=True)

    @staticmethod
    def afficher_top_erreurs_confusion(
        y_vrai: np.ndarray,
        y_pred: np.ndarray,
        label_encoder: Any,
        top_n: int = 10,
    ) -> pd.DataFrame:
        return extraire_top_confusions(
            y_vrai=y_vrai,
            y_pred=y_pred,
            label_encoder=label_encoder,
            top_n=top_n,
        )

    @staticmethod
    def tracer_comparaison_modeles(resultats: dict[str, dict[str, float]]) -> pd.DataFrame:
        return ModelEvaluator.comparer_modeles(resultats)
