"""Evaluation standardisee des modeles et variantes."""

from __future__ import annotations

from collections import Counter
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from .config import CV_N_JOBS, N_SPLITS, RANDOM_SEED


def _creer_cv(n_splits: int = N_SPLITS, random_state: int = RANDOM_SEED) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def evaluer_modele_cv(
    modele_ou_pipeline: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_SEED,
) -> dict[str, float]:
    """Evalue proprement un modele avec validation croisee stratifiee."""
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
    }
    cv = _creer_cv(n_splits=n_splits, random_state=random_state)
    debut = perf_counter()
    resultats = cross_validate(
        modele_ou_pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=CV_N_JOBS,
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


def calculer_metriques(y_vrai: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcule les metriques standards sur des predictions deja produites."""
    return {
        "accuracy": float(accuracy_score(y_vrai, y_pred)),
        "f1_macro": float(f1_score(y_vrai, y_pred, average="macro")),
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
        n_jobs=CV_N_JOBS,
    )
    dataframe = pd.DataFrame({"y_true": y, "y_pred": predictions})
    if label_encoder is not None:
        dataframe["species_true"] = label_encoder.inverse_transform(y)
        dataframe["species_pred"] = label_encoder.inverse_transform(predictions)
    return dataframe


def extraire_top_confusions(
    y_vrai: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: Any | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Extrait les paires de classes les plus confondues."""
    matrice = confusion_matrix(y_vrai, y_pred)
    np.fill_diagonal(matrice, 0)

    paires = []
    for index_ligne, ligne in enumerate(matrice):
        for index_colonne, valeur in enumerate(ligne):
            if valeur <= 0:
                continue
            paires.append((index_ligne, index_colonne, int(valeur)))

    paires = sorted(paires, key=lambda item: item[2], reverse=True)[:top_n]
    enregistrements = []
    for vrai, predit, compte in paires:
        vrai_label = vrai
        predit_label = predit
        if label_encoder is not None:
            vrai_label = label_encoder.inverse_transform([vrai])[0]
            predit_label = label_encoder.inverse_transform([predit])[0]
        enregistrements.append(
            {
                "classe_vraie": vrai_label,
                "classe_predite": predit_label,
                "nombre_erreurs": compte,
            }
        )
    return pd.DataFrame(enregistrements)


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


class ModelEvaluator:
    """Facade de compatibilite pour l'ancien code du projet."""

    @staticmethod
    def calculer_metriques(y_vrai: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return calculer_metriques(y_vrai, y_pred)

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
