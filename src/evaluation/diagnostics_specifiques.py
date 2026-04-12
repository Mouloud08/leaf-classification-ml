"""Diagnostics specifiques a chaque famille de modele.

Chaque fonction retourne les donnees brutes (DataFrames, arrays)
necessaires aux plots specifiques. Les plots eux-memes sont dans
src/plots/specifiques/.

NOTE METHODOLOGIQUE — Les fonctions qui appellent ``model.fit(X, y)``
sur le jeu complet sont des diagnostics *exploratoires* (visualisation
de la structure du modele : coefficients, importance, loss curve, etc.).
Elles ne servent PAS a l'evaluation de la performance generalisable —
celle-ci est assuree par ``evaluer_modele_cv`` et
``evaluer_modele_tuned_nested_cv`` dans ``evaluateur.py``.
Les fonctions qui evaluent la performance (``validation_curve_max_depth``,
``regularization_path``, ``compute_learning_curve``) utilisent
correctement la validation croisee stratifiee.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, learning_curve, validation_curve

from ..config import CV_N_JOBS, N_SPLITS, RANDOM_SEED


# ---------------------------------------------------------------------------
# Arbre de decision
# ---------------------------------------------------------------------------


def validation_curve_max_depth(
    estimateur: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    param_range: list[int] | None = None,
) -> pd.DataFrame:
    """Validation curve pour max_depth d'un arbre de decision."""
    if param_range is None:
        param_range = [2, 5, 10, 15, 20, 30, None]

    # validation_curve ne supporte pas None dans param_range,
    # on le traite separement
    range_sans_none = [v for v in param_range if v is not None]
    has_none = None in param_range

    train_scores, val_scores = validation_curve(
        estimateur,
        X,
        y,
        param_name="max_depth",
        param_range=range_sans_none,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED),
        scoring="f1_macro",
        n_jobs=CV_N_JOBS,
    )

    lignes = []
    for i, depth in enumerate(range_sans_none):
        lignes.append(
            {
                "max_depth": depth,
                "train_mean": float(train_scores[i].mean()),
                "train_std": float(train_scores[i].std()),
                "val_mean": float(val_scores[i].mean()),
                "val_std": float(val_scores[i].std()),
            }
        )

    if has_none:
        # Evaluer sans limite de profondeur
        est = clone(estimateur)
        est.set_params(max_depth=None)
        from .evaluateur import evaluer_modele_cv

        mesures = evaluer_modele_cv(est, X, y)
        lignes.append(
            {
                "max_depth": "None",
                "train_mean": mesures["train_f1_macro_mean"],
                "train_std": float("nan"),  # evaluer_modele_cv ne retourne pas le std train
                "val_mean": mesures["val_f1_macro_mean"],
                "val_std": mesures["val_f1_macro_std"],
            }
        )

    return pd.DataFrame(lignes)


def feature_importance_arbre(
    estimateur: Any,
    X: pd.DataFrame,
    y: np.ndarray,
) -> pd.DataFrame:
    """Importance des variables pour un arbre de decision."""
    modele = clone(estimateur)
    modele.fit(X, y)
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": modele.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Foret aleatoire
# ---------------------------------------------------------------------------


def oob_vs_n_estimators(
    estimateur: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    n_values: list[int] | None = None,
) -> pd.DataFrame:
    """Score OOB en fonction du nombre d'arbres."""
    if n_values is None:
        n_values = [50, 100, 200, 400, 600, 800]

    lignes = []
    for n in n_values:
        est = clone(estimateur)
        est.set_params(n_estimators=n, oob_score=True)
        est.fit(X, y)
        lignes.append(
            {
                "n_estimators": n,
                "oob_score": float(est.oob_score_),
            }
        )
    return pd.DataFrame(lignes)


def importance_avec_dispersion(
    estimateur: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Permutation importance avec dispersion inter-repetitions."""
    modele = clone(estimateur)
    modele.fit(X, y)
    result = permutation_importance(
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
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


def loss_curve_mlp(
    estimateur: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    best_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Retourne la loss curve d'un MLP apres refit avec les meilleurs hyperparametres.

    Option A (scientifique) : le modele est entraine sur la totalite des donnees
    en utilisant les hyperparametres identifies par le grid search exploratoire.
    Ce fit sert uniquement au diagnostic de convergence — les metriques de
    performance proviennent de la nested CV.

    Si ``best_params`` est fourni (dict sans prefixe, ex. {"alpha": 0.001}),
    ces hyperparametres remplacent les valeurs par defaut de l'estimateur.
    La colonne ``val_score`` contient le score de validation interne par
    iteration (holdout early-stopping de sklearn, 10% du jeu d'entrainement).
    """
    modele = clone(estimateur)
    if best_params:
        modele.set_params(**best_params)
    modele.fit(X, y)
    if not hasattr(modele, "loss_curve_"):
        raise AttributeError("Le modele n'expose pas loss_curve_.")

    n = len(modele.loss_curve_)
    val_scores = getattr(modele, "validation_scores_", None)
    if val_scores is not None and len(val_scores) == n:
        val_col = list(val_scores)
    else:
        val_col = [float("nan")] * n

    return pd.DataFrame(
        {
            "iteration": range(1, n + 1),
            "loss": modele.loss_curve_,
            "val_score": val_col,
        }
    )


# ---------------------------------------------------------------------------
# Regression logistique
# ---------------------------------------------------------------------------


def regularization_path(
    estimateur: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    C_values: list[float] | None = None,
) -> pd.DataFrame:
    """Score en fonction de la force de regularisation C."""
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    from .evaluateur import evaluer_modele_cv

    lignes = []
    for C in C_values:
        est = clone(estimateur)
        est.set_params(C=C)
        mesures = evaluer_modele_cv(est, X, y)
        lignes.append(
            {
                "C": C,
                "val_f1_macro_mean": mesures["val_f1_macro_mean"],
                "val_f1_macro_std": mesures["val_f1_macro_std"],
                "train_f1_macro_mean": mesures["train_f1_macro_mean"],
            }
        )
    return pd.DataFrame(lignes)


def coefficients_absolus(
    estimateur: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    top_n: int = 20,
) -> pd.DataFrame:
    """Top des coefficients absolus moyens sur toutes les classes."""
    modele = clone(estimateur)
    modele.fit(X, y)
    coef_abs_mean = np.abs(modele.coef_).mean(axis=0)
    df = pd.DataFrame(
        {"feature": X.columns, "coef_abs_mean": coef_abs_mean}
    )
    return df.sort_values("coef_abs_mean", ascending=False).head(top_n).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------


def analyse_support_vectors(
    estimateur: Any,
    X: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, Any]:
    """Analyse basique des vecteurs de support."""
    modele = clone(estimateur)
    modele.fit(X, y)
    n_sv = modele.n_support_
    classes = modele.classes_
    total = sum(n_sv)
    return {
        "n_support_per_class": dict(zip(classes.tolist(), n_sv.tolist())),
        "total_support_vectors": int(total),
        "ratio_sv_to_samples": float(total / len(y)),
    }


# ---------------------------------------------------------------------------
# Perceptron
# ---------------------------------------------------------------------------


def decision_function_histogram(
    estimateur: Any,
    X: pd.DataFrame,
    y: np.ndarray,
) -> pd.DataFrame:
    """Histogramme de la decision_function pour le perceptron."""
    modele = clone(estimateur)
    modele.fit(X, y)
    if not hasattr(modele, "decision_function"):
        raise AttributeError("Le modele n'expose pas decision_function.")
    scores = modele.decision_function(X)
    if scores.ndim == 1:
        return pd.DataFrame({"score": scores, "y_true": y})
    # Multiclasse: prendre le max score
    max_scores = scores.max(axis=1)
    return pd.DataFrame({"score": max_scores, "y_true": y})


# ---------------------------------------------------------------------------
# Generique: learning curve
# ---------------------------------------------------------------------------


def compute_learning_curve(
    estimateur: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    train_sizes: list[float] | None = None,
) -> pd.DataFrame:
    """Calcule la learning curve generique."""
    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimateur,
        X,
        y,
        train_sizes=train_sizes,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED),
        scoring="f1_macro",
        n_jobs=CV_N_JOBS,
    )

    lignes = []
    for i, size in enumerate(train_sizes_abs):
        lignes.append(
            {
                "train_size": int(size),
                "train_mean": float(train_scores[i].mean()),
                "train_std": float(train_scores[i].std()),
                "val_mean": float(val_scores[i].mean()),
                "val_std": float(val_scores[i].std()),
            }
        )
    return pd.DataFrame(lignes)
