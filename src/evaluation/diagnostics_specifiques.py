"""Diagnostics specifiques a chaque famille de modele.

Chaque fonction retourne les donnees brutes (DataFrames, arrays)
necessaires aux plots specifiques. Les plots eux-memes sont dans
src/plots/specifiques/.

NOTE METHODOLOGIQUE — Les fonctions qui appellent ``model.fit(X, y)``
sur le sous-ensemble de developpement disponible sont des diagnostics
*exploratoires* (visualisation
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
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    learning_curve,
    validation_curve,
)

from ..config import CV_N_JOBS, N_SPLITS, RANDOM_SEED

# ---------------------------------------------------------------------------
# Arbre de decision
# ---------------------------------------------------------------------------


def validation_curve_max_depth(
    estimateur: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    param_range: list[int | None] | None = None,
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
        est = clone(estimateur)
        est.set_params(max_depth=None)
        resultats_none = cross_validate(
            est,
            X,
            y,
            cv=StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=RANDOM_SEED,
            ),
            scoring="f1_macro",
            return_train_score=True,
            n_jobs=CV_N_JOBS,
        )
        lignes.append(
            {
                "max_depth": "None",
                "train_mean": float(resultats_none["train_score"].mean()),
                "train_std": float(resultats_none["train_score"].std()),
                "val_mean": float(resultats_none["test_score"].mean()),
                "val_std": float(resultats_none["test_score"].std()),
            }
        )

    return pd.DataFrame(lignes)


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
    """Permutation importance avec dispersion inter-repetitions.

    Pour la foret aleatoire, on privilegie une lecture OOB afin de rester sur
    un diagnostic hors apprentissage plus coherent avec la litterature.
    """
    nom_modele = getattr(estimateur, "__class__", type(estimateur)).__name__.lower()
    if "randomforest" in nom_modele:
        return importance_oob_foret(estimateur, X, y, n_repeats=n_repeats)

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


def importance_oob_foret(
    estimateur: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Calcule une importance par permutation fondee sur les observations OOB.

    Ce diagnostic suit l'esprit de l'importance de Breiman pour les forets :
    chaque arbre est evalue sur ses observations out-of-bag, puis on mesure la
    baisse de F1-macro apres permutation d'une feature dans cet echantillon OOB.
    Le resultat reste exploratoire, mais il evite le biais du réapprentissage in-sample
    complet qui rendait la lecture de la foret peu informative.
    """
    modele = clone(estimateur)
    modele.fit(X, y)

    if hasattr(modele, "named_steps"):
        foret = modele.named_steps["modele"]
        if len(modele.steps) > 1:
            transformateur = clone(modele[:-1])
            X_modele = transformateur.fit_transform(X, y)
        else:
            X_modele = X.to_numpy()
    else:
        foret = modele
        X_modele = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    if not hasattr(foret, "estimators_samples_"):
        raise ValueError(
            "Le diagnostic OOB exige une foret entrainee avec bootstrap et "
            "l'exposition des echantillons par arbre."
        )

    y_array = np.asarray(y)
    importances_par_feature: list[list[float]] = [[] for _ in range(X_modele.shape[1])]
    rng = np.random.RandomState(RANDOM_SEED)

    for arbre, indices_bootstrap in zip(foret.estimators_, foret.estimators_samples_):
        masque_oob = np.ones(len(y_array), dtype=bool)
        masque_oob[np.unique(indices_bootstrap)] = False
        indices_oob = np.flatnonzero(masque_oob)
        if len(indices_oob) == 0:
            continue

        X_oob = X_modele[indices_oob].copy()
        y_oob = y_array[indices_oob]
        score_reference = f1_score(y_oob, arbre.predict(X_oob), average="macro")

        for indice_feature in range(X_oob.shape[1]):
            for _ in range(n_repeats):
                X_permute = X_oob.copy()
                X_permute[:, indice_feature] = rng.permutation(
                    X_permute[:, indice_feature]
                )
                score_permute = f1_score(
                    y_oob,
                    arbre.predict(X_permute),
                    average="macro",
                )
                importances_par_feature[indice_feature].append(
                    float(score_reference - score_permute)
                )

    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance_mean": [
                    float(np.mean(valeurs)) if valeurs else 0.0
                    for valeurs in importances_par_feature
                ],
                "importance_std": [
                    float(np.std(valeurs, ddof=1)) if len(valeurs) > 1 else 0.0
                    for valeurs in importances_par_feature
                ],
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
    """Retourne la courbe de perte d'un MLP après réapprentissage.

    Le modèle réappris utilise les meilleurs hyperparamètres.

    Option A (scientifique) : le modele est entraine sur la totalite des donnees
    en utilisant les hyperparametres identifies par le grid search exploratoire.
    Ce fit sert uniquement au diagnostic de convergence — les metriques de
    performance proviennent de la nested CV.

    ``estimateur`` peut etre un ``MLPClassifier`` brut ou un pipeline complet
    dont la derniere etape s'appelle ``modele``.

    Si ``best_params`` est fourni, ces hyperparametres remplacent les valeurs
    par defaut de l'estimateur ou du pipeline. Les cles prefixees de type
    ``modele__alpha`` sont donc supportees.

    La colonne ``val_score`` contient le score de validation interne par
    iteration (holdout early-stopping de sklearn, 10% du jeu d'entrainement).
    """
    modele = clone(estimateur)
    if best_params:
        modele.set_params(**best_params)
    modele.fit(X, y)

    estimateur_appris = modele
    if hasattr(modele, "named_steps"):
        estimateur_appris = modele.named_steps.get("modele", estimateur_appris)

    if not hasattr(estimateur_appris, "loss_curve_"):
        raise AttributeError("Le modele n'expose pas loss_curve_.")

    n = len(estimateur_appris.loss_curve_)
    val_scores = getattr(estimateur_appris, "validation_scores_", None)
    if val_scores is not None and len(val_scores) == n:
        val_col = list(val_scores)
    else:
        val_col = [float("nan")] * n

    return pd.DataFrame(
        {
            "iteration": range(1, n + 1),
            "loss": estimateur_appris.loss_curve_,
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
        try:
            est.set_params(C=C)
        except ValueError:
            est.set_params(modele__C=C)
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
    estimateur_appris = (
        modele.named_steps["modele"] if hasattr(modele, "named_steps") else modele
    )
    coef_abs_mean = np.abs(estimateur_appris.coef_).mean(axis=0)
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
    svc = modele.named_steps["modele"] if hasattr(modele, "named_steps") else modele
    n_sv = svc.n_support_
    classes = svc.classes_
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
    """Histogramme de la decision_function pour le perceptron.

    Retourne le score maximum par echantillon, la classe predite et
    un flag correct/incorrect pour permettre une coloration par statut
    de prediction (analogue a l'histogramme de confiance du MLP).
    """
    modele = clone(estimateur)
    modele.fit(X, y)
    if not hasattr(modele, "decision_function"):
        raise AttributeError("Le modele n'expose pas decision_function.")
    scores = modele.decision_function(X)
    if scores.ndim == 1:
        y_pred = (scores >= 0).astype(int)
        return pd.DataFrame({
            "score": scores,
            "y_true": y,
            "y_pred": y_pred,
            "correct": y_pred == y,
        })
    # Multiclasse: prendre le max score et la classe predite
    max_scores = scores.max(axis=1)
    y_pred = modele.predict(X)
    return pd.DataFrame({
        "score": max_scores,
        "y_true": y,
        "y_pred": y_pred,
        "correct": y_pred == y,
    })


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
        # On evite les tres petites fractions d'entrainement pour ce dataset:
        # avec 99 classes et peu d'exemples par classe, elles deviennent peu
        # interpretables et peuvent declencher des warnings trompeurs.
        train_sizes = [0.4, 0.6, 0.8, 1.0]

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
