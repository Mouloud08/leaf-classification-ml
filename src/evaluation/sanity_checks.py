"""Contrôles élémentaires pour valider le pipeline expérimental.

Regle 5 des bonnes pratiques ML :
- DummyClassifier comme référence simple (déjà dans evaluateur.py)
- Test de bruit aleatoire : remplacer X par du bruit, verifier ~chance
- Test d'overfit petit echantillon : 5 samples, verifier loss -> 0
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from ..config import RANDOM_SEED
from .evaluateur import evaluer_modele_cv


def verifier_bruit_aleatoire(
    modele_ou_pipeline: Any,
    y: np.ndarray,
    n_features: int = 192,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Remplace les features par du bruit uniforme et evalue.

    Un modele sain doit obtenir une accuracy proche de 1/n_classes.
    Si l'accuracy depasse significativement le hasard, il y a
    probablement une fuite de donnees.
    """
    rng = np.random.RandomState(random_state)
    X_bruit = pd.DataFrame(
        rng.rand(len(y), n_features),
        columns=[f"noise_{i}" for i in range(n_features)],
    )
    mesures = evaluer_modele_cv(clone(modele_ou_pipeline), X_bruit, y)
    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes
    return {
        "val_accuracy_mean": mesures["val_accuracy_mean"],
        "val_f1_macro_mean": mesures["val_f1_macro_mean"],
        "chance_level": chance,
        "near_chance": mesures["val_accuracy_mean"] < chance + 0.05,
    }


def verifier_overfit_petit_echantillon(
    n_samples: int = 10,
    max_iter: int = 2000,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Entraine un MLP sur un micro-echantillon et verifie que la loss converge.

    Si le MLP ne peut pas overfitter 10 exemples, quelque chose est
    fondamentalement casse dans le pipeline d'entrainement.
    """
    rng = np.random.RandomState(random_state)
    X_mini = pd.DataFrame(
        rng.rand(n_samples, 10),
        columns=[f"f{i}" for i in range(10)],
    )
    y_mini = np.arange(n_samples) % 3

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
        tol=1e-8,
    )
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always", ConvergenceWarning)
        mlp.fit(X_mini, y_mini)

    train_accuracy = float(mlp.score(X_mini, y_mini))
    final_loss = float(mlp.loss_) if hasattr(mlp, "loss_") else float("nan")
    convergence_warning = any(
        issubclass(w.category, ConvergenceWarning) for w in warning_list
    )

    return {
        "train_accuracy": train_accuracy,
        "final_loss": final_loss,
        "overfit_ok": train_accuracy >= 0.95,
        "converged": not convergence_warning,
    }
