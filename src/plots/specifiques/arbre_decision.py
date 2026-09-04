"""Graphiques spécifiques pour l'arbre de décision."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree


def tracer_validation_curve_max_depth(
    donnees: pd.DataFrame,
    titre: str = "Courbe de validation : profondeur maximale",
) -> plt.Figure:
    """Trace la courbe de validation entraînement/validation selon max_depth."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = donnees["max_depth"].astype(str)
    ax.plot(
        x,
        donnees["train_mean"],
        "o-",
        label="F1-macro entraînement",
        color="#4C78A8",
    )
    ax.fill_between(
        x,
        donnees["train_mean"] - donnees["train_std"],
        donnees["train_mean"] + donnees["train_std"],
        alpha=0.15,
        color="#4C78A8",
    )
    ax.plot(x, donnees["val_mean"], "o-", label="F1-macro validation", color="#F58518")
    ax.fill_between(
        x,
        donnees["val_mean"] - donnees["val_std"],
        donnees["val_mean"] + donnees["val_std"],
        alpha=0.15,
        color="#F58518",
    )
    ax.set_title(titre)
    ax.set_xlabel("Profondeur maximale", fontweight="bold")
    ax.set_ylabel("F1-Macro", fontweight="bold")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def tracer_arbre_tronque(
    estimateur: Any,
    feature_names: list[str],
    class_names: list[str] | None = None,
    max_depth: int = 3,
    titre: str = "Arbre de décision tronqué",
) -> plt.Figure:
    """Trace un arbre tronqué pour visualisation."""
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        estimateur,
        feature_names=feature_names,
        class_names=class_names,
        max_depth=max_depth,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(titre)
    fig.tight_layout()
    return fig


def tracer_learning_curve(
    donnees: pd.DataFrame,
    titre: str = "Courbe d'apprentissage (arbre de décision)",
) -> plt.Figure:
    """Trace la courbe d'apprentissage."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        donnees["train_size"],
        donnees["train_mean"],
        "o-",
        label="Entraînement",
        color="#4C78A8",
    )
    ax.fill_between(
        donnees["train_size"],
        donnees["train_mean"] - donnees["train_std"],
        donnees["train_mean"] + donnees["train_std"],
        alpha=0.15,
        color="#4C78A8",
    )
    ax.plot(
        donnees["train_size"],
        donnees["val_mean"],
        "o-",
        label="Validation",
        color="#F58518",
    )
    ax.fill_between(
        donnees["train_size"],
        donnees["val_mean"] - donnees["val_std"],
        donnees["val_mean"] + donnees["val_std"],
        alpha=0.15,
        color="#F58518",
    )
    ax.set_title(titre)
    ax.set_xlabel("Taille du jeu d'entraînement", fontweight="bold")
    ax.set_ylabel("F1-Macro", fontweight="bold")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig
