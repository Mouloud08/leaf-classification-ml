"""Plots specifiques pour l'arbre de decision."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree


def tracer_validation_curve_max_depth(
    donnees: pd.DataFrame,
    titre: str = "Validation Curve: max_depth",
) -> plt.Figure:
    """Trace la validation curve train/val en fonction de max_depth."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = donnees["max_depth"].astype(str)
    ax.plot(x, donnees["train_mean"], "o-", label="Train F1-Macro", color="#4C78A8")
    ax.fill_between(
        x,
        donnees["train_mean"] - donnees["train_std"],
        donnees["train_mean"] + donnees["train_std"],
        alpha=0.15,
        color="#4C78A8",
    )
    ax.plot(x, donnees["val_mean"], "o-", label="Validation F1-Macro", color="#F58518")
    ax.fill_between(
        x,
        donnees["val_mean"] - donnees["val_std"],
        donnees["val_mean"] + donnees["val_std"],
        alpha=0.15,
        color="#F58518",
    )
    ax.set_title(titre)
    ax.set_xlabel("max_depth", fontweight="bold")
    ax.set_ylabel("F1-Macro", fontweight="bold")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def tracer_feature_importance(
    donnees: pd.DataFrame,
    top_n: int = 20,
    titre: str = "Feature Importance (Decision Tree)",
) -> plt.Figure:
    """Trace les importances des variables."""
    donnees_top = donnees.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=donnees_top,
        x="importance",
        y="feature",
        palette="mako",
        hue="feature",
        legend=False,
        ax=ax,
    )
    ax.set_title(titre)
    ax.set_xlabel("Importance", fontweight="bold")
    ax.set_ylabel("Feature", fontweight="bold")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def tracer_arbre_tronque(
    estimateur: Any,
    feature_names: list[str],
    class_names: list[str] | None = None,
    max_depth: int = 3,
    titre: str = "Truncated Decision Tree",
) -> plt.Figure:
    """Trace un arbre tronque pour visualisation."""
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
    titre: str = "Learning Curve (Decision Tree)",
) -> plt.Figure:
    """Trace la learning curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(donnees["train_size"], donnees["train_mean"], "o-", label="Train", color="#4C78A8")
    ax.fill_between(
        donnees["train_size"],
        donnees["train_mean"] - donnees["train_std"],
        donnees["train_mean"] + donnees["train_std"],
        alpha=0.15,
        color="#4C78A8",
    )
    ax.plot(donnees["train_size"], donnees["val_mean"], "o-", label="Validation", color="#F58518")
    ax.fill_between(
        donnees["train_size"],
        donnees["val_mean"] - donnees["val_std"],
        donnees["val_mean"] + donnees["val_std"],
        alpha=0.15,
        color="#F58518",
    )
    ax.set_title(titre)
    ax.set_xlabel("Training Set Size", fontweight="bold")
    ax.set_ylabel("F1-Macro", fontweight="bold")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig
