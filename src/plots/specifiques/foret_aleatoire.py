"""Plots specifiques pour la foret aleatoire."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def tracer_oob_vs_n_estimators(
    donnees: pd.DataFrame,
    titre: str = "OOB Score vs Number of Trees",
) -> plt.Figure:
    """Trace le score OOB en fonction du nombre d'estimateurs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        donnees["n_estimators"],
        donnees["oob_score"],
        "o-",
        color="#4C78A8",
        linewidth=2,
    )
    ax.set_title(titre)
    ax.set_xlabel("Number of Trees", fontweight="bold")
    ax.set_ylabel("OOB Score", fontweight="bold")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def tracer_importance_dispersion(
    donnees: pd.DataFrame,
    top_n: int = 20,
    titre: str = "Permutation Importance with Dispersion",
) -> plt.Figure:
    """Trace l'importance avec barres d'erreur."""
    donnees_top = donnees.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        donnees_top["feature"],
        donnees_top["importance_mean"],
        xerr=donnees_top["importance_std"],
        color="#4C78A8",
        alpha=0.75,
        capsize=3,
    )
    ax.invert_yaxis()
    ax.set_title(titre)
    ax.set_xlabel("Mean Importance", fontweight="bold")
    ax.set_ylabel("Feature", fontweight="bold")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig
