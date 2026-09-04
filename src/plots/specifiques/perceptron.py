"""Graphiques spécifiques pour le perceptron."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def tracer_decision_function_histogram(
    donnees: pd.DataFrame,
    titre: str = "Distribution de la fonction de décision",
) -> plt.Figure:
    """Trace l'histogramme de la fonction de décision coloré par statut de prédiction.

    Si la colonne ``correct`` est présente, les prédictions correctes sont
    affichées en vert et les erreurs en rouge — même convention que
    l'histogramme de confiance du MLP.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if "correct" in donnees.columns:
        bins = 50
        ax.hist(
            donnees.loc[donnees["correct"], "score"],
            bins=bins,
            color="#54A24B",
            alpha=0.75,
            edgecolor="white",
            label="Correctes",
        )
        ax.hist(
            donnees.loc[~donnees["correct"], "score"],
            bins=bins,
            color="#E45756",
            alpha=0.75,
            edgecolor="white",
            label="Erreurs",
        )
        ax.legend(frameon=False)
    else:
        ax.hist(
            donnees["score"],
            bins=50,
            color="#4C78A8",
            alpha=0.75,
            edgecolor="white",
        )

    ax.set_title(titre)
    ax.set_xlabel("Score maximal de la fonction de décision", fontweight="bold")
    ax.set_ylabel("Nombre de prédictions", fontweight="bold")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def tracer_coefficients_perceptron(
    donnees: pd.DataFrame,
    titre: str = "Coefficients absolus dominants (Perceptron)",
) -> plt.Figure:
    """Trace les coefficients absolus moyens du perceptron."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=donnees,
        x="coef_abs_mean",
        y="feature",
        palette="mako",
        hue="feature",
        legend=False,
        ax=ax,
    )
    ax.set_title(titre)
    ax.set_xlabel("Coefficient absolu moyen", fontweight="bold")
    ax.set_ylabel("Variable", fontweight="bold")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig
