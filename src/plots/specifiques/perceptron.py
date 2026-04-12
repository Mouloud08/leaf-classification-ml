"""Plots specifiques pour le Perceptron."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def tracer_decision_function_histogram(
    donnees: pd.DataFrame,
    titre: str = "Decision Function Distribution",
) -> plt.Figure:
    """Trace l'histogramme de la decision function."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(donnees["score"], bins=50, color="#4C78A8", alpha=0.75, edgecolor="white")
    ax.set_title(titre)
    ax.set_xlabel("Decision Function Score", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def tracer_coefficients_perceptron(
    donnees: pd.DataFrame,
    titre: str = "Top Absolute Coefficients (Perceptron)",
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
    ax.set_xlabel("Mean Absolute Coefficient", fontweight="bold")
    ax.set_ylabel("Feature", fontweight="bold")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig
