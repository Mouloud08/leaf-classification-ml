"""Graphiques de tuning d'hyperparametres et metriques par classe."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .comparison import _format_label


def tracer_heatmap_tuning(
    tableau: pd.DataFrame,
    x: str,
    y: str,
    valeur: str,
    titre: str = "Tuning Heatmap",
    verifier_doublons: bool = True,
) -> plt.Axes:
    """Plot a hyperparameter tuning heatmap.

    By default, raises if duplicate (x, y) pairs are detected so that
    PCA states (or other hidden dimensions) are never silently averaged.
    """
    if tableau.empty:
        raise ValueError("The tuning table is empty.")

    donnees = tableau.copy()
    if verifier_doublons:
        doublons = donnees.duplicated(subset=[y, x], keep=False)
        if doublons.any():
            raise ValueError(
                "Duplicate x/y combinations detected in the tuning table. "
                "Filter or group the data explicitly before plotting the heatmap."
            )
    donnees = donnees.pivot_table(index=y, columns=x, values=valeur, aggfunc="mean")
    figure, axe = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        donnees,
        annot=True,
        fmt=".3f",
        cmap="crest",
        linewidths=0.5,
        ax=axe,
    )
    axe.set_title(titre)
    axe.set_xlabel(_format_label(x), fontweight="bold")
    axe.set_ylabel(_format_label(y), fontweight="bold")
    figure.tight_layout()
    return axe


def tracer_metrique_par_classe(
    tableau_classes: pd.DataFrame,
    metrique: str = "recall",
    top_n: int = 15,
    ordre: str = "asc",
    titre: str = "Most Difficult Classes",
) -> plt.Axes:
    """Affiche les classes extremes pour une metrique donnee."""
    if tableau_classes.empty:
        raise ValueError("The class metrics table is empty.")
    if metrique not in tableau_classes.columns:
        raise ValueError(f"Unknown metric: {metrique}")

    ascending = ordre == "asc"
    donnees = tableau_classes.sort_values(metrique, ascending=ascending).head(top_n).copy()

    figure, axe = plt.subplots(figsize=(11, 6.5))
    sns.barplot(
        data=donnees,
        x=metrique,
        y="classe",
        palette="mako",
        hue="classe",
        dodge=False,
        legend=False,
        ax=axe,
    )
    axe.set_title(f"{titre} ({_format_label(metrique)})")
    axe.set_xlabel(_format_label(metrique), fontweight="bold")
    axe.set_ylabel("Class", fontweight="bold")
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


def tracer_hyperparametre_vs_performance(
    tableau_grid: pd.DataFrame,
    x: str,
    y: str = "mean_test_f1_macro",
    hue: str | None = None,
    titre: str = "Hyperparameter vs Performance",
) -> plt.Axes:
    """Trace l'effet d'un hyperparametre sur la performance moyenne."""
    if tableau_grid.empty:
        raise ValueError("The tuning table is empty.")
    if x not in tableau_grid.columns or y not in tableau_grid.columns:
        raise ValueError("The requested columns are missing from the tuning table.")

    donnees = tableau_grid.copy()
    figure, axe = plt.subplots(figsize=(10.8, 5.8))
    sns.lineplot(
        data=donnees,
        x=x,
        y=y,
        hue=hue,
        marker="o",
        estimator="mean",
        errorbar=None,
        ax=axe,
    )
    axe.set_title(titre)
    axe.set_xlabel(_format_label(x), fontweight="bold")
    axe.set_ylabel(_format_label(y), fontweight="bold")
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe
