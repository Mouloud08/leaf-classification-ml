"""Graphiques de matrice de confusion et paires confondues."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns

from .comparison import _format_label


def tracer_top_confusions(
    tableau_confusions: pd.DataFrame,
    top_n: int = 10,
    titre: str = "Top Confusions",
) -> plt.Axes:
    """Plot the most frequent confusion pairs."""
    if tableau_confusions.empty:
        raise ValueError("There are no confusions to plot.")

    donnees = (
        tableau_confusions.sort_values(by="nombre_erreurs", ascending=False)
        .head(top_n)
        .copy()
    )
    donnees["pair"] = (
        donnees["classe_vraie"].astype(str)
        + " -> "
        + donnees["classe_predite"].astype(str)
    )
    figure, axe = plt.subplots(figsize=(10.8, 6.2))
    sns.barplot(
        data=donnees,
        x="nombre_erreurs",
        y="pair",
        color="#4C78A8",
        ax=axe,
    )
    axe.set_title(f"{titre} (Top {top_n})")
    axe.set_xlabel("Number of Errors", fontweight="bold")
    axe.set_ylabel("Confusion Pair (True -> Predicted)", fontweight="bold")
    axe.xaxis.set_major_locator(MaxNLocator(integer=True))
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


def tracer_matrice_confusion(
    tableau_confusion: pd.DataFrame,
    titre: str = "Normalized Confusion Matrix",
    annot: bool = False,
    cmap: str | None = None,
    intervalle_ticks: int | None = None,
    rotation_x: int = 90,
) -> plt.Axes:
    """Trace une heatmap de matrice de confusion."""
    if tableau_confusion.empty:
        raise ValueError("The confusion matrix is empty.")

    valeurs = tableau_confusion.to_numpy(dtype=float)
    taille = len(tableau_confusion.index)
    figure_largeur = min(max(10.0, 0.22 * taille), 18.0)
    figure_hauteur = min(max(8.0, 0.22 * taille), 16.0)

    if cmap is None:
        cmap = sns.blend_palette(
            ["#fbf8f3", "#d8eef3", "#86c5d8", "#2878a3", "#0b1f3a"],
            as_cmap=True,
        )

    vmax = float(valeurs.max()) if valeurs.size else 1.0
    norm = None
    if vmax > 0:
        norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax)

    figure, axe = plt.subplots(figsize=(figure_largeur, figure_hauteur))
    sns.heatmap(
        tableau_confusion,
        cmap=cmap,
        annot=annot,
        fmt=".2f" if annot else "",
        linewidths=0.05,
        linecolor="#ffffff",
        square=True,
        norm=norm,
        cbar_kws={"label": "Score"},
        ax=axe,
    )
    axe.set_title(titre)
    axe.set_xlabel("Predicted Class", fontweight="bold")
    axe.set_ylabel("True Class", fontweight="bold")
    labels_x = list(tableau_confusion.columns)
    labels_y = list(tableau_confusion.index)
    if intervalle_ticks is None:
        intervalle_ticks = max(1, taille // 20)

    if taille > 25:
        positions_x = [index + 0.5 for index in range(0, len(labels_x), intervalle_ticks)]
        positions_y = [index + 0.5 for index in range(0, len(labels_y), intervalle_ticks)]
        axe.set_xticks(positions_x)
        axe.set_xticklabels(labels_x[::intervalle_ticks], rotation=rotation_x, ha="right", fontsize=8)
        axe.set_yticks(positions_y)
        axe.set_yticklabels(labels_y[::intervalle_ticks], fontsize=8)
    else:
        axe.set_xticklabels(labels_x, rotation=rotation_x, ha="right", fontsize=8)
        axe.set_yticklabels(labels_y, fontsize=8)

    figure.tight_layout()
    return axe
