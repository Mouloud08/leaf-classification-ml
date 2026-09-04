"""Graphiques de tuning d'hyperparamètres et métriques par classe."""

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
    titre: str = "Carte thermique de tuning",
    verifier_doublons: bool = True,
) -> plt.Axes:
    """Trace une carte thermique de tuning d'hyperparamètres.

    Par défaut, lève une erreur si des couples ``(x, y)`` sont dupliqués afin
    d'éviter qu'un état PCA, ou une autre dimension cachée, ne soit moyenné
    silencieusement.
    """
    if tableau.empty:
        raise ValueError("Le tableau de tuning est vide.")

    donnees = tableau.copy()
    if verifier_doublons:
        doublons = donnees.duplicated(subset=[y, x], keep=False)
        if doublons.any():
            raise ValueError(
                "Des combinaisons x/y dupliquées ont été détectées dans le tableau "
                "de tuning. Filtrez ou regroupez explicitement les données avant de "
                "tracer la carte thermique."
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
    titre: str = "Classes les plus difficiles",
) -> plt.Axes:
    """Affiche les classes extrêmes pour une métrique donnée."""
    if tableau_classes.empty:
        raise ValueError("Le tableau des métriques par classe est vide.")
    if metrique not in tableau_classes.columns:
        raise ValueError(f"Métrique inconnue : {metrique}")

    ascending = ordre == "asc"
    donnees = (
        tableau_classes.sort_values(metrique, ascending=ascending).head(top_n).copy()
    )

    figure, axe = plt.subplots(figsize=(11, 6.5))
    sns.barplot(
        data=donnees,
        x=metrique,
        y="classe",
        color="#4C78A8",
        ax=axe,
    )
    axe.set_title(f"{titre} ({_format_label(metrique)})")
    axe.set_xlabel(_format_label(metrique), fontweight="bold")
    axe.set_ylabel("Classe", fontweight="bold")
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


def tracer_hyperparametre_vs_performance(
    tableau_grid: pd.DataFrame,
    x: str,
    y: str = "mean_test_f1_macro",
    hue: str | None = None,
    titre: str = "Hyperparamètre vs performance",
) -> plt.Axes:
    """Trace l'effet d'un hyperparamètre sur la performance moyenne."""
    if tableau_grid.empty:
        raise ValueError("Le tableau de tuning est vide.")
    if x not in tableau_grid.columns or y not in tableau_grid.columns:
        raise ValueError("Les colonnes demandées sont absentes du tableau de tuning.")

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
