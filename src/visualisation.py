"""Fonctions de visualisation reutilisables pour les notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def tracer_barres_comparaison(
    tableau: pd.DataFrame,
    metrique: str = "val_f1_macro_mean",
    titre: str = "Comparaison des performances",
) -> plt.Axes:
    """Trace un barplot simple a partir du tableau de resultats."""
    if tableau.empty:
        raise ValueError("Le tableau de resultats est vide.")

    ordre = tableau.sort_values(by=metrique, ascending=False)
    figure, axe = plt.subplots(figsize=(12, 6))
    sns.barplot(data=ordre, x="modele", y=metrique, hue="variante", ax=axe)
    axe.set_title(titre)
    axe.set_xlabel("Modele")
    axe.set_ylabel(metrique)
    axe.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    return axe


def tracer_temps_vs_performance(
    tableau: pd.DataFrame,
    metrique: str = "val_f1_macro_mean",
) -> plt.Axes:
    """Trace un nuage de points temps d'entrainement vs performance."""
    if tableau.empty:
        raise ValueError("Le tableau de resultats est vide.")

    figure, axe = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=tableau,
        x="elapsed_seconds",
        y=metrique,
        hue="modele",
        style="variante",
        s=120,
        ax=axe,
    )
    axe.set_title("Compromis temps / performance")
    axe.set_xlabel("Temps total de CV (secondes)")
    axe.set_ylabel(metrique)
    figure.tight_layout()
    return axe


def tracer_top_confusions(
    tableau_confusions: pd.DataFrame,
    titre: str = "Top confusions",
) -> plt.Axes:
    """Trace les confusions les plus frequentes."""
    if tableau_confusions.empty:
        raise ValueError("Aucune confusion a tracer.")

    donnees = tableau_confusions.copy()
    donnees["pair"] = (
        donnees["classe_vraie"].astype(str)
        + " -> "
        + donnees["classe_predite"].astype(str)
    )
    figure, axe = plt.subplots(figsize=(12, 6))
    sns.barplot(data=donnees, x="nombre_erreurs", y="pair", ax=axe)
    axe.set_title(titre)
    axe.set_xlabel("Nombre d'erreurs")
    axe.set_ylabel("Confusion")
    figure.tight_layout()
    return axe

