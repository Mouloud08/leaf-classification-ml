"""Graphiques spécifiques pour la régression logistique."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def tracer_regularization_path(
    donnees: pd.DataFrame,
    titre: str = "Chemin de régularisation (C)",
) -> plt.Figure:
    """Trace la performance en fonction de C."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(
        donnees["C"],
        donnees["val_f1_macro_mean"],
        "o-",
        label="F1-macro de validation",
        color="#F58518",
    )
    ax.semilogx(
        donnees["C"],
        donnees["train_f1_macro_mean"],
        "o-",
        label="F1-macro d'entraînement",
        color="#4C78A8",
    )
    if "val_f1_macro_std" in donnees.columns:
        ax.fill_between(
            donnees["C"],
            donnees["val_f1_macro_mean"] - donnees["val_f1_macro_std"],
            donnees["val_f1_macro_mean"] + donnees["val_f1_macro_std"],
            alpha=0.15,
            color="#F58518",
        )
    ax.set_title(titre)
    ax.set_xlabel("C (inverse de la force de régularisation)", fontweight="bold")
    ax.set_ylabel("F1-Macro", fontweight="bold")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def tracer_coefficients_absolus(
    donnees: pd.DataFrame,
    titre: str = "Coefficients absolus dominants",
) -> plt.Figure:
    """Trace les coefficients absolus moyens."""
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
