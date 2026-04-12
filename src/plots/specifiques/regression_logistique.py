"""Plots specifiques pour la regression logistique."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def tracer_regularization_path(
    donnees: pd.DataFrame,
    titre: str = "Regularization Path (C)",
) -> plt.Figure:
    """Trace le score en fonction de C."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(
        donnees["C"],
        donnees["val_f1_macro_mean"],
        "o-",
        label="Validation F1-Macro",
        color="#F58518",
    )
    ax.semilogx(
        donnees["C"],
        donnees["train_f1_macro_mean"],
        "o-",
        label="Train F1-Macro",
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
    ax.set_xlabel("C (inverse regularization strength)", fontweight="bold")
    ax.set_ylabel("F1-Macro", fontweight="bold")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def tracer_coefficients_absolus(
    donnees: pd.DataFrame,
    titre: str = "Top Absolute Coefficients",
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
    ax.set_xlabel("Mean Absolute Coefficient", fontweight="bold")
    ax.set_ylabel("Feature", fontweight="bold")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig
