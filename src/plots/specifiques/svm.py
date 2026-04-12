"""Plots specifiques pour le SVM."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def tracer_heatmap_C_gamma(
    tableau_grid: pd.DataFrame,
    titre: str = "C vs Gamma Heatmap",
    facet_column: str | None = None,
) -> plt.Figure:
    """Trace la heatmap C x gamma."""
    if facet_column and facet_column in tableau_grid.columns:
        valeurs_facet = tableau_grid[facet_column].unique()
        n_facets = len(valeurs_facet)
        fig, axes = plt.subplots(1, n_facets, figsize=(9 * n_facets, 7))
        if n_facets == 1:
            axes = [axes]
        for ax, val in zip(axes, valeurs_facet):
            sous_ensemble = tableau_grid[tableau_grid[facet_column] == val]
            pivot = sous_ensemble.pivot_table(
                index="param_modele__C",
                columns="param_modele__gamma",
                values="mean_test_f1_macro",
                aggfunc="mean",
            )
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="crest", linewidths=0.5, ax=ax)
            ax.set_title(f"{titre} ({facet_column}={val})")
            ax.set_xlabel("Gamma", fontweight="bold")
            ax.set_ylabel("C", fontweight="bold")
    else:
        fig, ax = plt.subplots(figsize=(9, 7))
        pivot = tableau_grid.pivot_table(
            index="param_modele__C",
            columns="param_modele__gamma",
            values="mean_test_f1_macro",
            aggfunc="mean",
        )
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="crest", linewidths=0.5, ax=ax)
        ax.set_title(titre)
        ax.set_xlabel("Gamma", fontweight="bold")
        ax.set_ylabel("C", fontweight="bold")
    fig.tight_layout()
    return fig


def tracer_support_vectors_par_classe(
    info: dict[str, Any],
    titre: str = "Support Vectors per Class",
) -> plt.Figure:
    """Trace le nombre de vecteurs de support par classe."""
    classes = list(info["n_support_per_class"].keys())
    counts = list(info["n_support_per_class"].values())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(classes)), counts, color="#4C78A8", alpha=0.75)
    ax.set_title(f"{titre} (total: {info['total_support_vectors']}, "
                 f"ratio: {info['ratio_sv_to_samples']:.2%})")
    ax.set_xlabel("Class", fontweight="bold")
    ax.set_ylabel("Number of Support Vectors", fontweight="bold")
    if len(classes) > 30:
        ax.set_xticks(range(0, len(classes), max(1, len(classes) // 20)))
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def tracer_learning_curve_svm(
    donnees: pd.DataFrame,
    titre: str = "Learning Curve (SVM)",
) -> plt.Figure:
    """Trace la learning curve du SVM."""
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
