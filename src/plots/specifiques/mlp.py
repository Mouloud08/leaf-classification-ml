"""Plots specifiques pour le MLP."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def tracer_loss_curve(
    donnees: pd.DataFrame,
    titre: str = "MLP Loss Curve",
) -> plt.Figure:
    """Trace la loss curve et, si disponible, le score de validation interne.

    Si ``donnees`` contient une colonne ``val_score`` non-NaN, un second axe
    y (droite) affiche le score de validation de l'early stopping par iteration.
    Une ligne verticale indique l'iteration de convergence (dernier point).
    """
    has_val = (
        "val_score" in donnees.columns
        and donnees["val_score"].notna().any()
    )

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Training loss — axe gauche
    ax1.plot(
        donnees["iteration"],
        donnees["loss"],
        color="#4C78A8",
        linewidth=1.8,
        label="Training loss",
    )
    ax1.set_xlabel("Iteration", fontweight="bold")
    ax1.set_ylabel("Cross-entropy loss (train)", color="#4C78A8", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")

    if has_val:
        ax2 = ax1.twinx()
        ax2.plot(
            donnees["iteration"],
            donnees["val_score"],
            color="#F58518",
            linewidth=1.8,
            linestyle="--",
            label="Val accuracy (early stopping holdout)",
        )
        ax2.set_ylabel(
            "Validation accuracy\n(early stopping holdout — 10% du train)",
            color="#F58518",
            fontweight="bold",
        )
        ax2.tick_params(axis="y", labelcolor="#F58518")

        stop_iter = int(donnees["iteration"].iloc[-1])
        ax1.axvline(
            stop_iter,
            color="grey",
            linestyle=":",
            linewidth=1.2,
            label=f"Convergence iter={stop_iter}",
        )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9)
    else:
        ax1.legend(frameon=False)

    ax1.set_title(titre)
    sns.despine(ax=ax1, right=False)
    fig.tight_layout()
    return fig


def tracer_learning_curve_mlp(
    donnees: pd.DataFrame,
    titre: str = "Learning Curve (MLP)",
) -> plt.Figure:
    """Trace la learning curve du MLP."""
    fig, ax = plt.subplots(figsize=(10, 6))
    import numpy as np

    ax.plot(donnees["train_size"], donnees["train_mean"], "o-", label="Train", color="#4C78A8")
    ax.fill_between(
        donnees["train_size"],
        np.clip(donnees["train_mean"] - donnees["train_std"], 0, 1),
        np.clip(donnees["train_mean"] + donnees["train_std"], 0, 1),
        alpha=0.15,
        color="#4C78A8",
    )
    ax.plot(donnees["train_size"], donnees["val_mean"], "o-", label="Validation", color="#F58518")
    ax.fill_between(
        donnees["train_size"],
        np.clip(donnees["val_mean"] - donnees["val_std"], 0, 1),
        np.clip(donnees["val_mean"] + donnees["val_std"], 0, 1),
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


def tracer_heatmap_hidden_alpha(
    tableau_grid: pd.DataFrame,
    titre: str = "Hidden Layer Sizes vs Alpha",
) -> plt.Figure:
    """Trace la heatmap hidden_layer_sizes x alpha."""
    pivot = tableau_grid.pivot_table(
        index="param_modele__hidden_layer_sizes",
        columns="param_modele__alpha",
        values="mean_test_f1_macro",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="crest",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(titre)
    ax.set_xlabel("Alpha", fontweight="bold")
    ax.set_ylabel("Hidden Layer Sizes", fontweight="bold")
    fig.tight_layout()
    return fig
