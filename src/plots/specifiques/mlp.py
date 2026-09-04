"""Graphiques spécifiques pour le MLP."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def tracer_loss_curve(
    donnees: pd.DataFrame,
    titre: str = "Courbe de perte du MLP",
) -> plt.Figure:
    """Trace la courbe de perte et, si disponible, la performance de validation interne.

    Si ``donnees`` contient une colonne ``val_score`` non-NaN, un second axe
    y (droite) affiche la performance de validation de l'arrêt anticipé par itération.
    Une ligne verticale indique l'itération de convergence (dernier point).
    """
    has_val = "val_score" in donnees.columns and donnees["val_score"].notna().any()

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Perte d'entraînement — axe gauche
    ax1.plot(
        donnees["iteration"],
        donnees["loss"],
        color="#4C78A8",
        linewidth=1.8,
        label="Perte d'entraînement",
    )
    ax1.set_xlabel("Itération", fontweight="bold")
    ax1.set_ylabel(
        "Perte d'entraînement (entropie croisée)",
        color="#4C78A8",
        fontweight="bold",
    )
    ax1.tick_params(axis="y", labelcolor="#4C78A8")

    if has_val:
        ax2 = ax1.twinx()
        ax2.plot(
            donnees["iteration"],
            donnees["val_score"],
            color="#F58518",
            linewidth=1.8,
            linestyle="--",
            label="Exactitude de validation (jeu d'arrêt anticipé)",
        )
        ax2.set_ylabel(
            "Exactitude de validation\n(jeu d'arrêt anticipé — 10 % du train)",
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
            label=f"Convergence à l'itération {stop_iter}",
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
    titre: str = "Courbe d'apprentissage (MLP)",
) -> plt.Figure:
    """Trace la courbe d'apprentissage du MLP."""
    fig, ax = plt.subplots(figsize=(10, 6))
    import numpy as np

    ax.plot(
        donnees["train_size"],
        donnees["train_mean"],
        "o-",
        label="Entraînement",
        color="#4C78A8",
    )
    ax.fill_between(
        donnees["train_size"],
        np.clip(donnees["train_mean"] - donnees["train_std"], 0, 1),
        np.clip(donnees["train_mean"] + donnees["train_std"], 0, 1),
        alpha=0.15,
        color="#4C78A8",
    )
    ax.plot(
        donnees["train_size"],
        donnees["val_mean"],
        "o-",
        label="Validation",
        color="#F58518",
    )
    ax.fill_between(
        donnees["train_size"],
        np.clip(donnees["val_mean"] - donnees["val_std"], 0, 1),
        np.clip(donnees["val_mean"] + donnees["val_std"], 0, 1),
        alpha=0.15,
        color="#F58518",
    )
    ax.set_title(titre)
    ax.set_xlabel("Taille du jeu d'entraînement", fontweight="bold")
    ax.set_ylabel("F1-Macro", fontweight="bold")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def tracer_heatmap_hidden_alpha(
    tableau_grid: pd.DataFrame,
    titre: str = "Tailles des couches cachées vs alpha",
) -> plt.Figure:
    """Trace la carte thermique ``hidden_layer_sizes`` x ``alpha``."""
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
    ax.set_ylabel("Tailles des couches cachées", fontweight="bold")
    fig.tight_layout()
    return fig
