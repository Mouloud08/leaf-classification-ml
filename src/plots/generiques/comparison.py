"""Graphiques de comparaison entre modeles et variantes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _format_label(text: str) -> str:
    """Convert internal metric or column names into presentation labels."""
    replacements = {
        "val_accuracy_mean": "Validation Accuracy",
        "val_f1_macro_mean": "Validation F1-Macro",
        "train_accuracy_mean": "Train Accuracy",
        "train_f1_macro_mean": "Train F1-Macro",
        "generalization_gap_accuracy": "Accuracy Gap",
        "generalization_gap_f1_macro": "F1-Macro Gap",
        "elapsed_seconds": "Elapsed Seconds",
        "modele": "Model",
        "variante": "Variant",
        "nombre_erreurs": "Number of Errors",
        "classe": "Class",
        "classe_vraie": "True Class",
        "classe_predite": "Predicted Class",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "support": "Support",
        "fpr": "False Positive Rate",
        "fnr": "False Negative Rate",
        "specificite": "Specificity",
        "param_modele__C": "C",
        "param_modele__gamma": "Gamma",
        "param_modele__alpha": "Alpha",
        "param_modele__penalty": "Penalty",
        "param_modele__n_estimators": "Number of Estimators",
        "param_modele__max_depth": "Max Depth",
        "param_modele__min_samples_leaf": "Min Samples per Leaf",
        "param_modele__criterion": "Criterion",
        "param_modele__learning_rate": "Learning Rate",
        "param_modele__hidden_layer_sizes": "Hidden Layer Sizes",
        "param_n_estimators": "Number of Estimators",
        "param_max_depth": "Max Depth",
        "param_min_samples_leaf": "Min Samples per Leaf",
        "param_criterion": "Criterion",
        "param_learning_rate": "Learning Rate",
        "param_hidden_layer_sizes": "Hidden Layer Sizes",
    }
    if text in replacements:
        return replacements[text]
    return text.replace("_", " ").title()


def tracer_barres_comparaison(
    tableau: pd.DataFrame,
    metrique: str = "val_f1_macro_mean",
    titre: str = "Performance Comparison",
) -> plt.Axes:
    """Plot a ranked model comparison bar chart."""
    if tableau.empty:
        raise ValueError("The results table is empty.")

    ordre_modeles = (
        tableau.groupby("modele")[metrique]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    figure, axe = plt.subplots(figsize=(10.8, 6.2))
    sns.barplot(
        data=tableau,
        x="modele",
        y=metrique,
        hue="variante",
        order=ordre_modeles,
        palette="mako",
        ax=axe,
    )
    axe.set_title(titre)
    axe.set_xlabel("Model", fontweight="bold")
    axe.set_ylabel(_format_label(metrique), fontweight="bold")
    axe.set_xticks(axe.get_xticks())
    axe.set_xticklabels(axe.get_xticklabels(), rotation=45, ha="right")
    sns.move_legend(
        axe,
        "upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        title="Variant",
    )
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


def tracer_comparaison_variantes(
    tableau: pd.DataFrame,
    colonnes_metriques: list[str],
    titre: str = "Variant Comparison",
) -> plt.Axes:
    """Compare multiple variants of the same model across metrics."""
    if tableau.empty:
        raise ValueError("The results table is empty.")

    donnees = tableau.copy()
    donnees = donnees.melt(
        id_vars=["variante"],
        value_vars=colonnes_metriques,
        var_name="metrique",
        value_name="score",
    )
    donnees["metrique"] = donnees["metrique"].map(_format_label)

    ordre_metriques = [
        _format_label(metric_name)
        for metric_name in colonnes_metriques
    ]
    donnees["metrique"] = pd.Categorical(
        donnees["metrique"],
        categories=ordre_metriques,
        ordered=True,
    )

    palette_reference = {
        "Validation Accuracy": "#4C78A8",
        "Validation F1-Macro": "#F58518",
        "Train Accuracy": "#72B7B2",
        "Train F1-Macro": "#E45756",
        "Accuracy Gap": "#54A24B",
        "F1-Macro Gap": "#B279A2",
    }
    palette = {
        metric_name: palette_reference.get(metric_name, color)
        for metric_name, color in zip(
            ordre_metriques,
            sns.color_palette("Set2", n_colors=max(len(ordre_metriques), 1)),
        )
    }

    figure, axe = plt.subplots(figsize=(9.6, 5.6))
    sns.barplot(
        data=donnees,
        x="variante",
        y="score",
        hue="metrique",
        palette=palette,
        ax=axe,
    )
    axe.set_title(titre)
    axe.set_xlabel("Variant")
    axe.set_ylabel("Score")
    axe.set_ylim(0, 1.0)
    sns.move_legend(
        axe,
        "upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        title="Metric",
    )
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


def tracer_temps_vs_performance(
    tableau: pd.DataFrame,
    metrique: str = "val_f1_macro_mean",
) -> plt.Axes:
    """Plot training time against a chosen performance metric."""
    if tableau.empty:
        raise ValueError("The results table is empty.")

    figure, axe = plt.subplots(figsize=(11, 6.5))
    sns.scatterplot(
        data=tableau,
        x="elapsed_seconds",
        y=metrique,
        hue="modele",
        style="variante",
        s=120,
        alpha=0.75,
        ax=axe,
    )
    axe.set_title("Training Time vs Performance")
    axe.set_xlabel("Total CV Time (seconds)", fontweight="bold")
    axe.set_ylabel(_format_label(metrique), fontweight="bold")
    sns.move_legend(
        axe,
        "upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
    )
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe
