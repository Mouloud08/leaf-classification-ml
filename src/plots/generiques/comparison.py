"""Graphiques de comparaison entre modèles et variantes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _format_label(text: str) -> str:
    """Convertit des noms internes en libellés de présentation."""
    replacements = {
        "val_accuracy_mean": "Exactitude de validation",
        "val_f1_macro_mean": "F1-macro de validation",
        "train_accuracy_mean": "Exactitude d'entraînement",
        "train_f1_macro_mean": "F1-macro d'entraînement",
        "generalization_gap_accuracy": "Écart d'exactitude",
        "generalization_gap_f1_macro": "Écart de F1-macro",
        "elapsed_seconds": "Temps écoulé (s)",
        "modele": "Modèle",
        "variante": "Variante",
        "nombre_erreurs": "Nombre d'erreurs",
        "classe": "Classe",
        "classe_vraie": "Classe vraie",
        "classe_predite": "Classe prédite",
        "precision": "Précision",
        "recall": "Rappel",
        "f1": "Score F1",
        "support": "Support",
        "fpr": "Taux de faux positifs",
        "fnr": "Taux de faux négatifs",
        "specificite": "Spécificité",
        "param_modele__C": "C",
        "param_modele__gamma": "Gamma",
        "param_modele__alpha": "Alpha",
        "param_modele__penalty": "Pénalité",
        "param_modele__n_estimators": "Nombre d'estimateurs",
        "param_modele__max_depth": "Profondeur maximale",
        "param_modele__min_samples_leaf": "Échantillons min. par feuille",
        "param_modele__criterion": "Critère",
        "param_modele__learning_rate": "Taux d'apprentissage",
        "param_modele__hidden_layer_sizes": "Tailles des couches cachées",
        "param_n_estimators": "Nombre d'estimateurs",
        "param_max_depth": "Profondeur maximale",
        "param_min_samples_leaf": "Échantillons min. par feuille",
        "param_criterion": "Critère",
        "param_learning_rate": "Taux d'apprentissage",
        "param_hidden_layer_sizes": "Tailles des couches cachées",
    }
    if text in replacements:
        return replacements[text]
    return text.replace("_", " ").title()


def tracer_barres_comparaison(
    tableau: pd.DataFrame,
    metrique: str = "val_f1_macro_mean",
    titre: str = "Comparaison des performances",
) -> plt.Axes:
    """Trace un graphique en barres classant les modèles."""
    if tableau.empty:
        raise ValueError("Le tableau de résultats est vide.")

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
    axe.set_xlabel("Modèle", fontweight="bold")
    axe.set_ylabel(_format_label(metrique), fontweight="bold")
    axe.set_xticks(axe.get_xticks())
    axe.set_xticklabels(axe.get_xticklabels(), rotation=45, ha="right")
    sns.move_legend(
        axe,
        "upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        title="Variante",
    )
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


def tracer_corroboration_globale(
    tableau: pd.DataFrame,
    titre: str = "Corroboration secondaire sur le jeu de corroboration",
) -> plt.Axes:
    """Compare la référence et la variante ajustée sur le jeu de corroboration."""
    if tableau.empty:
        raise ValueError("Le tableau de résultats est vide.")

    donnees = tableau.copy()
    ordre_modeles = (
        donnees.sort_values(
            ["nested_cv_f1", "holdout_tuned_f1"],
            ascending=False,
        )["modele"]
        .astype(str)
        .tolist()
    )
    donnees = donnees.melt(
        id_vars=["modele"],
        value_vars=["holdout_reference_f1", "holdout_tuned_f1"],
        var_name="serie",
        value_name="score",
    )
    donnees["serie"] = donnees["serie"].map(
        {
            "holdout_reference_f1": "Référence de corroboration",
            "holdout_tuned_f1": "Variante ajustée",
        }
    )

    figure, axe = plt.subplots(figsize=(10.8, 6.2))
    sns.barplot(
        data=donnees,
        x="modele",
        y="score",
        hue="serie",
        order=ordre_modeles,
        palette={
            "Référence de corroboration": "#4C78A8",
            "Variante ajustée": "#F58518",
        },
        ax=axe,
    )
    axe.set_title(titre)
    axe.set_xlabel("Modèle", fontweight="bold")
    axe.set_ylabel("F1-macro sur le jeu de corroboration", fontweight="bold")
    axe.set_xticks(axe.get_xticks())
    axe.set_xticklabels(axe.get_xticklabels(), rotation=45, ha="right")

    axe.set_ylim(0.1, 1.0)

    sns.move_legend(
        axe,
        "upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        title="Série",
    )
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


tracer_corroboration_holdout_global = tracer_corroboration_globale


def tracer_comparaison_variantes(
    tableau: pd.DataFrame,
    colonnes_metriques: list[str],
    titre: str = "Comparaison des variantes",
) -> plt.Axes:
    """Compare plusieurs variantes d'un même modèle sur plusieurs métriques."""
    if tableau.empty:
        raise ValueError("Le tableau de résultats est vide.")

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
        "Exactitude de validation": "#4C78A8",
        "F1-macro de validation": "#F58518",
        "Exactitude d'entraînement": "#72B7B2",
        "F1-macro d'entraînement": "#E45756",
        "Écart d'exactitude": "#54A24B",
        "Écart de F1-macro": "#B279A2",
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
    axe.set_xlabel("Variante")
    axe.set_ylabel("Performance")
    min_val = donnees["score"].min()
    lower = min_val * 1.15 if min_val < 0 else 0
    axe.set_ylim(lower, 1.0)
    sns.move_legend(
        axe,
        "upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        title="Métrique",
    )
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


def tracer_comparaison_variantes_rapport(
    tableau: pd.DataFrame,
    titre: str = "Comparaison finale des variantes",
) -> plt.Axes:
    """Trace une comparaison finale par modèle incluant la variante ajustée.

    Le graphique conserve la logique visuelle simple des notebooks d'origine :
    des barres groupées par variante, une barre par métrique, sans surcharge.
    """
    if tableau.empty:
        raise ValueError("Le tableau de résultats est vide.")

    ordre_variantes = [
        "default",
        "scaled_only",
        "scaled_pca",
        "restricted_depth",
        "tuned",
    ]
    donnees = tableau.copy()
    donnees["variante"] = donnees["variante"].astype(str)
    donnees = donnees[donnees["variante"].isin(ordre_variantes)].copy()
    variantes_presentes = [v for v in ordre_variantes if v in set(donnees["variante"])]
    donnees = donnees.melt(
        id_vars=["variante"],
        value_vars=["val_f1_macro_mean", "val_accuracy_mean"],
        var_name="metrique",
        value_name="score",
    )
    donnees["metrique"] = donnees["metrique"].map(
        {
            "val_f1_macro_mean": "F1-macro",
            "val_accuracy_mean": "Exactitude",
        }
    )

    figure, axe = plt.subplots(figsize=(9.6, 5.6))
    sns.barplot(
        data=donnees,
        x="variante",
        y="score",
        hue="metrique",
        order=variantes_presentes,
        palette={"F1-macro": "#F58518", "Exactitude": "#4C78A8"},
        ax=axe,
    )

    axe.set_title(titre)
    axe.set_xlabel("Variante")
    axe.set_ylabel("Performance moyenne")
    axe.set_ylim(0.0, 1.0)
    sns.move_legend(
        axe,
        "upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        title="Metrique",
    )
    axe.tick_params(axis="x", rotation=0)
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


def tracer_temps_vs_performance(
    tableau: pd.DataFrame,
    metrique: str = "val_f1_macro_mean",
) -> plt.Axes:
    """Trace le temps d'entraînement en fonction d'une métrique de performance."""
    if tableau.empty:
        raise ValueError("Le tableau de résultats est vide.")

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
    axe.set_title("Temps d'entraînement vs performance")
    axe.set_xlabel("Temps total de CV (secondes)", fontweight="bold")
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
