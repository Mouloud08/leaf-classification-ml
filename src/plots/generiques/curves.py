"""Courbes ROC, précision-rappel, confiance et calibration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ...evaluation import (
    calculer_points_precision_rappel_ovr,
    calculer_points_roc_ovr,
    calculer_reliability_curve,
    extraire_scores_probabilites,
)


def tracer_courbes_roc_ovr(
    tableau_probabilites: pd.DataFrame,
    titre: str = "Courbes ROC (OvR)",
) -> plt.Axes:
    """Trace les courbes ROC micro et macro a partir des probabilites OOF."""
    points = calculer_points_roc_ovr(tableau_probabilites)
    figure, axe = plt.subplots(figsize=(8.8, 6.2))
    axe.plot(
        points["fpr_micro"],
        points["tpr_micro"],
        label=f"Moyenne micro (AUC = {points['auc_micro']:.3f})",
        color="#4C78A8",
        linewidth=2.2,
    )
    axe.plot(
        points["fpr_macro"],
        points["tpr_macro"],
        label=f"Moyenne macro (AUC = {points['auc_macro']:.3f})",
        color="#F58518",
        linewidth=2.2,
    )
    axe.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.2)
    axe.set_title(titre)
    axe.set_xlabel("Taux de faux positifs", fontweight="bold")
    axe.set_ylabel("Taux de vrais positifs", fontweight="bold")
    axe.legend(frameon=False)
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


def tracer_courbes_precision_rappel_ovr(
    tableau_probabilites: pd.DataFrame,
    titre: str = "Courbes précision-rappel (OvR)",
) -> plt.Axes:
    """Trace les courbes précision-rappel micro et macro."""
    points = calculer_points_precision_rappel_ovr(tableau_probabilites)
    figure, axe = plt.subplots(figsize=(8.8, 6.2))
    axe.plot(
        points["rappel_micro"],
        points["precision_micro"],
        label=f"Moyenne micro (AP = {points['ap_micro']:.3f})",
        color="#4C78A8",
        linewidth=2.2,
    )
    axe.plot(
        points["rappel_macro"],
        points["precision_macro"],
        label=f"Moyenne macro (AP = {points['ap_macro']:.3f})",
        color="#F58518",
        linewidth=2.2,
    )
    axe.set_title(titre)
    axe.set_xlabel("Rappel", fontweight="bold")
    axe.set_ylabel("Précision", fontweight="bold")
    axe.legend(frameon=False)
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


def tracer_histogramme_confiance(
    tableau_probabilites: pd.DataFrame,
    titre: str = "Distribution de la confiance",
) -> plt.Axes:
    """Trace la distribution de la confiance predictive maximale."""
    y_vrai, scores, _ = extraire_scores_probabilites(tableau_probabilites)
    max_proba = scores.max(axis=1)
    if "y_pred" in tableau_probabilites.columns:
        y_pred = tableau_probabilites["y_pred"].to_numpy()
    else:
        y_pred = scores.argmax(axis=1)
    correct = y_pred == y_vrai

    figure, axe = plt.subplots(figsize=(9.5, 5.8))
    axe.hist(
        max_proba[correct],
        bins=30,
        alpha=0.65,
        label="Correctes",
        color="#54A24B",
        edgecolor="white",
    )
    axe.hist(
        max_proba[~correct],
        bins=30,
        alpha=0.65,
        label="Erreurs",
        color="#E45756",
        edgecolor="white",
    )
    axe.axvline(
        x=1 / scores.shape[1],
        color="black",
        linestyle="--",
        alpha=0.6,
        label="Niveau du hasard",
    )
    axe.set_title(titre)
    axe.set_xlabel("Probabilité prédite maximale", fontweight="bold")
    axe.set_ylabel("Nombre de prédictions", fontweight="bold")
    axe.legend(frameon=False)
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


def tracer_reliability_diagram(
    tableau_probabilites: pd.DataFrame,
    titre: str = "Diagramme de fiabilité",
    n_bins: int = 10,
    strategy: str = "quantile",
) -> plt.Axes:
    """Trace un diagramme de fiabilité à partir de la confiance maximale."""
    donnees = calculer_reliability_curve(
        tableau_probabilites,
        n_bins=n_bins,
        strategy=strategy,
    )
    figure, axe = plt.subplots(figsize=(7.2, 6.2))
    axe.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#999999",
        label="Calibration parfaite",
    )
    axe.plot(
        donnees["confiance_moyenne"],
        donnees["precision_empirique"],
        marker="o",
        color="#4C78A8",
        linewidth=2,
        label="Observé",
    )
    axe.set_title(titre)
    axe.set_xlabel("Confiance prédite moyenne", fontweight="bold")
    axe.set_ylabel("Exactitude observée", fontweight="bold")
    axe.set_xlim(0, 1)
    axe.set_ylim(0, 1)
    axe.legend(frameon=False)
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe
