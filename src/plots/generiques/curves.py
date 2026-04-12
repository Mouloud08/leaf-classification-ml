"""Courbes ROC, Precision-Recall, confiance et calibration."""

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
    titre: str = "ROC Curves (OvR)",
) -> plt.Axes:
    """Trace les courbes ROC micro et macro a partir des probabilites OOF."""
    points = calculer_points_roc_ovr(tableau_probabilites)
    figure, axe = plt.subplots(figsize=(8.8, 6.2))
    axe.plot(
        points["fpr_micro"],
        points["tpr_micro"],
        label=f"Micro-average (AUC = {points['auc_micro']:.3f})",
        color="#4C78A8",
        linewidth=2.2,
    )
    axe.plot(
        points["fpr_macro"],
        points["tpr_macro"],
        label=f"Macro-average (AUC = {points['auc_macro']:.3f})",
        color="#F58518",
        linewidth=2.2,
    )
    axe.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.2)
    axe.set_title(titre)
    axe.set_xlabel("False Positive Rate", fontweight="bold")
    axe.set_ylabel("True Positive Rate", fontweight="bold")
    axe.legend(frameon=False)
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


def tracer_courbes_precision_rappel_ovr(
    tableau_probabilites: pd.DataFrame,
    titre: str = "Precision-Recall Curves (OvR)",
) -> plt.Axes:
    """Trace les courbes precision-rappel micro et macro."""
    points = calculer_points_precision_rappel_ovr(tableau_probabilites)
    figure, axe = plt.subplots(figsize=(8.8, 6.2))
    axe.plot(
        points["rappel_micro"],
        points["precision_micro"],
        label=f"Micro-average (AP = {points['ap_micro']:.3f})",
        color="#4C78A8",
        linewidth=2.2,
    )
    axe.plot(
        points["rappel_macro"],
        points["precision_macro"],
        label=f"Macro-average (AP = {points['ap_macro']:.3f})",
        color="#F58518",
        linewidth=2.2,
    )
    axe.set_title(titre)
    axe.set_xlabel("Recall", fontweight="bold")
    axe.set_ylabel("Precision", fontweight="bold")
    axe.legend(frameon=False)
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


def tracer_histogramme_confiance(
    tableau_probabilites: pd.DataFrame,
    titre: str = "Confidence Distribution",
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
    axe.hist(max_proba[correct], bins=30, alpha=0.65, label="Correct", color="#54A24B", edgecolor="white")
    axe.hist(max_proba[~correct], bins=30, alpha=0.65, label="Errors", color="#E45756", edgecolor="white")
    axe.axvline(x=1 / scores.shape[1], color="black", linestyle="--", alpha=0.6, label="Chance level")
    axe.set_title(titre)
    axe.set_xlabel("Maximum Predicted Probability", fontweight="bold")
    axe.set_ylabel("Number of Predictions", fontweight="bold")
    axe.legend(frameon=False)
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe


def tracer_reliability_diagram(
    tableau_probabilites: pd.DataFrame,
    titre: str = "Reliability Diagram",
    n_bins: int = 10,
) -> plt.Axes:
    """Trace un reliability diagram a partir de la confiance max."""
    donnees = calculer_reliability_curve(tableau_probabilites, n_bins=n_bins)
    figure, axe = plt.subplots(figsize=(7.2, 6.2))
    axe.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Perfect calibration")
    axe.plot(
        donnees["confiance_moyenne"],
        donnees["precision_empirique"],
        marker="o",
        color="#4C78A8",
        linewidth=2,
        label="Observed",
    )
    axe.set_title(titre)
    axe.set_xlabel("Mean Predicted Confidence", fontweight="bold")
    axe.set_ylabel("Observed Accuracy", fontweight="bold")
    axe.set_xlim(0, 1)
    axe.set_ylim(0, 1)
    axe.legend(frameon=False)
    sns.despine(ax=axe)
    figure.tight_layout()
    return axe
