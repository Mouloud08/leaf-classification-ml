"""Fonctions pures de calcul de metriques (pas de CV, pas d'I/O)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize

CHANCE_LEVEL: float = 1.0 / 99


def _obtenir_labels_et_noms(
    y_vrai: np.ndarray,
    y_pred: np.ndarray | None = None,
    label_encoder: Any | None = None,
) -> tuple[np.ndarray, list[Any]]:
    if label_encoder is not None:
        labels = np.arange(len(label_encoder.classes_))
        noms = label_encoder.inverse_transform(labels).tolist()
        return labels, noms

    elements = [np.asarray(y_vrai)]
    if y_pred is not None:
        elements.append(np.asarray(y_pred))
    labels = np.unique(np.concatenate(elements))
    return labels, labels.tolist()


def calculer_metriques(y_vrai: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcule les metriques standards sur des predictions deja produites."""
    return {
        "accuracy": float(accuracy_score(y_vrai, y_pred)),
        "f1_macro": float(f1_score(y_vrai, y_pred, average="macro")),
    }


def calculer_metriques_par_classe(
    y_vrai: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: Any | None = None,
) -> pd.DataFrame:
    """Calcule les metriques one-vs-rest pour chaque classe."""
    labels, noms_classes = _obtenir_labels_et_noms(y_vrai, y_pred=y_pred, label_encoder=label_encoder)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_vrai,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    matrice = confusion_matrix(y_vrai, y_pred, labels=labels)
    total = matrice.sum()

    enregistrements: list[dict[str, Any]] = []
    for index, nom_classe in enumerate(noms_classes):
        vrais_positifs = float(matrice[index, index])
        faux_negatifs = float(matrice[index, :].sum() - vrais_positifs)
        faux_positifs = float(matrice[:, index].sum() - vrais_positifs)
        vrais_negatifs = float(total - vrais_positifs - faux_negatifs - faux_positifs)

        specificite = vrais_negatifs / (vrais_negatifs + faux_positifs) if (vrais_negatifs + faux_positifs) else 0.0
        fnr = faux_negatifs / (faux_negatifs + vrais_positifs) if (faux_negatifs + vrais_positifs) else 0.0
        fpr = faux_positifs / (faux_positifs + vrais_negatifs) if (faux_positifs + vrais_negatifs) else 0.0

        enregistrements.append(
            {
                "classe": nom_classe,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "fpr": float(fpr),
                "fnr": float(fnr),
                "specificite": float(specificite),
            }
        )

    return pd.DataFrame(enregistrements)


def resumer_metriques_par_classe(tableau_classes: pd.DataFrame) -> dict[str, float]:
    """Retourne un resume macro simple des metriques par classe."""
    colonnes = ["precision", "recall", "f1", "fpr", "fnr", "specificite"]
    resume = {}
    for colonne in colonnes:
        if colonne in tableau_classes.columns:
            resume[colonne] = float(tableau_classes[colonne].mean())
    return resume


def construire_matrice_confusion(
    y_vrai: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: Any | None = None,
    normalize: str | None = "true",
) -> pd.DataFrame:
    """Construit une matrice de confusion lisible, brute ou normalisee."""
    labels, noms_classes = _obtenir_labels_et_noms(y_vrai, y_pred=y_pred, label_encoder=label_encoder)
    matrice = confusion_matrix(
        y_vrai,
        y_pred,
        labels=labels,
        normalize=normalize,
    )
    dataframe = pd.DataFrame(matrice, index=noms_classes, columns=noms_classes)
    dataframe.index.name = "classe_vraie"
    dataframe.columns.name = "classe_predite"
    return dataframe


def extraire_scores_probabilites(
    tableau_probabilites: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extrait les labels vrais, le score multiclasses et les noms de colonnes proba."""
    colonnes_proba = [
        colonne for colonne in tableau_probabilites.columns if colonne.startswith("proba__")
    ]
    if not colonnes_proba:
        raise ValueError("Le tableau doit contenir au moins une colonne 'proba__*'.")
    y_vrai = tableau_probabilites["y_true"].to_numpy()
    scores = tableau_probabilites[colonnes_proba].to_numpy()
    return y_vrai, scores, colonnes_proba


def _inferer_labels_depuis_tableau_probabilites(
    tableau_probabilites: pd.DataFrame,
) -> np.ndarray:
    """Infere l'ordre des labels associe aux colonnes ``proba__*``.

    Pour les tableaux produits par ``probabilites_oof()``, l'ordre des colonnes
    suit l'ordre trie des labels de ``y_true`` quand aucun ``label_encoder``
    n'est fourni, et suit toujours l'ordre des labels encodes sinon.
    """
    y_vrai = tableau_probabilites["y_true"].to_numpy()
    labels = np.unique(y_vrai)
    colonnes_proba = [
        colonne for colonne in tableau_probabilites.columns if colonne.startswith("proba__")
    ]
    if len(labels) == len(colonnes_proba):
        return labels

    suffixes = [colonne.replace("proba__", "", 1) for colonne in colonnes_proba]
    return np.asarray(suffixes, dtype=object)


def calculer_top_k_accuracies(
    tableau_probabilites: pd.DataFrame,
    ks: tuple[int, ...] = (3, 5),
) -> dict[str, float]:
    """Calcule les top-k accuracy a partir d'un tableau de probabilites OOF."""
    y_vrai, scores, _ = extraire_scores_probabilites(tableau_probabilites)
    labels = _inferer_labels_depuis_tableau_probabilites(tableau_probabilites)
    resultats: dict[str, float] = {}
    # Toujours exposer top-1 pour compatibilite avec les notebooks historiques.
    ks_effectifs = tuple(dict.fromkeys((1, *ks)))
    for k in ks_effectifs:
        k_effectif = min(k, scores.shape[1])
        resultats[f"top_{k}_accuracy"] = float(
            top_k_accuracy_score(y_vrai, scores, k=k_effectif, labels=labels)
        )
    return resultats


def calculer_statistiques_confiance(tableau_probabilites: pd.DataFrame) -> dict[str, float]:
    """Retourne des indicateurs simples de confiance predictive."""
    y_vrai, scores, _ = extraire_scores_probabilites(tableau_probabilites)
    labels = _inferer_labels_depuis_tableau_probabilites(tableau_probabilites)
    max_proba = scores.max(axis=1)
    y_pred = labels[scores.argmax(axis=1)]
    correct = y_pred == y_vrai

    confiance_moyenne_erreurs = float(max_proba[~correct].mean()) if (~correct).any() else 0.0

    return {
        "confiance_moyenne": float(max_proba.mean()),
        "confiance_moyenne_correctes": float(max_proba[correct].mean()),
        "confiance_moyenne_erreurs": confiance_moyenne_erreurs,
        # Alias historique conserve pour compatibilite avec les notebooks.
        "confiance_erreurs": confiance_moyenne_erreurs,
        "proportion_erreurs_confiance_sup_0_5": float((max_proba[~correct] > 0.5).mean()) if (~correct).any() else 0.0,
        "proportion_erreurs_confiance_inf_0_3": float((max_proba[~correct] < 0.3).mean()) if (~correct).any() else 0.0,
    }


def calculer_brier_score_multiclasse(tableau_probabilites: pd.DataFrame) -> float:
    """Calcule le Brier score multiclasse a partir des probabilites OOF.

    Le Brier score mesure la qualite de calibration : plus il est bas,
    meilleures sont les probabilites.  On utilise la decomposition
    one-vs-all standard : mean(sum((p_k - y_k)^2)).
    """
    y_vrai, scores, _ = extraire_scores_probabilites(tableau_probabilites)
    labels = _inferer_labels_depuis_tableau_probabilites(tableau_probabilites)
    y_bin = label_binarize(y_vrai, classes=labels)
    if y_bin.shape[1] == 1:
        # Cas binaire
        y_bin = np.hstack([1 - y_bin, y_bin])
    return float(np.mean(np.sum((scores - y_bin) ** 2, axis=1)))


def calculer_reliability_curve(
    tableau_probabilites: pd.DataFrame,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Construit les points du reliability diagram a partir de la confiance max."""
    y_vrai, scores, _ = extraire_scores_probabilites(tableau_probabilites)
    labels = _inferer_labels_depuis_tableau_probabilites(tableau_probabilites)
    max_proba = scores.max(axis=1)
    y_pred = labels[scores.argmax(axis=1)]
    correct = (y_pred == y_vrai).astype(int)
    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be either 'uniform' or 'quantile'")
    fraction_positifs, moyenne_predite = calibration_curve(
        correct,
        max_proba,
        n_bins=n_bins,
        strategy=strategy,
    )
    return pd.DataFrame(
        {
            "confiance_moyenne": moyenne_predite,
            "precision_empirique": fraction_positifs,
        }
    )


def calculer_points_roc_ovr(tableau_probabilites: pd.DataFrame) -> dict[str, Any]:
    """Calcule les courbes ROC micro et macro en one-vs-rest."""
    y_vrai, scores, colonnes_proba = extraire_scores_probabilites(tableau_probabilites)
    classes = _inferer_labels_depuis_tableau_probabilites(tableau_probabilites)
    y_bin = label_binarize(y_vrai, classes=classes)

    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), scores.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    grille_fpr = np.linspace(0.0, 1.0, 200)
    tprs_classes = []
    for index in range(scores.shape[1]):
        if y_bin[:, index].sum() == 0:
            continue
        fpr_classe, tpr_classe, _ = roc_curve(y_bin[:, index], scores[:, index])
        tprs_classes.append(np.interp(grille_fpr, fpr_classe, tpr_classe))

    if not tprs_classes:
        raise ValueError("Impossible de calculer la courbe ROC macro sans classes positives.")

    tpr_macro = np.mean(tprs_classes, axis=0)
    # Macro AUC = mean of per-class AUCs (not AUC of the averaged curve)
    aucs_par_classe = [auc(grille_fpr, tpr) for tpr in tprs_classes]
    auc_macro = float(np.mean(aucs_par_classe))
    return {
        "fpr_micro": fpr_micro,
        "tpr_micro": tpr_micro,
        "auc_micro": float(auc_micro),
        "fpr_macro": grille_fpr,
        "tpr_macro": tpr_macro,
        "auc_macro": auc_macro,
        "classes": [colonne.replace("proba__", "", 1) for colonne in colonnes_proba],
    }


def calculer_points_precision_rappel_ovr(tableau_probabilites: pd.DataFrame) -> dict[str, Any]:
    """Calcule les courbes precision-rappel micro et macro en one-vs-rest."""
    y_vrai, scores, colonnes_proba = extraire_scores_probabilites(tableau_probabilites)
    classes = _inferer_labels_depuis_tableau_probabilites(tableau_probabilites)
    y_bin = label_binarize(y_vrai, classes=classes)

    precision_micro, rappel_micro, _ = precision_recall_curve(y_bin.ravel(), scores.ravel())
    ap_micro = average_precision_score(y_bin, scores, average="micro")

    grille_rappel = np.linspace(0.0, 1.0, 200)
    precisions_classes = []
    for index in range(scores.shape[1]):
        if y_bin[:, index].sum() == 0:
            continue
        precision_classe, rappel_classe, _ = precision_recall_curve(
            y_bin[:, index],
            scores[:, index],
        )
        precisions_classes.append(
            np.interp(grille_rappel, rappel_classe[::-1], precision_classe[::-1])
        )

    if not precisions_classes:
        raise ValueError("Impossible de calculer la courbe precision-rappel macro.")

    precision_macro = np.mean(precisions_classes, axis=0)
    ap_macro = average_precision_score(y_bin, scores, average="macro")
    return {
        "precision_micro": precision_micro,
        "rappel_micro": rappel_micro,
        "ap_micro": float(ap_micro),
        "precision_macro": precision_macro,
        "rappel_macro": grille_rappel,
        "ap_macro": float(ap_macro),
        "classes": [colonne.replace("proba__", "", 1) for colonne in colonnes_proba],
    }


def extraire_top_confusions(
    y_vrai: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: Any | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Extrait les paires de classes les plus confondues."""
    matrice = confusion_matrix(y_vrai, y_pred)
    np.fill_diagonal(matrice, 0)

    paires = []
    for index_ligne, ligne in enumerate(matrice):
        for index_colonne, valeur in enumerate(ligne):
            if valeur <= 0:
                continue
            paires.append((index_ligne, index_colonne, int(valeur)))

    paires = sorted(paires, key=lambda item: item[2], reverse=True)[:top_n]
    enregistrements = []
    for vrai, predit, compte in paires:
        vrai_label = vrai
        predit_label = predit
        if label_encoder is not None:
            vrai_label = label_encoder.inverse_transform([vrai])[0]
            predit_label = label_encoder.inverse_transform([predit])[0]
        enregistrements.append(
            {
                "classe_vraie": vrai_label,
                "classe_predite": predit_label,
                "nombre_erreurs": compte,
            }
        )
    return pd.DataFrame(enregistrements)
