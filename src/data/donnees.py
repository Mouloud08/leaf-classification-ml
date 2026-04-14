"""Fonctions de chargement et de validation des donnees."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..config import DEFAULT_TRAIN_CSV


def _resoudre_csv(csv_path: str | Path | None = None) -> Path:
    """Retourne le chemin absolu du CSV d'entrainement."""
    if csv_path is None:
        return DEFAULT_TRAIN_CSV
    return Path(csv_path)


def charger_donnees(
    csv_path: str | Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Charge le dataset Kaggle et encode les labels."""
    chemin = _resoudre_csv(csv_path)
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable: {chemin}")

    dataframe = pd.read_csv(chemin)
    colonnes_requises = {"id", "species"}
    if not colonnes_requises.issubset(dataframe.columns):
        raise ValueError(
            "Le CSV doit contenir les colonnes 'id' et 'species'."
        )

    if dataframe.isna().any().any():
        raise ValueError("Le dataset contient des valeurs manquantes.")

    features = dataframe.drop(columns=["id", "species"]).copy()
    labels_bruts = dataframe["species"].copy()

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_bruts)

    return features, labels, label_encoder
