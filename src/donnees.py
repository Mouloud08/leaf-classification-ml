"""Fonctions de chargement et de validation des donnees."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import DEFAULT_TRAIN_CSV


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


class LeafDataLoader:
    """Compatibilite orientee objet pour le notebook EDA existant."""

    def __init__(self, csv_path: str | Path | None = None) -> None:
        self.csv_path = _resoudre_csv(csv_path)
        self.label_encoder = LabelEncoder()
        self.feature_names: list[str] = []

    def charger_donnees(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Charge les donnees et memorise les noms de colonnes."""
        dataframe = pd.read_csv(self.csv_path)
        self.feature_names = [
            colonne
            for colonne in dataframe.columns
            if colonne not in {"id", "species"}
        ]
        self.label_encoder.fit(dataframe["species"])
        X = dataframe[self.feature_names].copy()
        y = self.label_encoder.transform(dataframe["species"])
        return X, y

    def decoder_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode les classes numeriques vers les noms d'especes."""
        return self.label_encoder.inverse_transform(y_encoded)

