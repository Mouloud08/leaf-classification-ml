"""Gestion du chargement des donnees du projet Leaf Classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class LeafDataLoader:
    """Charge et prepare les donnees tabulaires du dataset Leaf Classification.

    Le fichier source attendu est un CSV contenant :

    - une colonne `id` identifiant chaque echantillon;
    - une colonne `species` contenant l'etiquette de classe;
    - 192 variables numeriques decrivant les feuilles :
      64 variables `margin*`, 64 variables `shape*` et 64 variables `texture*`.

    Attributes:
        csv_path: Chemin vers le fichier CSV d'entrainement.
        label_encoder: Encodeur permettant de convertir les especes en labels entiers.
        feature_names: Liste des noms de variables apres suppression des colonnes cibles.
    """

    def __init__(self, csv_path: Path) -> None:
        """Initialise le chargeur de donnees.

        Args:
            csv_path: Chemin relatif ou absolu vers le fichier CSV source.
        """
        self.csv_path = csv_path
        self.label_encoder = LabelEncoder()
        self.feature_names: list[str] = []

    def charger_donnees(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Charge les donnees, retire les colonnes non predictives et encode les labels.

        Returns:
            Un tuple `(X, y)` ou `X` est un DataFrame de variables numeriques et `y`
            un tableau NumPy de labels encodes.
        """
        dataframe = pd.read_csv(self.csv_path)
        self.feature_names = [
            colonne
            for colonne in dataframe.columns
            if colonne not in {"id", "species"}
        ]

        X = dataframe[self.feature_names].copy()
        y = self.label_encoder.fit_transform(dataframe["species"])

        # TODO: ajouter une validation plus stricte du schema du CSV
        # TODO: verifier explicitement l'absence de valeurs manquantes

        return X, y

    def decoder_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Reconstruit les noms d'especes a partir des labels encodes.

        Args:
            y_encoded: Tableau de labels entiers.

        Returns:
            Tableau des etiquettes d'origine.
        """
        return self.label_encoder.inverse_transform(y_encoded)
