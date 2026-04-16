"""Fonctions de chargement, validation et split gele des donnees."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..artifacts.paths import global_paths
from ..config import DEFAULT_TRAIN_CSV, RANDOM_SEED, RESULTS_DIR

FROZEN_HOLDOUT_TEST_SIZE = 0.2


@dataclass(frozen=True)
class FrozenModelingSplit:
    """Sous-ensembles gelés pour la modélisation."""

    X_dev: pd.DataFrame
    y_dev: np.ndarray
    X_holdout: pd.DataFrame
    y_holdout: np.ndarray
    label_encoder: LabelEncoder
    metadata: dict[str, Any]

    @property
    def X_corroboration(self) -> pd.DataFrame:
        """Alias français de compatibilité pour le sous-ensemble de corroboration."""
        return self.X_holdout

    @property
    def y_corroboration(self) -> np.ndarray:
        """Alias français de compatibilité pour les labels de corroboration."""
        return self.y_holdout


def _resoudre_csv(csv_path: str | Path | None = None) -> Path:
    """Retourne le chemin absolu du CSV d'entrainement."""
    if csv_path is None:
        return DEFAULT_TRAIN_CSV
    return Path(csv_path)


def _charger_dataframe(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Charge et valide le dataset brut avec les colonnes `id` et `species`."""
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
    if dataframe["id"].duplicated().any():
        raise ValueError("La colonne 'id' doit contenir des identifiants uniques.")
    return dataframe


def _encoder_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Encode les labels et retire les colonnes non-features."""
    features = dataframe.drop(columns=["id", "species"]).copy()
    labels_bruts = dataframe["species"].copy()

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_bruts)
    return features, labels, label_encoder


def _lire_split_gelee(chemin: Path) -> dict[str, Any]:
    contenu = json.loads(chemin.read_text(encoding="utf-8"))
    for cle in ("train_ids", "holdout_ids"):
        if cle not in contenu or not isinstance(contenu[cle], list):
            raise ValueError(f"Artefact de split invalide: {cle} manquant.")
    return contenu


def _ecrire_split_gelee(chemin: Path, contenu: dict[str, Any]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(
        json.dumps(contenu, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _subset_dataframe_par_ids(
    dataframe: pd.DataFrame,
    ids: list[int | str],
) -> pd.DataFrame:
    """Retourne un sous-tableau dans l'ordre exact des IDs fournis."""
    indexe = dataframe.set_index("id", drop=False)
    try:
        subset = indexe.loc[ids].reset_index(drop=True)
    except KeyError as exc:
        raise ValueError("Artefact de split incompatible avec le CSV courant.") from exc
    return subset


def _charger_ou_creer_split_gelee(
    dataframe: pd.DataFrame,
    *,
    output_root: Path,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    """Charge ou cree le split gele `80/20` persiste par IDs."""
    chemins_globaux = global_paths(root=output_root)
    chemins_globaux.ensure_dirs()
    chemin_split = chemins_globaux.frozen_holdout_split_file

    if chemin_split.exists():
        contenu = _lire_split_gelee(chemin_split)
        ids_attendus = set(dataframe["id"].tolist())
        ids_train = set(contenu["train_ids"])
        ids_holdout = set(contenu["holdout_ids"])
        if ids_train & ids_holdout:
            raise ValueError(
                "Artefact de split invalide: chevauchement entre train_ids "
                "et holdout_ids."
            )
        if ids_train | ids_holdout != ids_attendus:
            raise ValueError("Artefact de split incompatible avec le CSV courant.")
        return contenu

    ids_dev, ids_holdout = train_test_split(
        dataframe["id"].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=dataframe["species"].tolist(),
    )
    contenu = {
        "random_seed": int(random_state),
        "test_size": float(test_size),
        "train_ids": list(ids_dev),
        "holdout_ids": list(ids_holdout),
        "n_dev": int(len(ids_dev)),
        "n_holdout": int(len(ids_holdout)),
    }
    _ecrire_split_gelee(chemin_split, contenu)
    return contenu


def charger_donnees(
    csv_path: str | Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Charge le dataset Kaggle complet et encode les labels."""
    dataframe = _charger_dataframe(csv_path)
    return _encoder_dataframe(dataframe)


def charger_donnees_modelisation(
    csv_path: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
    test_size: float = FROZEN_HOLDOUT_TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> FrozenModelingSplit:
    """Charge les sous-ensembles gelés de développement et de corroboration."""
    dataframe = _charger_dataframe(csv_path)
    sortie = Path(output_root) if output_root is not None else RESULTS_DIR
    split = _charger_ou_creer_split_gelee(
        dataframe,
        output_root=sortie,
        test_size=test_size,
        random_state=random_state,
    )

    df_dev = _subset_dataframe_par_ids(dataframe, split["train_ids"])
    df_holdout = _subset_dataframe_par_ids(dataframe, split["holdout_ids"])
    _, _, label_encoder = _encoder_dataframe(dataframe)

    X_dev = df_dev.drop(columns=["id", "species"]).copy()
    X_holdout = df_holdout.drop(columns=["id", "species"]).copy()
    y_dev = label_encoder.transform(df_dev["species"].to_numpy())
    y_holdout = label_encoder.transform(df_holdout["species"].to_numpy())

    metadata = {
        **split,
        "split_artifact": str(global_paths(root=sortie).frozen_holdout_split_file),
    }
    return FrozenModelingSplit(
        X_dev=X_dev,
        y_dev=y_dev,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        label_encoder=label_encoder,
        metadata=metadata,
    )
