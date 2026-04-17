"""Validation stricte des artefacts de metriques."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ..config import TIMING_POLICY
from .schema import (
    METRICS_SCHEMA_VERSION,
    NESTED_CV_INVALID_FIELDS,
    REQUIRED_METRIC_KEYS,
    SECONDARY_HOLDOUT_CORROBORATION_SCOPE,
    SECONDARY_HOLDOUT_ORIGIN_EXPLORATORY_REFIT,
    VALID_EVALUATION_STAGES,
    VALID_SECONDARY_HOLDOUT_CORROBORATION_SCOPES,
    VALID_SECONDARY_HOLDOUT_ORIGINS,
    VALID_SELECTION_PROTOCOLS,
)


class MetricsArtifactValidationError(ValueError):
    """Levée quand des artefacts de métriques ne respectent pas le schéma sûr."""


STAGE_MAP: dict[str, str] = {
    "simple": "baseline",
    "default": "baseline",
    "preprocessed": "improved_untuned",
    "scaled_only": "improved_untuned",
    "scaled_pca": "improved_untuned",
    "restricted_depth": "improved_untuned",
    "conservative_learning": "improved_untuned",
    "tuned": "tuned",
}

def derive_stage(variante: str) -> str:
    return STAGE_MAP.get(str(variante).lower(), "autre")


def derive_selection_protocol(variante: str, contenu: dict[str, Any]) -> str:
    protocole = contenu.get("selection_protocol")
    if protocole:
        return str(protocole)
    if str(variante).lower() == "tuned":
        return "nested_cv"
    return "simple_cv"


def _is_nan_like(valeur: Any) -> bool:
    if valeur is None:
        return True
    if isinstance(valeur, float):
        return math.isnan(valeur)
    return False


def _prefixer(source: str | None, message: str) -> str:
    return f"{source}: {message}" if source else message


def _valider_secondary_holdout(
    secondary_holdout: dict[str, Any],
    source: str | None = None,
) -> list[str]:
    """Valide la metadata de corroboration secondaire confirmatoire."""
    erreurs: list[str] = []
    if not secondary_holdout.get("enabled"):
        erreurs.append(
            _prefixer(
                source,
                "secondary_holdout.enabled doit etre True pour un artefact tuned",
            )
        )
        return erreurs

    if str(secondary_holdout.get("corroboration_scope", "")).strip() not in VALID_SECONDARY_HOLDOUT_CORROBORATION_SCOPES:
        erreurs.append(
            _prefixer(
                source,
                "secondary_holdout.corroboration_scope doit valoir "
                f"'{SECONDARY_HOLDOUT_CORROBORATION_SCOPE}'",
            )
        )

    if secondary_holdout.get("ranking_eligible") is not False:
        erreurs.append(
            _prefixer(
                source,
                "secondary_holdout.ranking_eligible doit etre False",
            )
        )

    if str(secondary_holdout.get("origin", "")).strip() not in VALID_SECONDARY_HOLDOUT_ORIGINS:
        erreurs.append(
            _prefixer(
                source,
                "secondary_holdout.origin doit valoir "
                f"'{SECONDARY_HOLDOUT_ORIGIN_EXPLORATORY_REFIT}'",
            )
        )

    if not str(secondary_holdout.get("reference_variant", "")).strip():
        erreurs.append(
            _prefixer(
                source,
                "secondary_holdout.reference_variant est obligatoire pour la corroboration",
            )
        )

    return erreurs


def valider_enregistrement_mesures(
    enregistrement: dict[str, Any],
    source: str | None = None,
) -> list[str]:
    """Valide un enregistrement de metriques contre le schema."""
    erreurs: list[str] = []

    for cle in sorted(REQUIRED_METRIC_KEYS):
        if cle not in enregistrement:
            erreurs.append(_prefixer(source, f"champ requis manquant: {cle}"))

    if erreurs:
        return erreurs

    protocole = str(enregistrement["selection_protocol"])
    if protocole not in VALID_SELECTION_PROTOCOLS:
        erreurs.append(
            _prefixer(source, f"selection_protocol invalide: {protocole}")
        )

    stade = str(enregistrement["evaluation_stage"])
    if stade not in VALID_EVALUATION_STAGES:
        erreurs.append(
            _prefixer(source, f"evaluation_stage invalide: {stade}")
        )

    variante = str(enregistrement["variante"]).lower()
    stade_attendu = derive_stage(variante)
    if stade != stade_attendu:
        erreurs.append(
            _prefixer(
                source,
                f"evaluation_stage incoherent pour la variante '{variante}': "
                f"{stade} != {stade_attendu}",
            )
        )

    if str(enregistrement["timing_policy"]) != TIMING_POLICY:
        erreurs.append(
            _prefixer(
                source,
                f"timing_policy incoherent: "
                f"{enregistrement['timing_policy']} != {TIMING_POLICY}",
            )
        )

    if variante == "tuned" and protocole != "nested_cv":
        erreurs.append(
            _prefixer(
                source,
                "une variante ajustée doit utiliser selection_protocol='nested_cv'",
            )
        )

    secondary_holdout = enregistrement.get("secondary_holdout")
    if secondary_holdout is not None:
        if not isinstance(secondary_holdout, dict):
            erreurs.append(
                _prefixer(
                    source,
                    "secondary_holdout doit etre un dictionnaire",
                )
            )
        elif variante != "tuned" or protocole != "nested_cv":
            erreurs.append(
                _prefixer(
                    source,
                    "secondary_holdout est reserve aux artefacts tuned en nested_cv",
                )
            )
        else:
            erreurs.extend(_valider_secondary_holdout(secondary_holdout, source=source))

    if protocole == "nested_cv":
        if "inner_splits" not in enregistrement:
            erreurs.append(
                _prefixer(
                    source,
                    "inner_splits manquant pour un artefact de validation "
                    "croisée imbriquée",
                )
            )
        for champ in NESTED_CV_INVALID_FIELDS:
            if not _is_nan_like(enregistrement.get(champ)):
                erreurs.append(
                    _prefixer(
                        source,
                        f"{champ} doit etre NaN/null pour un artefact de "
                        "validation croisée imbriquée",
                    )
                )
    elif "n_splits" not in enregistrement:
        erreurs.append(
            _prefixer(
                source,
                "n_splits manquant pour un artefact de validation croisée simple",
            )
        )

    return erreurs


def valider_artefacts_mesures(
    enregistrements: list[dict[str, Any]],
    sources: list[str] | None = None,
) -> list[str]:
    """Retourne la liste complete des erreurs de schema et protocole."""
    erreurs: list[str] = []
    for index, enregistrement in enumerate(enregistrements):
        source = None if sources is None else sources[index]
        erreurs.extend(valider_enregistrement_mesures(enregistrement, source=source))
    return erreurs


def normaliser_stades(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne `stade`.

    Les valeurs cibles sont `baseline`, `improved_untuned` et `tuned`.
    """
    if df.empty:
        return df.copy()
    resultat = df.copy()
    if "evaluation_stage" in resultat.columns:
        resultat["stade"] = resultat["evaluation_stage"].fillna("autre")
    elif "variante" in resultat.columns:
        resultat["stade"] = resultat["variante"].map(
            lambda v: STAGE_MAP.get(str(v).lower(), "autre")
        )
    else:
        resultat["stade"] = "autre"
    return resultat


def filtrer_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """Facade de compatibilite pour l'ancien pipeline `resultats`.

    Avec le layout canonique actuel, `charger_tableau_mesures_valides` ne charge
    deja que des artefacts compatibles. La fonction reste donc volontairement
    conservative et renvoie simplement une copie du tableau.
    """
    return df.copy()


def normaliser_mesures_pour_schema(
    nom_modele: str,
    variante: str,
    mesures: dict[str, Any],
) -> dict[str, Any]:
    """Normalise un dict de metriques pour le rendre conforme au schema."""
    contenu = dict(mesures)
    contenu["modele"] = nom_modele
    contenu["variante"] = variante
    contenu.setdefault("schema_version", METRICS_SCHEMA_VERSION)
    contenu["selection_protocol"] = derive_selection_protocol(variante, contenu)
    contenu["evaluation_stage"] = derive_stage(variante)
    contenu.setdefault("timing_policy", TIMING_POLICY)

    if contenu["selection_protocol"] == "nested_cv":
        for champ in NESTED_CV_INVALID_FIELDS:
            contenu[champ] = float("nan")

    return contenu
