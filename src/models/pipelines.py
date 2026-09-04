"""Construction de pipelines sklearn a partir des classes modeles."""

from __future__ import annotations

from sklearn.pipeline import Pipeline


def _get_modeles():
    """Import paresseux pour eviter l'import circulaire."""
    from . import MODELES

    return MODELES


def creer_pipeline(
    nom_modele: str,
    utiliser_pca: bool | None = None,
) -> Pipeline:
    """Cree le pipeline preprocesse recommande pour un modele.

    Si ``utiliser_pca`` vaut ``None``, la valeur par defaut du modele
    (``utiliser_pca_par_defaut``) est utilisee.
    """
    modeles = _get_modeles()
    cle = nom_modele.lower()
    if cle not in modeles:
        raise KeyError(f"Modele inconnu: {nom_modele}")
    return modeles[cle]().creer_pipeline(utiliser_pca=utiliser_pca)


def creer_estimateur(nom_modele: str):
    """Instancie l'estimateur sklearn brut d'un modele."""
    modeles = _get_modeles()
    cle = nom_modele.lower()
    if cle not in modeles:
        raise KeyError(f"Modele inconnu: {nom_modele}")
    return modeles[cle]().creer_estimateur()


def obtenir_grille_tuning(
    nom_modele: str,
    prefixer_modele: bool = True,
):
    """Retourne la grille de tuning du modele."""
    modeles = _get_modeles()
    cle = nom_modele.lower()
    if cle not in modeles:
        raise KeyError(f"Modele inconnu: {nom_modele}")

    grille = modeles[cle]().grille_tuning()
    if prefixer_modele:
        return grille

    return {
        parametre.removeprefix("modele__"): valeurs
        for parametre, valeurs in grille.items()
    }
