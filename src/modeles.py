"""Registre central des modeles utilises dans les notebooks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .config import RANDOM_SEED


def _creer_perceptron() -> BaseEstimator:
    return Perceptron(random_state=RANDOM_SEED, max_iter=2000, tol=1e-3)


def _creer_regression_logistique() -> BaseEstimator:
    return LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=2500,
        solver="lbfgs",
    )


def _creer_svm() -> BaseEstimator:
    return SVC(kernel="rbf", probability=True)


def _creer_foret_aleatoire() -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )


def _creer_adaboost() -> BaseEstimator:
    return AdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0,
        random_state=RANDOM_SEED,
    )


def _creer_mlp() -> BaseEstimator:
    return MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=600,
        early_stopping=True,
        random_state=RANDOM_SEED,
    )


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "perceptron": {
        "nom_affiche": "Perceptron",
        "famille": "lineaire",
        "description": "Modele lineaire simple servant de baseline rapide.",
        "hypothese": "Le Perceptron donne une baseline correcte mais limitera la separabilite non lineaire.",
        "utiliser_pca_par_defaut": True,
        "factory": _creer_perceptron,
        "grille_tuning": {
            "modele__alpha": [0.0001, 0.001, 0.01],
            "modele__penalty": [None, "l2", "l1"],
        },
    },
    "regression_logistique": {
        "nom_affiche": "Regression logistique",
        "famille": "lineaire",
        "description": "Modele lineaire probabiliste robuste pour la multiclassification.",
        "hypothese": "La regularisation et le scaling devraient nettement ameliorer le score.",
        "utiliser_pca_par_defaut": True,
        "factory": _creer_regression_logistique,
        "grille_tuning": {
            "modele__C": [0.1, 1.0, 10.0, 100.0],
            "modele__solver": ["lbfgs", "saga"],
        },
    },
    "svm": {
        "nom_affiche": "SVM RBF",
        "famille": "noyau",
        "description": "Modele a noyau non lineaire potentiellement tres performant sur ce dataset.",
        "hypothese": "Le SVM RBF devrait mieux separer les zones ou les classes se chevauchent.",
        "utiliser_pca_par_defaut": True,
        "factory": _creer_svm,
        "grille_tuning": {
            "modele__C": [0.1, 1.0, 10.0, 100.0],
            "modele__gamma": ["scale", "auto", 0.01, 0.1],
        },
    },
    "foret_aleatoire": {
        "nom_affiche": "Foret aleatoire",
        "famille": "ensemble",
        "description": "Ensemble d'arbres capable de capturer des interactions non lineaires.",
        "hypothese": "La foret aleatoire devrait etre competitive sans aucun scaling.",
        "utiliser_pca_par_defaut": False,
        "factory": _creer_foret_aleatoire,
        "grille_tuning": {
            "modele__n_estimators": [100, 200, 400],
            "modele__max_depth": [None, 10, 20],
        },
    },
    "adaboost": {
        "nom_affiche": "AdaBoost",
        "famille": "ensemble",
        "description": "Boosting de classifieurs faibles pour raffiner la frontiere de decision.",
        "hypothese": "AdaBoost peut gagner sur les frontieres complexes mais sera sensible au bruit.",
        "utiliser_pca_par_defaut": False,
        "factory": _creer_adaboost,
        "grille_tuning": {
            "modele__n_estimators": [50, 100, 200],
            "modele__learning_rate": [0.01, 0.1, 1.0],
        },
    },
    "mlp": {
        "nom_affiche": "MLP",
        "famille": "reseau_de_neurones",
        "description": "Reseau dense multicouche adapte aux frontieres tres non lineaires.",
        "hypothese": "Le MLP devrait profiter fortement du scaling et du tuning.",
        "utiliser_pca_par_defaut": True,
        "factory": _creer_mlp,
        "grille_tuning": {
            "modele__hidden_layer_sizes": [(128,), (128, 64), (256,)],
            "modele__alpha": [0.0001, 0.001, 0.01],
        },
    },
}


def noms_modeles() -> list[str]:
    """Retourne les identifiants normalises des modeles."""
    return list(MODEL_REGISTRY.keys())


def obtenir_definition_modele(nom_modele: str) -> dict[str, Any]:
    """Retourne les metadonnees d'un modele."""
    cle = nom_modele.lower()
    if cle not in MODEL_REGISTRY:
        raise KeyError(f"Modele inconnu: {nom_modele}")
    definition = deepcopy(MODEL_REGISTRY[cle])
    definition.pop("factory", None)
    return definition


def creer_modele_simple(nom_modele: str) -> BaseEstimator:
    """Instancie la baseline brute d'un modele."""
    cle = nom_modele.lower()
    if cle not in MODEL_REGISTRY:
        raise KeyError(f"Modele inconnu: {nom_modele}")
    return MODEL_REGISTRY[cle]["factory"]()


def initialiser_grille_tuning(nom_modele: str) -> dict[str, list[Any]]:
    """Retourne la grille de tuning boilerplate du modele."""
    cle = nom_modele.lower()
    if cle not in MODEL_REGISTRY:
        raise KeyError(f"Modele inconnu: {nom_modele}")
    return deepcopy(MODEL_REGISTRY[cle]["grille_tuning"])
