"""Point d'entree principal : python -m src.cli.main"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..config import RESULTS_DIR
from ..data import charger_donnees_modelisation
from ..experiments import (
    ALL_MODEL_NAMES,
    StudySpec,
    charger_predictions_tuned_si_disponibles,
    construire_estimateur_variante,
    creer_cv,
    executer_etude,
    generer_comparaison_globale,
    generer_figures_generiques,
    generer_figures_specifiques,
    generer_manifeste,
    selectionner_meilleure_variante_untuned,
)

logger = logging.getLogger("src")


def _parse_advanced_figures(values: list[str] | None) -> dict[str, list[str]]:
    """Parse --advanced-figures mlp:tsne svm:learning_curve."""
    result: dict[str, list[str]] = {}
    if not values:
        return result
    for item in values:
        if ":" not in item:
            continue
        model, fig = item.split(":", 1)
        result.setdefault(model.lower(), []).append(fig)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.main",
        description="Execute l'etude complète de classification Leaf.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Modèles à exécuter (ex : perceptron svm mlp).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        dest="all_models",
        help="Executer tous les modeles.",
    )

    parser.add_argument(
        "--figures",
        choices=["core", "all", "none"],
        default="core",
        help="Mode de generation des figures (default: core).",
    )
    parser.add_argument(
        "--advanced-figures",
        nargs="*",
        metavar="MODEL:FIGURE",
        help="Figures avancees specifiques (ex: mlp:tsne svm:learning_curve).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Ne pas re-executer les modeles deja presents.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forcer la re-execution de tout.",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Sauter le tuning.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Ne generer aucune figure.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_DIR,
        help="Racine de sortie des resultats.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Afficher les logs detailles.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    models = ALL_MODEL_NAMES if args.all_models else [m.lower() for m in args.models]

    # Valider les noms de modeles
    unknown = set(models) - set(ALL_MODEL_NAMES)
    if unknown:
        logger.error("Modèles inconnus : %s", ", ".join(unknown))
        logger.info("Modèles disponibles : %s", ", ".join(ALL_MODEL_NAMES))
        return 1

    study = StudySpec(
        models=models,
        output_root=args.output_root,
        figures_mode=args.figures if not args.no_figures else "none",
        advanced_figures=_parse_advanced_figures(args.advanced_figures),
        skip_existing=args.skip_existing,
        force=args.force,
        no_tuning=args.no_tuning,
        no_figures=args.no_figures,
    )

    logger.info("Debut de l'etude — %d modele(s): %s", len(models), ", ".join(models))

    # Execution des modeles
    resultats = executer_etude(study)

    # Generation des figures
    if not study.no_figures and study.figures_mode != "none":
        split = charger_donnees_modelisation(output_root=study.output_root)
        X, y, label_encoder = split.X_dev, split.y_dev, split.label_encoder
        for result in resultats:
            if not result.success:
                continue
            logger.info("Figures pour %s...", result.model_name)

            # Reconstituer les predictions pour les figures generiques
            pred_df = charger_predictions_tuned_si_disponibles(result.model_name, study)
            if pred_df is not None:
                y_pred = pred_df["y_pred"].to_numpy()
            elif result.untuned_results:
                from sklearn.model_selection import cross_val_predict

                from ..experiments.model_specs import obtenir_model_study_spec

                spec = obtenir_model_study_spec(result.model_name)
                variante_spec = selectionner_meilleure_variante_untuned(
                    spec,
                    result.untuned_results,
                )
                est = construire_estimateur_variante(result.model_name, variante_spec)
                cv = creer_cv(study.n_splits, study.random_seed)
                y_pred = cross_val_predict(est, X, y, cv=cv)
            else:
                continue

            untuned_dicts = [r.to_dict() for r in result.untuned_results]
            tuned_dict = (
                result.tuning_result.to_dict() if result.tuning_result else None
            )

            generer_figures_generiques(
                result.model_name,
                untuned_dicts,
                tuned_dict,
                y,
                y_pred,
                label_encoder,
                study,
            )

            result.figures = generer_figures_specifiques(
                result.model_name,
                study,
            )

    # Rapport global
    logger.info("Generation du rapport global...")
    generer_comparaison_globale(resultats, study)
    generer_manifeste(study, resultats)

    # Resume console
    print("\n" + "=" * 60)
    print("RESUME DE L'ETUDE")
    print("=" * 60)
    for result in resultats:
        status = "OK" if result.success else f"ERREUR: {result.error}"
        best_f1 = "N/A"
        if result.tuning_result:
            best_f1 = f"{result.tuning_result.val_f1_macro_mean:.4f}"
        elif result.untuned_results:
            best_f1 = f"{max(r.val_f1_macro_mean for r in result.untuned_results):.4f}"
        print(f"  {result.model_name:25s} {status:10s} best_f1={best_f1}")
        if result.figures:
            n_figs = len(result.figures.core_generated) + len(
                result.figures.advanced_generated
            )
            print(f"    -> {n_figs} figures generees")
            if result.figures.errors:
                for fig, err in result.figures.errors.items():
                    print(f"    -> ERREUR figure {fig}: {err}")
    print("=" * 60)
    print(f"Résultats dans : {study.output_root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
