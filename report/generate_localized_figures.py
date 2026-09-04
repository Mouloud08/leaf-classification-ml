"""Regenerate figures whose embedded labels must be language-specific."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

REPORT_DIR = Path(__file__).resolve().parent
DATA_PATH = REPORT_DIR.parent / "data" / "raw" / "train.csv"
COLORS = ("#2563eb", "#14b8a6")


MODEL_RESULTS = {
    "figA2b_comparaison_perceptron.png": (
        "Perceptron",
        ("default", "scaled_only", "scaled_pca", "tuned"),
        (0.6979, 0.8228, 0.8057, 0.8085),
        (0.7436, 0.8460, 0.8346, 0.8409),
    ),
    "fig_lr_comparaison_variantes.png": (
        "Logistic regression",
        ("default", "scaled_only", "scaled_pca", "tuned"),
        (0.2591, 0.9798, 0.9784, 0.9787),
        (0.3106, 0.9861, 0.9848, 0.9848),
    ),
    "fig_svm_comparaison_variantes.png": (
        "RBF SVM",
        ("default", "scaled_only", "scaled_pca", "tuned"),
        (0.8389, 0.9712, 0.9712, 0.9756),
        (0.8561, 0.9785, 0.9785, 0.9823),
    ),
    "fig_rf_comparaison_variantes.png": (
        "Random forest",
        ("default", "restricted_depth", "tuned"),
        (0.9764, 0.9345, 0.9702),
        (0.9785, 0.9445, 0.9747),
    ),
    "fig_dt_comparaison_variantes.png": (
        "Decision tree",
        ("default", "restricted_depth", "tuned"),
        (0.5809, 0.1468, 0.4295),
        (0.6174, 0.1529, 0.4610),
    ),
    "fig_mlp_comparaison_variantes.png": (
        "MLP",
        ("default", "scaled_only", "scaled_pca", "tuned"),
        (0.1642, 0.9389, 0.9323, 0.9389),
        (0.1863, 0.9482, 0.9444, 0.9482),
    ),
}

HOLDOUT_RESULTS = {
    "fig_perceptron_corroboration_holdout.png": (
        "Perceptron",
        0.8815,
        0.8694,
        0.8939,
        0.8838,
    ),
    "fig_lr_corroboration_holdout.png": (
        "Logistic regression",
        0.9892,
        0.9892,
        0.9899,
        0.9899,
    ),
    "fig_svm_corroboration_holdout.png": ("RBF SVM", 0.9838, 0.9946, 0.9848, 0.9949),
    "fig_rf_corroboration_holdout.png": (
        "Random forest",
        0.9842,
        0.9842,
        0.9848,
        0.9848,
    ),
    "fig_mlp_corroboration_holdout.png": ("MLP", 0.9785, 0.9785, 0.9798, 0.9798),
}

FRENCH_MODEL_NAMES = {
    "Logistic regression": "Régression logistique",
    "RBF SVM": "SVM RBF",
    "Random forest": "Forêt aléatoire",
    "Decision tree": "Arbre de décision",
}


def global_model_comparison(language: str) -> None:
    if language == "fr":
        models = [
            "Régression logistique",
            "SVM RBF",
            "Forêt aléatoire",
            "MLP",
            "Perceptron",
            "Arbre de décision",
        ]
        title = "Comparaison confirmatoire des modèles (validation croisée imbriquée)"
        xlabel = "Score moyen"
        labels = ("F1-macro", "Exactitude")
    else:
        models = [
            "Logistic regression",
            "RBF SVM",
            "Random forest",
            "MLP",
            "Perceptron",
            "Decision tree",
        ]
        title = "Confirmatory model comparison (nested cross-validation)"
        xlabel = "Mean score"
        labels = ("Macro F1", "Accuracy")

    f1 = np.array([0.9787, 0.9756, 0.9702, 0.9389, 0.8085, 0.4295])
    f1_std = np.array([0.0198, 0.0207, 0.0140, 0.0322, 0.0523, 0.0546])
    accuracy = np.array([0.9848, 0.9823, 0.9747, 0.9482, 0.8409, 0.4610])
    accuracy_std = np.array([0.0146, 0.0151, 0.0089, 0.0279, 0.0411, 0.0627])

    y = np.arange(len(models))
    height = 0.34
    fig, ax = plt.subplots(figsize=(11, 6.6), constrained_layout=True)
    ax.barh(
        y - height / 2,
        f1,
        height,
        xerr=f1_std,
        capsize=3,
        color="#2563eb",
        label=labels[0],
    )
    ax.barh(
        y + height / 2,
        accuracy,
        height,
        xerr=accuracy_std,
        capsize=3,
        color="#14b8a6",
        label=labels[1],
    )
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlim(0.35, 1.03)
    ax.set_xlabel(xlabel)
    ax.set_title(title, weight="bold", pad=12)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    for row, (left, right) in enumerate(zip(f1, accuracy)):
        ax.text(
            left - 0.008,
            row - height / 2,
            f"{left:.4f}",
            va="center",
            ha="right",
            color="white",
            fontsize=8,
        )
        ax.text(
            right - 0.008,
            row + height / 2,
            f"{right:.4f}",
            va="center",
            ha="right",
            color="white",
            fontsize=8,
        )

    output = REPORT_DIR / "figures" / language / "fig02_barplot_comparatif_tuned.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def save(fig: plt.Figure, language: str, filename: str) -> None:
    output = REPORT_DIR / "figures" / language / filename
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def model_variant_comparisons(language: str) -> None:
    for filename, (english_name, variants, f1, accuracy) in MODEL_RESULTS.items():
        model_name = (
            FRENCH_MODEL_NAMES.get(english_name, english_name)
            if language == "fr"
            else english_name
        )
        title = (
            f"{model_name} — variantes exploratoires et variante ajustée"
            if language == "fr"
            else f"{model_name} — exploratory and tuned variants"
        )
        xlabel = "Variante" if language == "fr" else "Variant"
        ylabel = "Performance moyenne" if language == "fr" else "Mean performance"
        labels = (
            ("F1-macro", "Exactitude") if language == "fr" else ("Macro F1", "Accuracy")
        )
        x = np.arange(len(variants))
        width = 0.37
        fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
        ax.bar(x - width / 2, f1, width, color=COLORS[0], label=labels[0])
        ax.bar(x + width / 2, accuracy, width, color=COLORS[1], label=labels[1])
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title, ylim=(0, 1.04))
        ax.set_xticks(x, variants)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right")
        save(fig, language, filename)


def holdout_comparisons(language: str) -> None:
    series_labels = (
        ("Référence", "Ajustée") if language == "fr" else ("Reference", "Tuned")
    )
    metric_labels = (
        ("F1-macro", "Exactitude") if language == "fr" else ("Macro F1", "Accuracy")
    )
    for filename, (
        english_name,
        reference_f1,
        tuned_f1,
        reference_acc,
        tuned_acc,
    ) in HOLDOUT_RESULTS.items():
        model_name = (
            FRENCH_MODEL_NAMES.get(english_name, english_name)
            if language == "fr"
            else english_name
        )
        title = (
            f"{model_name} — corroboration secondaire"
            if language == "fr"
            else f"{model_name} — secondary holdout corroboration"
        )
        values = np.array([[reference_f1, reference_acc], [tuned_f1, tuned_acc]])
        x = np.arange(2)
        width = 0.36
        fig, ax = plt.subplots(figsize=(8.8, 5.5), constrained_layout=True)
        ax.bar(x - width / 2, values[0], width, color=COLORS[0], label=series_labels[0])
        ax.bar(x + width / 2, values[1], width, color=COLORS[1], label=series_labels[1])
        ax.set_xticks(x, metric_labels)
        ax.set_ylim(0.80, 1.01)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right")
        save(fig, language, filename)


def global_supporting_figures(language: str) -> None:
    english_models = [
        "Logistic regression",
        "RBF SVM",
        "Random forest",
        "MLP",
        "Perceptron",
        "Decision tree",
    ]
    models = (
        [FRENCH_MODEL_NAMES.get(name, name) for name in english_models]
        if language == "fr"
        else english_models
    )
    f1 = np.array([0.9787, 0.9756, 0.9702, 0.9389, 0.8085, 0.4295])
    times = np.array([77.05, 56.49, 250.11, 38.25, 29.79, 11.47])

    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    ax.scatter(times, f1, s=85, color="#2563eb")
    for x, y, model in zip(times, f1, models):
        ax.annotate(
            model, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=9
        )
    ax.set_xlabel(
        "Temps total de CV (secondes)"
        if language == "fr"
        else "Total CV time (seconds)"
    )
    ax.set_ylabel(
        "F1-macro de validation" if language == "fr" else "Validation macro F1"
    )
    ax.set_title(
        "Temps d'entraînement et performance"
        if language == "fr"
        else "Training time versus performance"
    )
    ax.grid(alpha=0.25)
    save(fig, language, "fig3b_temps_vs_performance.png")

    holdout_models_en = [
        "Logistic regression",
        "RBF SVM",
        "Random forest",
        "MLP",
        "Perceptron",
    ]
    holdout_models = (
        [FRENCH_MODEL_NAMES.get(name, name) for name in holdout_models_en]
        if language == "fr"
        else holdout_models_en
    )
    reference = np.array([0.9892, 0.9838, 0.9842, 0.9785, 0.8815])
    tuned = np.array([0.9892, 0.9946, 0.9842, 0.9785, 0.8694])
    x = np.arange(len(holdout_models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    ax.bar(
        x - width / 2,
        reference,
        width,
        color=COLORS[0],
        label="Référence" if language == "fr" else "Reference",
    )
    ax.bar(
        x + width / 2,
        tuned,
        width,
        color=COLORS[1],
        label="Ajustée" if language == "fr" else "Tuned",
    )
    ax.set_xticks(x, holdout_models, rotation=22, ha="right")
    ax.set_ylim(0.82, 1.01)
    ax.set_ylabel(
        "F1-macro sur le jeu de corroboration"
        if language == "fr"
        else "Holdout macro F1"
    )
    ax.set_title(
        "Corroboration secondaire par modèle"
        if language == "fr"
        else "Secondary holdout corroboration by model"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    save(fig, language, "fig_global_corroboration_holdout.png")


def eda_figures(language: str) -> None:
    data = pd.read_csv(DATA_PATH)
    feature_data = data.drop(columns=["id", "species"])

    counts = data["species"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(13, 5.8), constrained_layout=True)
    ax.bar(np.arange(len(counts)), counts.to_numpy(), color="#2563eb", width=0.82)
    ax.set_ylim(0, 11)
    ax.set_xlabel("Espèces" if language == "fr" else "Species")
    ax.set_ylabel(
        "Nombre d'observations" if language == "fr" else "Number of observations"
    )
    ax.set_title(
        "Répartition équilibrée des 99 espèces"
        if language == "fr"
        else "Balanced distribution across 99 species"
    )
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.25)
    save(fig, language, "fig01_distribution_classes.png")

    selections = {
        "margin": [
            "margin12",
            "margin24",
            "margin32",
            "margin35",
            "margin40",
            "margin46",
            "margin48",
            "margin5",
            "margin54",
            "margin56",
            "margin57",
            "margin6",
            "margin60",
            "margin63",
            "margin64",
        ],
        "shape": [
            "shape11",
            "shape12",
            "shape17",
            "shape23",
            "shape30",
            "shape38",
            "shape4",
            "shape45",
            "shape48",
            "shape49",
            "shape5",
            "shape51",
            "shape57",
            "shape58",
            "shape64",
        ],
        "texture": [
            "texture12",
            "texture19",
            "texture20",
            "texture22",
            "texture24",
            "texture29",
            "texture30",
            "texture35",
            "texture38",
            "texture40",
            "texture48",
            "texture5",
            "texture50",
            "texture53",
            "texture54",
        ],
    }
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    for ax, (family, columns) in zip(axes, selections.items()):
        ax.boxplot(
            [feature_data[column] for column in columns],
            tick_labels=columns,
            showfliers=True,
        )
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.set_ylabel("Valeur" if language == "fr" else "Value")
        ax.set_title(
            f"Boîtes à moustaches de 15 variables {family}"
            if language == "fr"
            else f"Boxplots of 15 {family} variables"
        )
        ax.grid(axis="y", alpha=0.25)
    save(fig, language, "figA11_boxplots_variables.png")

    corr = feature_data.corr().to_numpy()
    fig, ax = plt.subplots(figsize=(11.5, 10.5), constrained_layout=True)
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    for boundary in (63.5, 127.5):
        ax.axhline(boundary, color="black", lw=1.4)
        ax.axvline(boundary, color="black", lw=1.4)
    ax.set_xticks((31.5, 95.5, 159.5), ("margin", "shape", "texture"))
    ax.set_yticks((31.5, 95.5, 159.5), ("margin", "shape", "texture"))
    ax.set_title(
        "Matrice de corrélation des 192 variables"
        if language == "fr"
        else "Correlation matrix of all 192 features"
    )
    fig.colorbar(
        image,
        ax=ax,
        label="Corrélation" if language == "fr" else "Correlation",
        shrink=0.82,
    )
    save(fig, language, "figA12_heatmap_correlation.png")

    scaled = StandardScaler().fit_transform(feature_data)
    cumulative = np.cumsum(PCA().fit(scaled).explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    ax.plot(
        np.arange(1, cumulative.size + 1), cumulative, marker=".", ms=4, color="#2563eb"
    )
    ax.axhline(
        0.99,
        color="#dc2626",
        ls="--",
        label="Seuil 99 %" if language == "fr" else "99% threshold",
    )
    ax.axvline(116, color="#16a34a", ls="--", label="n_components = 116")
    ax.set_xlabel(
        "Nombre de composantes" if language == "fr" else "Number of components"
    )
    ax.set_ylabel(
        "Variance expliquée cumulée"
        if language == "fr"
        else "Cumulative explained variance"
    )
    ax.set_title(
        "Variance cumulée de la PCA"
        if language == "fr"
        else "PCA cumulative explained variance"
    )
    ax.set_ylim(0.2, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, language, "figA13_pca_variance_cumulative.png")

    reduced = PCA(n_components=50, random_state=42).fit_transform(scaled)
    embedding = TSNE(
        n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto"
    ).fit_transform(reduced)
    species_codes = pd.Categorical(data["species"]).codes
    fig, ax = plt.subplots(figsize=(10.8, 8.2), constrained_layout=True)
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=species_codes,
        cmap="gist_ncar",
        s=18,
        alpha=0.78,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(
        "Projection t-SNE 2D des 99 espèces"
        if language == "fr"
        else "2D t-SNE projection of all 99 species"
    )
    ax.grid(alpha=0.2)
    save(fig, language, "figA14_tsne_2d.png")


def numeric_diagnostics(language: str) -> None:
    if language == "fr":
        training = "F1-macro d'entraînement"
        validation = "F1-macro de validation"
    else:
        training = "Training macro F1"
        validation = "Validation macro F1"

    c_values = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    val_mean = np.array([0.6050, 0.9550, 0.9770, 0.9798, 0.9756, 0.9691])
    val_std = np.array([0.0361, 0.0099, 0.0176, 0.0173, 0.0171, 0.0143])
    train_mean = np.array([0.7410, 0.9952, 1.0, 1.0, 1.0, 1.0])
    fig, ax = plt.subplots(figsize=(9.8, 5.7), constrained_layout=True)
    ax.semilogx(c_values, val_mean, marker="o", color="#f97316", label=validation)
    ax.fill_between(
        c_values, val_mean - val_std, val_mean + val_std, color="#f97316", alpha=0.16
    )
    ax.semilogx(c_values, train_mean, marker="o", color="#2563eb", label=training)
    ax.set_xlabel(
        "C (inverse de la force de régularisation)"
        if language == "fr"
        else "C (inverse regularization strength)"
    )
    ax.set_ylabel("F1-macro" if language == "fr" else "Macro F1")
    ax.set_title(
        "Régression logistique — chemin de régularisation"
        if language == "fr"
        else "Logistic regression — regularization path"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    save(fig, language, "fig06_regularization_path.png")

    estimators = np.array([50, 100, 200, 400, 600, 800])
    oob = np.array([0.9205, 0.9508, 0.9659, 0.9735, 0.9773, 0.9760])
    fig, ax = plt.subplots(figsize=(9.8, 5.7), constrained_layout=True)
    ax.plot(estimators, oob, marker="o", color="#2563eb")
    ax.set_xlabel("Nombre d'arbres" if language == "fr" else "Number of trees")
    ax.set_ylabel("Performance OOB" if language == "fr" else "OOB performance")
    ax.set_title(
        "Forêt aléatoire — convergence OOB"
        if language == "fr"
        else "Random forest — OOB convergence"
    )
    ax.grid(alpha=0.25)
    save(fig, language, "fig10_oob_vs_n_estimators.png")

    depths = ["2", "5", "10", "15", "20", "30", "None"]
    train_mean = np.array([0.0179, 0.0774, 0.2265, 0.4012, 0.6904, 0.9707, 1.0])
    train_std = np.array([0.0035, 0.0185, 0.0456, 0.0458, 0.0963, 0.0310, 0.0])
    val_mean = np.array([0.0108, 0.0520, 0.1468, 0.2545, 0.4320, 0.5539, 0.5809])
    val_std = np.array([0.0037, 0.0204, 0.0277, 0.0343, 0.0443, 0.0395, 0.0425])
    x = np.arange(len(depths))
    fig, ax = plt.subplots(figsize=(9.8, 5.7), constrained_layout=True)
    ax.plot(x, train_mean, marker="o", color="#2563eb", label=training)
    ax.fill_between(
        x, train_mean - train_std, train_mean + train_std, color="#2563eb", alpha=0.14
    )
    ax.plot(x, val_mean, marker="o", color="#f97316", label=validation)
    ax.fill_between(
        x, val_mean - val_std, val_mean + val_std, color="#f97316", alpha=0.14
    )
    ax.set_xticks(x, depths)
    ax.set_xlabel("Profondeur maximale" if language == "fr" else "Maximum depth")
    ax.set_ylabel("F1-macro" if language == "fr" else "Macro F1")
    ax.set_title(
        "Arbre de décision — compromis profondeur/performance"
        if language == "fr"
        else "Decision tree — depth/performance trade-off"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    save(fig, language, "figA3_validation_curve_max_depth.png")


def reframe_existing(
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    crop: tuple[float, float, float, float],
    figsize: tuple[float, float] = (10.5, 6.3),
    mask_regions: tuple[tuple[float, float, float, float], ...] = (),
    legend: tuple[tuple[str, str, str], ...] = (),
    legend_location: str = "upper right",
) -> None:
    source = REPORT_DIR / "figures" / "fr" / filename
    image = plt.imread(source)
    height, width = image.shape[:2]
    left, top, right, bottom = crop
    cropped = image[
        int(top * height) : int(bottom * height), int(left * width) : int(right * width)
    ]
    cropped_height, cropped_width = cropped.shape[:2]
    for mask_left, mask_top, mask_right, mask_bottom in mask_regions:
        cropped[
            int(mask_top * cropped_height) : int(mask_bottom * cropped_height),
            int(mask_left * cropped_width) : int(mask_right * cropped_width),
        ] = 1.0
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.10, 0.12, 0.87, 0.79])
    ax.imshow(cropped)
    ax.axis("off")
    if legend:
        handles = [
            Line2D([0], [0], color=color, linestyle=style, linewidth=2.3, label=label)
            for label, color, style in legend
        ]
        ax.legend(handles=handles, loc=legend_location, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=14, y=0.98)
    if xlabel:
        fig.text(0.54, 0.025, xlabel, ha="center", fontsize=11, weight="bold")
    if ylabel:
        fig.text(
            0.015,
            0.52,
            ylabel,
            va="center",
            rotation="vertical",
            fontsize=11,
            weight="bold",
        )
    save(fig, "en", filename)


def reframe_english_diagnostics() -> None:
    diagnostic_specs = {
        "fig_perceptron_histogramme_fonction_decision.png": (
            "Perceptron — decision-function distribution (exploratory)",
            "Maximum decision-function score",
            "Number of predictions",
            (0.024, 0.055, 0.985, 0.955),
        ),
        "figA5_loss_curve.png": (
            "MLP — exploratory loss curve (complete refit)",
            "Iteration",
            "Training loss / validation accuracy",
            (0.023, 0.055, 0.96, 0.955),
        ),
    }
    for filename, (title, xlabel, ylabel, crop) in diagnostic_specs.items():
        if filename.startswith("fig_perceptron"):
            reframe_existing(
                filename,
                title,
                xlabel,
                ylabel,
                crop,
                mask_regions=((0.77, 0.0, 1.0, 0.15),),
                legend=(("Correct", "#76b36d", "-"), ("Errors", "#e57373", "-")),
            )
        else:
            reframe_existing(
                filename,
                title,
                xlabel,
                ylabel,
                crop,
                mask_regions=((0.28, 0.0, 0.78, 0.14),),
                legend=(
                    ("Training loss", "#4c78a8", "-"),
                    ("Convergence at iteration 54", "#888888", ":"),
                    ("Validation accuracy", "#f58518", "--"),
                ),
                legend_location="upper center",
            )

    confusion_titles = {
        "figA_confusion_perceptron_tuned.png": "Perceptron — normalized OOF confusion matrix",
        "figA_confusion_lr_tuned.png": "Logistic regression — normalized OOF confusion matrix",
        "figA_confusion_svm_tuned.png": "RBF SVM — normalized OOF confusion matrix",
        "figA_confusion_rf_tuned.png": "Random forest — normalized OOF confusion matrix",
        "figA_confusion_dt_tuned.png": "Decision tree — normalized OOF confusion matrix",
        "figA_confusion_mlp_tuned.png": "MLP — normalized OOF confusion matrix",
    }
    for filename, title in confusion_titles.items():
        reframe_existing(
            filename,
            title,
            "Predicted class",
            "True class",
            (0.02, 0.065, 0.91, 0.955),
            figsize=(11.2, 10.2),
        )

    top_titles = {
        "fig_perceptron_top12_confusions.png": "Perceptron — most frequent OOF confusions",
        "fig_lr_top12_confusions.png": "Logistic regression — most frequent OOF confusions",
        "fig_svm_top12_confusions.png": "RBF SVM — most frequent OOF confusions",
        "fig_rf_top12_confusions.png": "Random forest — most frequent OOF confusions",
        "fig_dt_top12_confusions.png": "Decision tree — most frequent OOF confusions",
        "fig_mlp_top12_confusions.png": "MLP — most frequent OOF confusions",
    }
    for filename, title in top_titles.items():
        reframe_existing(
            filename,
            title,
            "Number of errors",
            "True → predicted class pair",
            (0.03, 0.055, 0.995, 0.96),
        )


def main() -> None:
    for language in ("fr", "en"):
        eda_figures(language)
        model_variant_comparisons(language)
        holdout_comparisons(language)
        global_model_comparison(language)
        global_supporting_figures(language)
        numeric_diagnostics(language)
    reframe_english_diagnostics()


if __name__ == "__main__":
    main()
