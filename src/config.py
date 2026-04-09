"""Configuration centrale du projet Leaf Classification."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
GRID_SEARCH_DIR = RESULTS_DIR / "grid_search"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"
SAVED_MODELS_DIR = RESULTS_DIR / "models"

DEFAULT_TRAIN_CSV = RAW_DATA_DIR / "train.csv"

RANDOM_SEED = 42
N_SPLITS = 5
PRIMARY_METRIC = "f1_macro"
SECONDARY_METRIC = "accuracy"
PCA_COMPONENTS = 116
CV_N_JOBS = 1
SUPPORTED_VARIANTS = ("simple", "preprocessed", "tuned_placeholder")
