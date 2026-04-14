"""Configuration centrale du projet Leaf Classification."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT_DIR / "results"

DEFAULT_TRAIN_CSV = RAW_DATA_DIR / "train.csv"

RANDOM_SEED = 42
N_SPLITS = 5
INNER_TUNING_SPLITS = 3
EXPLORATORY_TUNING_SPLITS = 3
PRIMARY_METRIC = "f1_macro"
# La PCA conserve 99% de la variance expliquee afin de rester valide
# dans chaque pli de validation croisee.
PCA_EXPLAINED_VARIANCE = 0.99
CV_N_JOBS = 1
TIMING_POLICY = f"cv_n_jobs={CV_N_JOBS}"
