"""
config.py
=========
Central configuration for the F1 qualifying prediction study.
All model scripts and utilities import from here.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "f1_v3_predictive.csv"
DATA_PATH_WITH_FP = DATA_DIR / "qualifying_dataset_merged_with_fp.csv"
DATA_PATH_WIDE_WITH_FP = DATA_DIR / "qualifying_dataset_wide_with_fp.csv"
DATA_PATH_WITHOUT_FP = DATA_DIR / "qualifying_dataset_merged.csv"
FASTF1_WITH_FP_DIR = DATA_DIR / "fastF1" / "with_fp"
FASTF1_WITHOUT_FP_DIR = DATA_DIR / "fastF1" / "without_fp"
OUTPUT_DIR = ROOT / "outputs"
OUT_PREDS = OUTPUT_DIR / "predictions"
OUT_FIGS = OUTPUT_DIR / "figures"
OUT_RESULTS = OUTPUT_DIR / "results_table.csv"
# ── Output CSV Paths (Train/Test Split) ───────────────────────────
OUT_TRAIN = DATA_DIR / "qualifying_dataset_train.csv"
OUT_TEST = DATA_DIR / "qualifying_dataset_test.csv"

# Categorical columns to be individually encoded
ENCODE_COLS = ['Circuit', 'Driver', 'Team']

# Compound columns that must share the same vocabulary/encoder
COMPOUND_COLS = ['q1_compound', 'q2_compound', 'q3_compound']

# Create output directories if they do not exist
OUT_PREDS.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

# ── Experiment settings ───────────────────────────────────────────
TARGET_COL = "GridPosition"
SEASON_COL = "year"
TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASONS = [2025]
RANDOM_STATE = 42
CV_FOLDS = 5

# ── Feature exclusions ────────────────────────────────────────────
# Columns excluded from the feature matrix for all models.
DROP_COLS = [
    
]

# Metadata columns attached to enriched prediction outputs
META_COLS = [
    "raceId",
    "year",
    "round",
    "location",
    "country",
    "driverId",
    "constructorId",
]
