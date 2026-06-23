"""
config.py
=========
Central configuration for the F1 qualifying prediction study.
All model scripts and utilities import from here.
"""

from pathlib import Path
import numpy as np

np.random.seed(42)

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
OUT_PREDS = OUTPUT_DIR / "predictions"
OUT_FIGS = OUTPUT_DIR / "figures"
# ── Output CSV Paths (Train/Test Split) ───────────────────────────
OUT_TRAIN = DATA_DIR / "qualifying_dataset_train.csv"
OUT_TEST = DATA_DIR / "qualifying_dataset_test.csv"

# Categorical columns to be individually encoded
ENCODE_COLS = ["Circuit", "Driver", "Team"]

# Compound columns that must share the same vocabulary/encoder
COMPOUND_COLS = []

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
# Columns excluded from the feature matrix for all models, in every
# ablation. These are dropped at preprocess_and_split() time and
# never reach the saved train/test CSVs.
DROP_COLS = ["SubSession", "Compound"]

# Metadata columns attached to enriched prediction outputs
META_COLS = [
    "year",
    "Round",
    "Circuit",
    "Driver",
    "Team",
]

# ── Ablation studies ──────────────────────────────────────────────
# Each entry maps an ablation tag -> extra columns to exclude from
# the feature matrix at LOAD time (inside load_and_split()), on top
# of the columns already removed by DROP_COLS / the leakage purge.
#
# "full" is the baseline (no extra exclusions) and always exists.
# Adding a new ablation here makes it immediately available to every
# model pipeline and to run_all.py's --ablation flag.
ABLATIONS = {
    "full": [],
    "no_fp": ["FP1_s_Delta_pct", "FP2_s_Delta_pct", "FP3_s_Delta_pct"],
    "no_fp1_fp2": ["FP1_s_Delta_pct", "FP2_s_Delta_pct"],
    "no_fp1": ["FP1_s_Delta_pct"],
    "no_weather": [
        "AirTemp_C",
        "TrackTemp_C",
        "Humidity_pct",
        "Pressure_hPa",
        "WindSpeed_ms",
        "Rainfall",
        "Wet",
    ],
    "no_circuit_metadata": [
        "circuit_layout",
        "circuit_speed",
        "circuit_character",
        "track_length_km",
        "num_corners",
        "elevation_change_m",
    ],
}

# Composite ablation: derived from the two lists above so it can never
# drift out of sync if "no_fp" or "no_weather" is edited later.
ABLATIONS["no_weather_no_fp"] = list(
    dict.fromkeys(ABLATIONS["no_fp"] + ABLATIONS["no_weather"])
)

ABLATIONS["fp_only"] = [
    # Inverse of no_weather + no_sprint: isolates FP deltas as the
    # only non-identity signal, alongside Circuit/Driver/Team.
    "AirTemp_C",
    "TrackTemp_C",
    "Humidity_pct",
    "Pressure_hPa",
    "WindSpeed_ms",
    "Rainfall",
    "Wet",
    "IsSprintWeekend",
]

ABLATIONS["weather_only"] = [
    # Inverse of no_fp: isolates weather as the only non-identity
    # signal, removing FP deltas and sprint flag.
    "FP1_s_Delta_pct",
    "FP2_s_Delta_pct",
    "FP3_s_Delta_pct",
    "IsSprintWeekend",
]
