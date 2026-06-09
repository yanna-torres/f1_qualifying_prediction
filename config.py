"""
config.py
=========
Central configuration for the F1 qualifying prediction study.
All model scripts and utilities import from here.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "f1_v3_predictive.csv"
OUT_PREDS = ROOT / "outputs" / "predictions"
OUT_FIGS = ROOT / "outputs" / "figures"
OUT_RESULTS = ROOT / "outputs" / "results_table.csv"

# Create output directories if they do not exist
OUT_PREDS.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

# ── Experiment settings ───────────────────────────────────────────
TARGET_COL = "quali_position"
SEASON_COL = "year"
TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASONS = [2025]
RANDOM_STATE = 42
CV_FOLDS = 5

# ── Feature exclusions ────────────────────────────────────────────
# Columns excluded from the feature matrix for all models.
DROP_COLS = [
    "raceId",
    "driverId",
    "constructorId",
    "circuitId",
    "location",
    "country",
    "nationality_driver",
    "nationality_constructor",
    "weather_label",
    # raw session splits already summarised by best_quali_time & gap_to_pole_pct
    "q1_seconds",
    "q2_seconds",
    "q3_seconds",
    # race grid may reflect post-qualifying penalties
    "grid",
    # race finish position → target leakage
    "positionOrder",
    TARGET_COL,
    SEASON_COL,
    "round",
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
