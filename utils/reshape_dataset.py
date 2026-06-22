"""
utils/reshape_dataset.py
------------------
Reshapes the qualifying dataset from long format (one row per driver
per sub-session) to wide format (one row per driver per weekend).

Sub-session features are pivoted into q1_*, q2_*, q3_* columns.
Drivers who did not reach a sub-session have NaN for those columns.
A quali_session_reached column (1, 2, or 3) is added.

Usage
-----
    python reshape_dataset.py --fp        # reshape the with-FP variant
    python reshape_dataset.py --no-fp     # reshape the without-FP variant
"""

import argparse
import sys
from pathlib import Path
from functools import reduce

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Reshape qualifying dataset to wide format."
)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "--fp", dest="with_fp", action="store_true", help="Reshape the with-FP variant."
)
group.add_argument(
    "--no-fp",
    dest="with_fp",
    action="store_false",
    help="Reshape the without-FP variant.",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
suffix = "_with_fp" if args.with_fp else ""
in_file = DATA_DIR / f"qualifying_dataset_merged{suffix}.csv"
out_file = DATA_DIR / f"qualifying_dataset_wide{suffix}.csv"

df = pd.read_csv(in_file)
print(f"Loaded  : {in_file}  ({len(df):,} rows)")

KEY = ["Year", "Round", "Driver"]

# ---------------------------------------------------------------------------
# 2. Columns by scope
# ---------------------------------------------------------------------------
# Weekend-level: identical across all sub-sessions for the same driver-weekend
WEEKEND_COLS = [
    "Year",
    "Round",
    "Circuit",
    "Driver",
    "Team",
    "AirTemp_C",
    "TrackTemp_C",
    "Humidity_pct",
    "Pressure_hPa",
    "WindSpeed_ms",
    "Rainfall",
    "Wet",
    "GridPosition",
]

FP_COLS = ["FP1_s", "FP2_s", "FP3_s", "IsSprintWeekend"]

# Sub-session-level: pivot these into q1_*, q2_*, q3_*
SESSION_COLS = [
    "LapTime_s",
    "Sector1_s",
    "Sector2_s",
    "Sector3_s",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "TyreLife",
    "Compound",
    "FreshTyre",
]

if args.with_fp:
    WEEKEND_COLS += FP_COLS

# ---------------------------------------------------------------------------
# 3. Build weekend-level base (one row per driver-weekend)
# ---------------------------------------------------------------------------
weekend = df.sort_values(KEY).groupby(KEY, as_index=False)[WEEKEND_COLS].first()

# quali_session_reached: highest sub-session the driver participated in
session_reached = (
    df.groupby(KEY)["SubSession_ord"]
    .max()
    .reset_index()
    .rename(columns={"SubSession_ord": "quali_session_reached"})
)
weekend = pd.merge(weekend, session_reached, on=KEY, how="left")

# ---------------------------------------------------------------------------
# 4. Pivot sub-session features
# ---------------------------------------------------------------------------
SESSION_MAP = {"Q1": "q1", "Q2": "q2", "Q3": "q3"}

parts = []
for session, prefix in SESSION_MAP.items():
    sub = df[df.SubSession == session][KEY + SESSION_COLS].copy()
    sub = sub.rename(columns={c: f"{prefix}_{c.lower()}" for c in SESSION_COLS})
    parts.append(sub)

pivoted = reduce(lambda a, b: pd.merge(a, b, on=KEY, how="outer"), parts)

# ---------------------------------------------------------------------------
# 5. Merge and finalise
# ---------------------------------------------------------------------------
wide = pd.merge(weekend, pivoted, on=KEY, how="left")

# Drop redundant KEY cols already captured in weekend cols
# (KEY cols remain because they were part of weekend)

print(f"Shape   : {wide.shape}")
print(f"Columns : {wide.columns.tolist()}")
print(f"\nMissing values:")
m = wide.isnull().sum()
print(m[m > 0].sort_values(ascending=False).to_string())

print(f"\nSample (LEC, Bahrain 2022):")
print(
    wide[(wide.Year == 2022) & (wide.Round == 1) & (wide.Driver == "LEC")].T.to_string()
)

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
wide.to_csv(out_file, index=False)
print(f"\nSaved   : {out_file}")
