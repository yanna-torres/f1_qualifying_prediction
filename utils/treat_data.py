"""
merge_and_clean.py
------------------
Merges the four qualifying datasets (2022-2025) and normalises:
  - Team names  : AlphaTauri / RB / Racing Bulls  -> Racing Bulls
                  Alfa Romeo / Kick Sauber         -> Kick Sauber
  - Circuit names: strips the "FORMULA 1 <SPONSOR>" prefix
                  and the trailing four-digit year.

Usage
-----
  python merge_and_clean.py --fp          # datasets that include FP data
  python merge_and_clean.py --no-fp       # datasets without FP data
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FASTF1_WITH_FP_DIR, FASTF1_WITHOUT_FP_DIR, DATA_DIR

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Merge and clean F1 qualifying datasets.",
)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "--fp",
    dest="with_fp",
    action="store_true",
    help="Use datasets that include Free Practice (FP) features.",
)
group.add_argument(
    "--no-fp",
    dest="with_fp",
    action="store_false",
    help="Use datasets without Free Practice (FP) features.",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1. Resolve paths based on the chosen variant
# ---------------------------------------------------------------------------
if args.with_fp:
    SOURCE_DIR = FASTF1_WITH_FP_DIR
    FILENAME_TEMPLATE = "qualifying_dataset_with_fp_{year}.csv"
    OUTPUT_NAME = "qualifying_dataset_merged_with_fp.csv"
else:
    SOURCE_DIR = FASTF1_WITHOUT_FP_DIR
    FILENAME_TEMPLATE = "qualifying_dataset_{year}.csv"
    OUTPUT_NAME = "qualifying_dataset_merged.csv"

FILES = [SOURCE_DIR / FILENAME_TEMPLATE.format(year=y) for y in range(2022, 2026)]

print(f"Variant : {'with FP' if args.with_fp else 'without FP'}")
print(f"Source  : {SOURCE_DIR}")
print(f"Output  : {DATA_DIR / OUTPUT_NAME}\n")

# ---------------------------------------------------------------------------
# 2. Load and concatenate
# ---------------------------------------------------------------------------
df = pd.concat(
    [pd.read_csv(f) for f in FILES],
    ignore_index=True,
)

# ---------------------------------------------------------------------------
# 3. Normalise team names
# ---------------------------------------------------------------------------
TEAM_MAP = {
    "AlphaTauri": "Racing Bulls",
    "Alpha Tauri": "Racing Bulls",
    "RB": "Racing Bulls",
    "Alfa Romeo": "Kick Sauber",
}

df["Team"] = df["Team"].replace(TEAM_MAP)

# ---------------------------------------------------------------------------
# 4. Normalise circuit names
# ---------------------------------------------------------------------------
# Keywords that mark where the actual GP name begins (after any sponsor words).
GP_KEYWORDS = (
    r"(?:"
    r"GRAN PREMIO|GRANDE PR[EÊ]MIO|GRAND PRIX|GROSSER PREIS|MAGYAR|"
    r"AUSTRALIAN|BAHRAIN|SAUDI|CHINESE|JAPANESE|MIAMI|MONACO|CANADIAN|"
    r"SPANISH|AUSTRIAN|BRITISH|HUNGARIAN|BELGIAN|DUTCH|ITALIAN|SINGAPORE|"
    r"AZERBAIJAN|UNITED STATES|MEXICO|LAS VEGAS|QATAR|ABU DHABI"
    r")"
)

_circuit_re = re.compile(
    r"^FORMULA\s+1\s+(?:.*?\s+)?(" + GP_KEYWORDS + r".*?)\s*\d{4}\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)

# Manual overrides for edge cases the regex cannot resolve cleanly.
CIRCUIT_OVERRIDES = {
    # Monaco: regex stops at "GRAND PRIX" before capturing "DE MONACO".
    "GRAND PRIX": "GRAND PRIX DE MONACO",
    "MONACO": "GRAND PRIX DE MONACO",
    # 2022 Hungarian GP was announced in Hungarian.
    "MAGYAR NAGYDÍJ": "HUNGARIAN GRAND PRIX",
    "NAGYDÍJ": "HUNGARIAN GRAND PRIX",
    # 2022 Austrian GP was announced in German.
    "GROSSER PREIS VON ÖSTERREICH": "AUSTRIAN GRAND PRIX",
}


def clean_circuit(name: str) -> str:
    name = name.strip()
    m = _circuit_re.match(name)
    if m:
        cleaned = m.group(1).strip()
    else:
        # Fallback: strip "FORMULA 1 " prefix and trailing year.
        cleaned = re.sub(r"^FORMULA\s+1\s+", "", name, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*\d{4}\s*$", "", cleaned).strip()
    return CIRCUIT_OVERRIDES.get(cleaned, cleaned)


df["Circuit"] = df["Circuit"].apply(clean_circuit)

# ---------------------------------------------------------------------------
# 5. Sanity check
# ---------------------------------------------------------------------------
print("=== Teams ===")
print(sorted(df["Team"].unique()))

print("\n=== Circuits ===")
for c in sorted(df["Circuit"].unique()):
    print(" ", c)

print(f"\nTotal rows: {len(df):,}")

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
OUT = DATA_DIR / OUTPUT_NAME
df.to_csv(OUT, index=False)
print(f"\nSaved -> {OUT}")
