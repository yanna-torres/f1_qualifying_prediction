"""
data.py
=============
Preprocessing, feature engineering (Label Encoding, Deltas), anti-leakage purges,
and train/test split for the F1 qualifying dataset, using centralized configurations.
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

import sys
from pathlib import Path

# Injeta a raiz do projeto no path para importar o config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_PATH_WIDE_WITH_FP,
    OUT_TRAIN,
    OUT_TEST,
    TARGET_COL,
    SEASON_COL,
    TRAIN_SEASONS,
    TEST_SEASONS,
    DROP_COLS,
    ENCODE_COLS,
    COMPOUND_COLS,
)
from utils.circuit_metadata import get_circuit_metadata_df


def apply_feature_engineering(df):
    """
    Transforms absolute times into Relative Deltas (%).
    A delta of 0.0 means the driver was the fastest in the session.
    A delta of 0.01 means the driver was 1% slower than the leader.
    """
    print("\n  [Feature Engineering] Calculating Free Practice Deltas...")

    for fp in ["FP1_s", "FP2_s", "FP3_s"]:
        if fp in df.columns:
            # Transform absolute zeros into NaN to avoid breaking the math
            df[fp] = df[fp].replace({0.0: np.nan})

            # Find the minimum time (Free Practice Pole) for each specific race
            min_times = df.groupby([SEASON_COL, "Round"])[fp].transform("min")

            # Calculate the percentage difference: (My_Time - Best_Time) / Best_Time
            delta_col_name = f"{fp}_Delta_pct"
            df[delta_col_name] = (df[fp] - min_times) / min_times

            # Remove the absolute time column to force the model to look only at the Delta
            df.drop(columns=[fp], inplace=True)
            print(
                f"    -> Created column {delta_col_name} (Absolute {fp} times removed)."
            )

    return df


def purge_data_leakage(df):
    """
    Removes all columns generated DURING or AFTER qualifying.
    This guarantees an honest predictive baseline.
    """
    print("\n  [Anti-Leakage] Purging future variables...")

    # Find all timing, speed, and tire columns from Q1, Q2, and Q3
    leakage_cols = [c for c in df.columns if c.startswith(("q1_", "q2_", "q3_"))]

    # Add result variables that act as the exam's answer key
    if "quali_session_reached" in df.columns:
        leakage_cols.append("quali_session_reached")

    df.drop(columns=leakage_cols, errors="ignore", inplace=True)
    print(
        f"    -> {len(leakage_cols)} qualifying columns dropped to prevent Data Leakage."
    )

    return df


def apply_circuit_metadata(df):
    """
    Merges circuit-level metadata (layout, speed character, track
    length, corner count, elevation change) onto the dataset, keyed
    by the raw 'Circuit' string column.

    Must run BEFORE Label Encoding is applied to 'Circuit', since the
    merge relies on matching the original circuit name strings against
    circuit_metadata.CIRCUIT_METADATA.

    The new categorical columns (circuit_layout, circuit_speed,
    circuit_character) are added to ENCODE_COLS dynamically here so
    they go through the same Label Encoder treatment as Driver/Team/
    Circuit later in preprocess_and_split().
    """
    print("\n  [Circuit Metadata] Merging circuit layout/speed/character features...")

    metadata_df = get_circuit_metadata_df()
    before_cols = set(df.columns)

    df = df.merge(metadata_df, on="Circuit", how="left")

    new_cols = [c for c in df.columns if c not in before_cols]
    missing = df[new_cols].isnull().any(axis=1).sum()

    print(f"    -> Added columns: {new_cols}")
    if missing > 0:
        print(
            f"    [WARNING] {missing} rows have no circuit metadata match "
            f"(unrecognised Circuit name). These rows will get NaN here "
            f"and be median/mode-imputed later, which is NOT ideal for "
            f"categorical columns -- check circuit_metadata.py for missing entries."
        )

    return df


def preprocess_and_split(input_path=DATA_PATH_WIDE_WITH_FP):
    """
    Loads the dataset, applies anti-leakage rules, generates deltas,
    applies Label Encoding, handles missing values, and saves the CSVs.

    Note: this function runs ONCE to produce the saved train/test CSVs
    (OUT_TRAIN / OUT_TEST). Feature ablations (see config.ABLATIONS)
    are applied later, in load_and_split(), so that switching between
    ablations does not require re-running this expensive step.
    """
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    df.rename(columns={"Year": "year", "YEAR": "year"}, inplace=True)

    # Restrict to configured seasons and drop NaNs in the Target Column
    df = df[df[SEASON_COL].isin(TRAIN_SEASONS + TEST_SEASONS)].copy()
    if TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL])

    # 1. PURGA DE DATA LEAKAGE (Deve ocorrer antes do Label Encoding)
    df = purge_data_leakage(df)

    # 2. CIRCUIT METADATA
    df = apply_circuit_metadata(df)

    # 3. FEATURE ENGINEERING (DELTAS)
    df = apply_feature_engineering(df)
    # ------------------------------------

    # Apply Label Encoder to standard categorical columns.
    # New circuit-metadata categorical columns (circuit_layout,
    # circuit_speed, circuit_character) are added here so they go
    # through the same encoding as Driver/Team/Circuit, without
    # mutating the imported ENCODE_COLS constant.
    CIRCUIT_CATEGORICAL_COLS = ["circuit_layout", "circuit_speed", "circuit_character"]
    le = LabelEncoder()
    # Filter to ensure we only try to encode columns that survived the purge
    cols_to_encode = [
        c for c in ENCODE_COLS + CIRCUIT_CATEGORICAL_COLS if c in df.columns
    ]
    for col in cols_to_encode:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        print(f"  [Encoder] Column '{col}' transformed into numeric values.")

    # Apply shared Label Encoder to Compound columns
    existing_compounds = [c for c in COMPOUND_COLS if c in df.columns]
    if existing_compounds:
        # Flatten all compound values to create a single consistent vocabulary
        all_compounds = pd.concat(
            [df[c].astype(str) for c in existing_compounds]
        ).unique()
        compound_le = LabelEncoder()
        compound_le.fit(all_compounds)

        for col in existing_compounds:
            df[col] = df[col].astype(str)
            df[col] = compound_le.transform(df[col])
            print(
                f"  [Encoder] Column '{col}' transformed with shared compound mapping."
            )

    # Convert boolean columns to integers (1 and 0)
    pure_bools = df.select_dtypes(include=["bool"]).columns
    if len(pure_bools) > 0:
        df[pure_bools] = df[pure_bools].astype(int)
        print(
            f"  [Booleans] {len(pure_bools)} pure boolean columns converted to integers."
        )

    replace_map = {True: 1, False: 0, "True": 1, "False": 0}

    for col in df.columns:
        if df[col].dtype == "object":
            # Checks if the column actually contains the word True or False.
            if df[col].isin([True, False, "True", "False"]).any():
                df[col] = df[col].replace(replace_map)

                # Convert to numeric (float) so that median imputation can fill in the NaNs afterwards.
                df[col] = pd.to_numeric(df[col], errors="ignore")
                print(
                    f"  [Booleans] Mixed boolean column '{col}' converted to numeric (1/0/NaN)."
                )

    # Split into Train and Test sets based on config
    train_mask = df[SEASON_COL].isin(TRAIN_SEASONS)
    test_mask = df[SEASON_COL].isin(TEST_SEASONS)

    # Drop configured columns before splitting
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    df = df[feature_cols]

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print(f"\nData Split:")
    print(f"  Train {TRAIN_SEASONS}: {len(train_df)} rows")
    print(f"  Test  {TEST_SEASONS}: {len(test_df)} rows")

    # Replace NaN values with medians (Strictly avoiding Data Leakage)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns

    # Calculate the median using only past knowledge (Train)
    train_medians = train_df[numeric_cols].median()

    # Apply this same median to both train and test sets
    train_df[numeric_cols] = train_df[numeric_cols].fillna(train_medians)
    test_df[numeric_cols] = test_df[numeric_cols].fillna(train_medians)
    print("  [Imputation] NaN values filled with the train set median.")

    # Save the CSV files separately using config paths
    os.makedirs(OUT_TRAIN.parent, exist_ok=True)

    train_df.to_csv(OUT_TRAIN, index=False)
    test_df.to_csv(OUT_TEST, index=False)

    print(f"\nFiles saved successfully:")
    print(f"  - {OUT_TRAIN}")
    print(f"  - {OUT_TEST}")


def load_and_split(extra_drop_cols=None):
    """
    Load preprocessed train/test CSVs and return feature matrices,
    target vectors, and full DataFrames for metadata attachment.

    Runs preprocess_and_split() automatically if the files are missing.

    Parameters
    ----------
    extra_drop_cols : list[str] or None
        Additional columns to exclude from the feature matrix, on top
        of DROP_COLS and TARGET_COL. Used for feature ablation studies
        (see config.ABLATIONS) — e.g. dropping FP1_s_Delta_pct/
        FP2_s_Delta_pct/FP3_s_Delta_pct to test performance without
        Free Practice data. Columns named here that are not present
        in the dataset are ignored (no error raised), so the same
        ablation list can be reused safely even if a column was
        renamed or removed upstream.

    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : np.ndarray
    train_df, test_df : pd.DataFrame  (full rows, for metadata attachment)
    """
    if not OUT_TRAIN.exists() or not OUT_TEST.exists():
        print("Preprocessed files not found — running preprocess_and_split().")
        preprocess_and_split()

    train_df = pd.read_csv(OUT_TRAIN)
    test_df = pd.read_csv(OUT_TEST)

    exclude = set([TARGET_COL] + DROP_COLS)
    if extra_drop_cols:
        exclude |= set(extra_drop_cols)

    feature_cols = [c for c in train_df.columns if c not in exclude]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    print(f"Loaded  train: {X_train.shape}  |  test: {X_test.shape}")
    if extra_drop_cols:
        actually_dropped = [
            c
            for c in extra_drop_cols
            if c in set(pd.read_csv(OUT_TRAIN, nrows=0).columns)
        ]
        print(f"  [Ablation] Extra columns excluded: {actually_dropped}")

    return X_train, X_test, y_train, y_test, train_df, test_df


if __name__ == "__main__":
    preprocess_and_split()
