"""
data.py
=============
Data loading, preprocessing, and train/test splitting.
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_PATH,
    TARGET_COL,
    SEASON_COL,
    TRAIN_SEASONS,
    TEST_SEASONS,
    DROP_COLS,
)


def load_and_split(path=DATA_PATH):
    """
    Load the dataset, restrict to the ground-effect regulatory era
    (TRAIN_SEASONS + TEST_SEASONS), and return train/test arrays.

    Returns
    -------
    X_train, y_train : np.ndarray  training features and target
    X_test,  y_test  : np.ndarray  test features and target
    feature_cols     : list[str]   names of retained feature columns
    test_df          : pd.DataFrame original test rows (for enrichment)
    """
    df = pd.read_csv(path)
    df = df[df[SEASON_COL].isin(TRAIN_SEASONS + TEST_SEASONS)].copy()
    df = df.dropna(subset=[TARGET_COL])

    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    # Convert boolean columns to integer flags
    for col in feature_cols:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Fill remaining NaN with column median
    df[feature_cols] = df[feature_cols].fillna(
        df[feature_cols].median(numeric_only=True)
    )

    train_mask = df[SEASON_COL].isin(TRAIN_SEASONS)
    test_mask = df[SEASON_COL].isin(TEST_SEASONS)

    X_train = df.loc[train_mask, feature_cols].values.astype(float)
    y_train = df.loc[train_mask, TARGET_COL].values.astype(float)
    X_test = df.loc[test_mask, feature_cols].values.astype(float)
    y_test = df.loc[test_mask, TARGET_COL].values.astype(float)

    test_df = df[test_mask].reset_index(drop=True)

    print(f"  Features : {len(feature_cols)}")
    print(f"  Train    : {len(X_train)} samples  {TRAIN_SEASONS}")
    print(f"  Test     : {len(X_test)}  samples  {TEST_SEASONS}")

    return X_train, y_train, X_test, y_test, feature_cols, test_df
