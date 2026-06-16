"""
data.py
=============
Preprocessing, feature engineering (Label Encoding), and train/test split 
for the F1 qualifying dataset, using centralized configurations.
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
    COMPOUND_COLS
)

def preprocess_and_split(input_path=DATA_PATH_WIDE_WITH_FP):
    """
    Loads the dataset, restricts seasons, applies Label Encoding, 
    converts booleans, handles missing values with train medians, 
    removes drop columns, and saves the split CSVs.
    """
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Year": "year", "YEAR": "year"})

    # Restrict to configured seasons and drop NaNs in the Target Column
    df = df[df[SEASON_COL].isin(TRAIN_SEASONS + TEST_SEASONS)].copy()
    if TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL])

    # Apply Label Encoder to standard categorical columns
    le = LabelEncoder()
    for col in ENCODE_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            print(f"  [Encoder] Column '{col}' transformed into numeric values.")

    # Apply shared Label Encoder to Compound columns
    existing_compounds = [c for c in COMPOUND_COLS if c in df.columns]
    if existing_compounds:
        # Flatten all compound values to create a single consistent vocabulary
        all_compounds = pd.concat([df[c].astype(str) for c in existing_compounds]).unique()
        compound_le = LabelEncoder()
        compound_le.fit(all_compounds)
        
        for col in existing_compounds:
            df[col] = df[col].astype(str)
            df[col] = compound_le.transform(df[col])
            print(f"  [Encoder] Column '{col}' transformed with shared compound mapping.")

    # Convert boolean columns to integers (1 and 0)
    pure_bools = df.select_dtypes(include=['bool']).columns
    if len(pure_bools) > 0:
        df[pure_bools] = df[pure_bools].astype(int)
        print(f"  [Booleans] {len(pure_bools)} pure boolean columns converted to integers.")

    replace_map = {True: 1, False: 0, 'True': 1, 'False': 0}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Checks if the column actually contains the word True or False.
            if df[col].isin([True, False, 'True', 'False']).any():
                df[col] = df[col].replace(replace_map)
                
                # Convert to numeric (float) so that median imputation can fill in the NaNs afterwards.
                df[col] = pd.to_numeric(df[col], errors='ignore')
                print(f"  [Booleans] Mixed boolean column '{col}' converted to numeric (1/0/NaN).")

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

def load_and_split():
    """
    Load preprocessed train/test CSVs and return feature matrices,
    target vectors, and full DataFrames for metadata attachment.

    Runs preprocess_and_split() automatically if the files are missing.

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
    feature_cols = [c for c in train_df.columns if c not in exclude]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    print(f"Loaded  train: {X_train.shape}  |  test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, train_df, test_df


if __name__ == "__main__":
    preprocess_and_split()