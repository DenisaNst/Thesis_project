"""
Utility functions for standardizing and splitting drug-target interaction data
for machine learning model training and evaluation. Handles format normalization,
label generation from binding affinity, and temporal/random data partitioning.

Key functionality:
  - Load and standardize interaction data with flexible column naming
  - Convert pChEMBL binding affinity values to binary labels
  - Rename ChEMBL-style IDs (molecule_chembl_id, target_chembl_id) to standard form
  - Sort by binding affinity and remove duplicate drug-target pairs
  - Temporal splitting: partition by year for time-aware train/test splits
  - Fallback random splitting with stratification to maintain class balance

Temporal splitting strategy:
  If year information is available, splits based on cutoff year (train: year ≤ cutoff,
  test: year > cutoff). This simulates real-world scenarios where models must predict
  on future data. Falls back to random stratified split if insufficient year coverage.

Used in:
  - train_rf_timeslice.py: Time-aware model evaluation with year-based splits
  - similarity_based.py: Random split baseline for model comparison

Dependencies:
  - pandas: Data manipulation
  - sklearn.model_selection: Random stratified splitting
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_standardize_interactions(csv_path, positive_pchembl_threshold=6.0):
    df = pd.read_csv(csv_path)

    rename_map = {
        "molecule_chembl_id": "drug_id",
        "target_chembl_id": "target_id",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "label" not in df.columns:
        if "pchembl_value" in df.columns:
            df["label"] = (pd.to_numeric(df["pchembl_value"], errors="coerce") >= positive_pchembl_threshold).astype(int)
        else:
            df["label"] = 1

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "pchembl_value" in df.columns:
        df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
        df = df.sort_values(by=["pchembl_value"], ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["drug_id", "target_id"]).reset_index(drop=True)
    return df


def split_by_date(df, cutoff_year=2019, test_size=0.2, random_state=42):
    if "year" in df.columns and df["year"].notna().sum() > 0:
        train = df[df["year"] <= cutoff_year]
        test = df[df["year"] > cutoff_year]
        if len(train) > 0 and len(test) > 0:
            return train.reset_index(drop=True), test.reset_index(drop=True)

    stratify_col = df["label"] if df["label"].nunique() > 1 else None
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_col)
    return train.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    print("evaluation_protocol.py provides utility functions for pipeline evaluation.")
