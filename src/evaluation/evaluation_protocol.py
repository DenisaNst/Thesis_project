import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def load_and_standardize_interactions(csv_path, positive_pchembl_threshold=6.0):
    df = pd.read_csv(csv_path)

    rename_map = {
        "molecule_chembl_id": "drug_id",
        "target_chembl_id": "target_id",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required_cols = ["drug_id", "target_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in interactions file: {missing}")

    if "label" not in df.columns:
        if "pchembl_value" in df.columns:
            df["label"] = (pd.to_numeric(df["pchembl_value"], errors="coerce") >= positive_pchembl_threshold).astype(int)
        else:
            df["label"] = 1

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Keep one row per (drug, target) pair; prefer the strongest pChEMBL where available.
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


def double_member_exclusion(train_df, test_df):
    train_drugs = set(train_df["drug_id"])
    train_targets = set(train_df["target_id"])
    filtered_test = test_df[
        (~test_df["drug_id"].isin(train_drugs))
        & (~test_df["target_id"].isin(train_targets))
    ]
    return filtered_test.reset_index(drop=True)


def check_structural_similarity(smiles1, smiles2):
    ms1 = Chem.MolFromSmiles(smiles1)
    ms2 = Chem.MolFromSmiles(smiles2)
    if not ms1 or not ms2:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(ms1, 2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(ms2, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def similarity_based_partitioning(train_df, test_df, threshold=0.7, max_train_refs=500):
    if "smiles" not in train_df.columns or "smiles" not in test_df.columns:
        return test_df.reset_index(drop=True)

    train_smiles = train_df["smiles"].dropna().astype(str).unique()[:max_train_refs]
    keep_rows = []
    for _, row in test_df.iterrows():
        smi = row.get("smiles")
        if not isinstance(smi, str):
            continue
        similarities = [check_structural_similarity(smi, train_smi) for train_smi in train_smiles]
        if not similarities or max(similarities) < threshold:
            keep_rows.append(row)

    if not keep_rows:
        return test_df.iloc[0:0].copy()
    return pd.DataFrame(keep_rows).reset_index(drop=True)


if __name__ == "__main__":
    print("evaluation_protocol.py provides utility functions for pipeline evaluation.")
