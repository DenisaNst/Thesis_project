import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
df = pd.read_csv(PROJECT_ROOT / "data/raw/chembl_pd_interactions.csv")
df = df.rename(columns={"molecule_chembl_id":"drug_id",
                         "target_chembl_id":"target_id"})
df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
df = df.sort_values("pchembl_value", ascending=False, na_position="last")
df = df.drop_duplicates(subset=["drug_id","target_id"]).reset_index(drop=True)

# Check positive rate per target
rates = df.groupby("target_id")["label"].agg(["mean","count"])
rates.columns = ["positive_rate","n_pairs"]
rates = rates.sort_values("positive_rate", ascending=False)
print("Positive rate per target:")
print(rates.to_string())
print(f"\nOverall mean: {rates['positive_rate'].mean():.3f}")
print(f"Std dev:      {rates['positive_rate'].std():.3f}")