import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load both embeddings
esm2 = pd.read_csv(f"{PROJECT_ROOT}/data/processed/protein_embeddings.csv")
drkg = pd.read_csv(f"{PROJECT_ROOT}/data/processed/drkg_target_embeddings.csv")

# Align to common targets
common = sorted(set(esm2["target_id"]) & set(drkg["target_id"]))
esm2 = esm2[esm2["target_id"].isin(common)].sort_values("target_id")
drkg = drkg[drkg["target_id"].isin(common)].sort_values("target_id")

esm2_cols = [c for c in esm2.columns if c.startswith("target_emb_")]
drkg_cols = [c for c in drkg.columns if c.startswith("target_emb_")]

E = normalize(esm2[esm2_cols].values)
D = normalize(drkg[drkg_cols].values)

esm2_sim = E @ E.T
drkg_sim  = D @ D.T

ids = esm2["target_id"].tolist()
n   = len(ids)

# Find pairs where DRKG says "very similar" (sim > 0.7)
# and check what ESM2 says about the same pairs
print("Pairs where DRKG similarity > 0.7:")
print(f"{'Target 1':<20} {'Target 2':<20} {'DRKG sim':>10} {'ESM2 sim':>10} {'Agree?':>8}")
print("-" * 70)

agreements = []
for i in range(n):
    for j in range(i+1, n):
        d_sim = drkg_sim[i,j]
        e_sim = esm2_sim[i,j]
        if d_sim > 0.7:
            agree = "YES" if e_sim > 0.9 else "NO"
            agreements.append((d_sim, e_sim, agree))
            print(f"{ids[i]:<20} {ids[j]:<20} {d_sim:>10.3f} {e_sim:>10.3f} {agree:>8}")

print(f"\nAgreement rate: {sum(1 for _,_,a in agreements if a=='YES')}/{len(agreements)}")
print("\nPairs where ESM2 similarity > 0.98 (nearly identical):")
print(f"{'Target 1':<20} {'Target 2':<20} {'ESM2 sim':>10} {'DRKG sim':>10}")
print("-" * 60)
for i in range(n):
    for j in range(i+1, n):
        if esm2_sim[i,j] > 0.98:
            print(f"{ids[i]:<20} {ids[j]:<20} {esm2_sim[i,j]:>10.3f} {drkg_sim[i,j]:>10.3f}")