"""
How this script works:
1. Alignment: Loads both the sequence-based (ESM2) and topology-based (DRKG)
   embeddings and aligns them so they contain the exact same Parkinson's targets
   in the exact same order.
2. Cosine Similarity: Computes the dot product of the normalized matrices, which
   mathematically yields the pairwise cosine similarity for every target against
   every other target.
3. Cross-Reference Analysis: Scans the similarity matrices to identify targets that
   DRKG considers highly similar and checks if ESM2 agrees, and vice-versa.

"""

import pandas as pd
from sklearn.preprocessing import normalize
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

esm2 = pd.read_csv(f"{PROJECT_ROOT}/data/processed/protein_embeddings.csv")
drkg = pd.read_csv(f"{PROJECT_ROOT}/data/processed/drkg_target_embeddings.csv")

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

for i in range(n):
    for j in range(i+1, n):
        if esm2_sim[i,j] > 0.98:
            print(f"{ids[i]:<20} {ids[j]:<20} {esm2_sim[i,j]:>10.3f} {drkg_sim[i,j]:>10.3f}")