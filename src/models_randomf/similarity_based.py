"""
How this script works :
1. Random Split: Divides the drug-target interactions into an 80/20 training and
   test set using a naive random split (stratified by label).
2. Chemical Similarity (Drugs): Uses RDKit to convert drug SMILES into Morgan
   Fingerprints, then calculates the max Tanimoto similarity between every training
   drug and the test set.
3. Sequence Similarity (Proteins): Uses the ESM2 embeddings to calculate the
   max Cosine similarity between every training target and the test set.
4. Aggressive Filtering: Iterates through various similarity thresholds (e.g.,
   removing any training drug that is >90% similar to a test drug).
5. Retraining & Evaluation: Retrains the Random Forest on these progressively
   smaller, structurally isolated training sets and evaluates on the original test
   set to measure how much AUC drops when the model can't rely on memorization.
6. Pareto Visualization: Generates a trade-off plot comparing AUC loss against
   training data reduction.

Thesis Relevance:
  This script generates the "Random split" data for Table II. It proves that even
  without temporal filtering, standard random splits artificially inflate predictive
  accuracy through structural memorization.
"""

from __future__ import annotations

import sys
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from evaluation import evaluation_protocol as eval_protocol
except ImportError:
    from src.evaluation import evaluation_protocol as eval_protocol

OUT_DIR = PROJECT_ROOT / "artifacts" / "rf_similarity_random"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    interactions = eval_protocol.load_and_standardize_interactions(
        str(PROJECT_ROOT / "data/raw/chembl_pd_interactions.csv")
    )

# For the time-slice + similarity_based approach, I modified the train/test split by year
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(interactions))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42,
        stratify=interactions["label"].values
    )
    train_int = interactions.iloc[train_idx].copy().reset_index(drop=True)
    test_int = interactions.iloc[test_idx].copy().reset_index(drop=True)
    print(f"  Train: {len(train_int):,} pairs | Test: {len(test_int):,} pairs")

    drug_emb = pd.read_csv(
        PROJECT_ROOT / "data/processed/chembl_drug_embeddings.csv"
    ).rename(columns={"molecule_chembl_id": "drug_id"})

    prot_emb = pd.read_csv(
        PROJECT_ROOT / "data/processed/protein_embeddings.csv"
    )

    smiles_df = pd.read_csv(
        PROJECT_ROOT / "data/raw/pd_molecule_smiles.csv"
    ).rename(columns={"molecule_chembl_id": "drug_id"})

    return train_int, test_int, drug_emb, prot_emb, smiles_df


def compute_drug_similarity(train_drug_ids, test_drug_ids, smiles_df):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        raise ImportError("RDKit required: pip install rdkit")

    smiles_map = dict(zip(smiles_df["drug_id"], smiles_df["smiles"]))

    def get_fp(drug_id):
        smi = smiles_map.get(drug_id)
        mol = Chem.MolFromSmiles(smi)
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    test_fps = []
    valid_test = []
    for did in test_drug_ids:
        fp = get_fp(did)
        if fp is not None:
            test_fps.append(fp)
            valid_test.append(did)
    print(f"  Test drugs with valid fingerprints: {len(valid_test):,}")

    max_sim = {}
    n_train = len(train_drug_ids)
    for i, did in enumerate(train_drug_ids):
        if i % 5000 == 0:
            print(f"  Processing training drug {i:,}/{n_train:,}...")
        fp = get_fp(did)
        if fp is None:
            max_sim[did] = 0.0
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, test_fps)
        max_sim[did] = float(max(sims)) if sims else 0.0

    print(f"  Drug similarity computed for {len(max_sim):,} training drugs")
    return max_sim


def compute_protein_similarity(train_target_ids, test_target_ids, prot_emb):
    emb_cols = [c for c in prot_emb.columns if c.startswith("target_emb_")]
    prot_map = {
        row["target_id"]: row[emb_cols].values.astype(np.float32)
        for _, row in prot_emb.iterrows()
    }

    # Normalise embeddings for cosine similarity
    test_vecs = np.array([
        prot_map[t] for t in test_target_ids if t in prot_map
    ])
    valid_test = [t for t in test_target_ids if t in prot_map]

    if len(test_vecs) == 0:
        return {t: 0.0 for t in train_target_ids}

    test_normed = normalize(test_vecs)

    max_sim = {}
    for tid in train_target_ids:
        if tid not in prot_map:
            max_sim[tid] = 0.0
            continue
        vec = normalize(prot_map[tid].reshape(1, -1))
        sims = (vec @ test_normed.T).flatten()
        max_sim[tid] = float(np.max(sims))


    sim_values = list(max_sim.values())
    print(f"  Max prot similarity: min={min(sim_values):.3f} "
          f"mean={np.mean(sim_values):.3f} "
          f"max={max(sim_values):.3f}")
    return max_sim


def prepare_feature_matrix(interactions_df, drug_emb, prot_emb):
    drug_cols = [c for c in drug_emb.columns if c.startswith("drug_emb_")]
    prot_cols = [c for c in prot_emb.columns if c.startswith("target_emb_")]

    merged = interactions_df.merge(
        drug_emb[["drug_id"] + drug_cols], on="drug_id", how="inner"
    ).merge(
        prot_emb[["target_id"] + prot_cols], on="target_id", how="inner"
    )

    feat_cols = drug_cols + prot_cols
    X = merged[feat_cols].to_numpy(dtype=np.float32)
    y = merged["label"].to_numpy(dtype=int)
    return merged, X, y, feat_cols


def run_experiment(
    train_int, test_int,
    drug_emb, prot_emb,
    drug_max_sim, prot_max_sim,
    drug_threshold, prot_threshold,
):
    filtered_train = train_int[
        (train_int["drug_id"].map(drug_max_sim).fillna(0) < drug_threshold) &
        (train_int["target_id"].map(prot_max_sim).fillna(0) < prot_threshold)
    ].copy()

    original_n = len(train_int)
    filtered_n = len(filtered_train)
    reduction  = 1 - filtered_n / original_n

    _, X_train, y_train, _ = prepare_feature_matrix(
        filtered_train, drug_emb, prot_emb
    )
    _, X_test,  y_test,  _ = prepare_feature_matrix(
        test_int, drug_emb, prot_emb
    )

    if len(X_train) < 50 or len(X_test) < 10:
        return None

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        min_samples_split=30,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    pos_idx = list(clf.classes_).index(1)
    y_prob  = clf.predict_proba(X_test)[:, pos_idx]

    if len(np.unique(y_test)) < 2:
        return None

    return {
        "drug_threshold":  drug_threshold,
        "prot_threshold":  prot_threshold,
        "train_pairs":     filtered_n,
        "train_reduction": float(reduction),
        "test_pairs":      len(X_test),
        "test_roc_auc":    float(roc_auc_score(y_test, y_prob)),
        "test_pr_auc":     float(average_precision_score(y_test, y_prob)),
    }


def plot_pareto(results_df, baseline_auc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              facecolor="#F8FAFC")
    fig.suptitle(
        "Similarity-Based Partitioning: Pareto Analysis",
        fontsize=13, fontweight="bold",
    )

    df = results_df.copy()
    df["delta_auc"]   = baseline_auc - df["test_roc_auc"]
    df["delta_pairs"] = df["train_reduction"] * 100

    ax1 = axes[0]
    ax1.set_facecolor("#F8FAFC")

    scatter = ax1.scatter(
        df["delta_auc"] * 100,
        df["delta_pairs"],
        c=df["test_roc_auc"],
        cmap="RdYlBu",
        vmin=0.5, vmax=baseline_auc,
        s=80, alpha=0.8, zorder=3,
    )
    plt.colorbar(scatter, ax=ax1, label="Test ROC-AUC")

    ax1.axvline(x=0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax1.axhline(y=0, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    ax1.scatter([0], [0], color="red", s=150, zorder=5,
                label=f"No filtering (AUC={baseline_auc:.3f})")

    ax1.set_xlabel("ΔAUC ROC (% drop from baseline)", fontsize=11)
    ax1.set_ylabel("ΔKnown Pairs (% reduction in training)", fontsize=11)
    ax1.set_title("Trade-off: AUC Loss vs Training Data Reduction", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.yaxis.grid(True, alpha=0.4)
    ax1.xaxis.grid(True, alpha=0.4)

    ax2 = axes[1]
    ax2.set_facecolor("#F8FAFC")

    # Group by drug threshold for cleaner lines
    for d_thresh in sorted(df["drug_threshold"].unique()):
        subset = df[df["drug_threshold"] == d_thresh].sort_values("prot_threshold")
        if len(subset) > 1:
            ax2.plot(
                subset["delta_pairs"],
                subset["test_roc_auc"],
                marker="o", markersize=5,
                label=f"Drug sim < {d_thresh}",
                linewidth=1.5, alpha=0.8,
            )

    ax2.axhline(y=baseline_auc, color="red", linewidth=2,
                linestyle="--", label=f"No filtering ({baseline_auc:.3f})")
    ax2.axhline(y=0.5, color="gray", linewidth=1,
                linestyle=":", label="Random (0.5)", alpha=0.6)

    ax2.set_xlabel("Training Pairs Removed (%)", fontsize=11)
    ax2.set_ylabel("Test ROC-AUC", fontsize=11)
    ax2.set_title("AUC vs Training Data Reduction\nby Drug Similarity Threshold",
                  fontsize=11)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.set_ylim(0.45, baseline_auc + 0.05)
    ax2.yaxis.grid(True, alpha=0.4)

    plt.tight_layout()
    out = FIG_DIR / "fig_similarity_partitioning_random.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    train_int, test_int, drug_emb, prot_emb, smiles_df = load_data()

    train_drugs   = train_int["drug_id"].unique().tolist()
    test_drugs    = test_int["drug_id"].unique().tolist()
    train_targets = train_int["target_id"].unique().tolist()
    test_targets  = test_int["target_id"].unique().tolist()

    print(f"  Unique train drugs:   {len(train_drugs):,}")
    print(f"  Unique test drugs:    {len(test_drugs):,}")
    print(f"  Unique train targets: {len(train_targets)}")
    print(f"  Unique test targets:  {len(test_targets)}")

    drug_max_sim = compute_drug_similarity(train_drugs, test_drugs, smiles_df)
    prot_max_sim = compute_protein_similarity(
        train_targets, test_targets, prot_emb
    )

    _, X_train_full, y_train_full, _ = prepare_feature_matrix(
        train_int, drug_emb, prot_emb
    )
    _, X_test_full,  y_test_full,  _ = prepare_feature_matrix(
        test_int, drug_emb, prot_emb
    )

    clf_base = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        min_samples_leaf=5, min_samples_split=30,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf_base.fit(X_train_full, y_train_full)
    pos_idx = list(clf_base.classes_).index(1)
    y_prob_base = clf_base.predict_proba(X_test_full)[:, pos_idx]
    baseline_auc = float(roc_auc_score(y_test_full, y_prob_base))
    print(f"\n  Baseline AUC (no filtering): {baseline_auc:.4f}")
    print(f"  Train pairs (no filtering):  {len(X_train_full):,}")

    drug_thresholds = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    prot_thresholds = [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]

    results = []
    total = len(drug_thresholds) * len(prot_thresholds)
    done  = 0

    for d_thresh in drug_thresholds:
        for p_thresh in prot_thresholds:
            done += 1
            result = run_experiment(
                train_int, test_int,
                drug_emb, prot_emb,
                drug_max_sim, prot_max_sim,
                d_thresh, p_thresh,
            )
            if result:
                results.append(result)
                print(
                    f"  [{done:3d}/{total}] drug<{d_thresh} prot<{p_thresh} "
                    f"→ train={result['train_pairs']:,} "
                    f"({result['train_reduction']*100:.1f}% removed) "
                    f"AUC={result['test_roc_auc']:.4f}"
                )
            else:
                print(f"  [{done:3d}/{total}] drug<{d_thresh} prot<{p_thresh} → skipped (too few pairs)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "similarity_results.csv", index=False)

    plot_pareto(results_df, baseline_auc)
    print("  RESULTS SUMMARY")
    print(f"  Baseline (no filtering): AUC {baseline_auc:.4f}")
    print(f"\n  Top results by AUC:")
    top = results_df.nlargest(5, "test_roc_auc")
    for _, row in top.iterrows():
        print(
            f"    drug<{row['drug_threshold']} prot<{row['prot_threshold']} "
            f"→ AUC {row['test_roc_auc']:.4f} "
            f"({row['train_reduction']*100:.1f}% pairs removed)"
        )

    print(f"\n  Most aggressive filtering:")
    bottom = results_df.nsmallest(3, "train_pairs")
    for _, row in bottom.iterrows():
        print(
            f"    drug<{row['drug_threshold']} prot<{row['prot_threshold']} "
            f"→ {row['train_pairs']:,} pairs "
            f"AUC {row['test_roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()