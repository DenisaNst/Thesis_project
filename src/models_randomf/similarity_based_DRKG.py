"""
This script is similar to similarity_based.py, but instead of using ESM2,
is using DRKG TransE embeddings.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from evaluation import evaluation_protocol as eval_protocol
except ImportError:
    from src.evaluation import evaluation_protocol as eval_protocol

OUT_DIR = PROJECT_ROOT / "artifacts" / "rf_similarity_drkg_random2"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Initialize with time-slice
# def load_data():
#     interactions = eval_protocol.load_and_standardize_interactions(
#         str(PROJECT_ROOT / "data/raw/chembl_pd_interactions.csv")
#     )
#
#     CUTOFF = 2018
#     train_int = interactions[interactions["year"] <= CUTOFF].copy()
#     test_int = interactions[interactions["year"] > CUTOFF].copy()
#
#     drug_emb = pd.read_csv(
#         PROJECT_ROOT / "data/processed/chembl_drug_embeddings.csv"
#     ).rename(columns={"molecule_chembl_id": "drug_id"})
#
#     prot_emb = pd.read_csv(
#         PROJECT_ROOT / "data/processed/drkg_target_embeddings.csv"
#     )
#
#     return train_int, test_int, drug_emb, prot_emb

# If you want random split
def load_data():
    interactions = eval_protocol.load_and_standardize_interactions(
        str(PROJECT_ROOT / "data/raw/chembl_pd_interactions.csv")
    )
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
        PROJECT_ROOT / "data/processed/drkg_target_embeddings.csv"
    )

    return train_int, test_int, drug_emb, prot_emb


def compute_drug_similarity(train_drug_ids, test_drug_ids, drug_emb):
    emb_cols = [c for c in drug_emb.columns if c.startswith("drug_emb_")]
    drug_map = {
        row["drug_id"]: row[emb_cols].values.astype(np.float32)
        for _, row in drug_emb.iterrows()
    }

    test_vecs = np.array([
        drug_map[d] for d in test_drug_ids if d in drug_map
    ])

    if len(test_vecs) == 0:
        return {d: 0.0 for d in train_drug_ids}

    test_normed = normalize(test_vecs)

    max_sim = {}
    n_train = len(train_drug_ids)
    for i, did in enumerate(train_drug_ids):
        if i % 5000 == 0:
            print(f"  Processing training drug {i:,}/{n_train:,}")

        if did not in drug_map:
            max_sim[did] = 0.0
            continue

        vec = normalize(drug_map[did].reshape(1, -1))
        sims = (vec @ test_normed.T).flatten()
        max_sim[did] = float(np.max(sims))

    sim_values = list(max_sim.values())
    print(f"  Max drug similarity: min={min(sim_values):.3f} "
          f"mean={np.mean(sim_values):.3f} "
          f"max={max(sim_values):.3f}")
    return max_sim


def compute_protein_similarity(train_target_ids, test_target_ids, prot_emb):
    emb_cols = [c for c in prot_emb.columns if c.startswith("target_emb_")]
    prot_map = {
        row["target_id"]: row[emb_cols].values.astype(np.float32)
        for _, row in prot_emb.iterrows()
    }

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

    print(f"  Protein similarity computed for {len(max_sim):,} training proteins")

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
    reduction = 1 - filtered_n / original_n

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
        "Similarity-Based Partitioning with DRKG TransE — Pareto Analysis",
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

    ax2.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_ylim(0.45, baseline_auc + 0.05)
    ax2.yaxis.grid(True, alpha=0.4)

    plt.tight_layout()
    out = FIG_DIR / "fig_similarity_partitioning_drkg_random2.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")


def main():
    print("  Similarity-Based Partitioning with DRKG TransE ")
    train_int, test_int, drug_emb, prot_emb = load_data()

    train_drugs   = train_int["drug_id"].unique().tolist()
    test_drugs    = test_int["drug_id"].unique().tolist()
    train_targets = train_int["target_id"].unique().tolist()
    test_targets  = test_int["target_id"].unique().tolist()

    drug_max_sim = compute_drug_similarity(train_drugs, test_drugs, drug_emb)
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

    drug_thresholds = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50]
    prot_thresholds = [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.90, 0.85]
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
    results_df.to_csv(OUT_DIR / "similarity_results_drkg_random2.csv", index=False)

    plot_pareto(results_df, baseline_auc)

    print("\n  RESULTS SUMMARY")
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