"""
generate_preliminary_results.py
--------------------------------
Generates all preliminary result visualisations for the thesis:

  1. t-SNE of drug embeddings coloured by interaction label
  2. t-SNE of drug embeddings coloured by target
  3. t-SNE of protein (target) embeddings
  4. Model performance comparison: Baseline vs Time-slice
  5. Data overview: label distribution and year distribution

Run from your project root:
    python src/visualisation/generate_preliminary_results.py

Outputs saved to: reports/figures/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA = {
    "interactions":     PROJECT_ROOT / "data" / "raw"       / "chembl_pd_interactions.csv",
    "drug_emb":         PROJECT_ROOT / "data" / "processed" / "chembl_drug_embeddings.csv",
    "protein_emb":      PROJECT_ROOT / "data" / "processed" / "protein_embeddings.csv",
    "baseline_metrics": PROJECT_ROOT / "random_forest" / "rf_baseline"  / "metrics.json",
    "timeslice_metrics":PROJECT_ROOT / "random_forest" / "rf_timeslice" / "metrics.json",
}

# Consistent colour palette
BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
GREY   = "#6B7280"
BG     = "#F8FAFC"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_interactions():
    df = pd.read_csv(DATA["interactions"])
    df = df.rename(columns={
        "molecule_chembl_id": "drug_id",
        "target_chembl_id":   "target_id",
    })
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df["year"]          = pd.to_numeric(df["year"],          errors="coerce")
    df = df.sort_values("pchembl_value", ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["drug_id", "target_id"]).reset_index(drop=True)
    return df


def load_drug_embeddings():
    df = pd.read_csv(DATA["drug_emb"])
    df = df.rename(columns={"molecule_chembl_id": "drug_id"})
    return df


def run_tsne(matrix, n_components=2, perplexity=40, random_state=42, n_iter=1000):
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, matrix.shape[0] - 1),
        random_state=random_state,
        max_iter=n_iter,
        learning_rate="auto",
        init="pca",
        n_jobs=-1,
    )
    return tsne.fit_transform(matrix)


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [saved] {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — t-SNE of drug embeddings coloured by label
# ---------------------------------------------------------------------------

def fig_drug_tsne_by_label(interactions, drug_emb, sample_n=5000):
    print("[fig 1] Drug embedding t-SNE by interaction label...")

    emb_cols = [c for c in drug_emb.columns if c.startswith("drug_emb_")]

    # Merge to get labels
    merged = interactions[["drug_id", "label", "target_id"]].merge(
        drug_emb[["drug_id"] + emb_cols], on="drug_id", how="inner"
    )
    # One row per drug — keep majority label
    per_drug = (
        merged.groupby("drug_id")
        .agg(label=("label", lambda x: int(x.mode()[0])))
        .reset_index()
        .merge(drug_emb[["drug_id"] + emb_cols], on="drug_id", how="inner")
    )

    # Sample for speed
    if len(per_drug) > sample_n:
        per_drug = per_drug.sample(sample_n, random_state=42).reset_index(drop=True)

    X = per_drug[emb_cols].to_numpy(dtype=np.float32)
    coords = run_tsne(X, perplexity=40)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    ax.set_facecolor(BG)

    colours = np.where(per_drug["label"] == 1, BLUE, RED)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colours, alpha=0.45, s=8, linewidths=0,
    )

    legend_handles = [
        mpatches.Patch(color=BLUE, label=f"Active (label=1)  n={int((per_drug['label']==1).sum())}"),
        mpatches.Patch(color=RED,  label=f"Inactive (label=0)  n={int((per_drug['label']==0).sum())}"),
    ]
    ax.legend(handles=legend_handles, fontsize=11, framealpha=0.9,
              loc="upper right", markerscale=2)

    ax.set_title(
        "t-SNE of ChemBERTa Drug Embeddings\nColoured by Interaction Label",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.tick_params(labelsize=9)

    # Annotation box
    ax.text(
        0.02, 0.02,
        f"n = {len(per_drug):,} drugs (sampled)\nModel: ChemBERTa-zinc-base-v1\nDims: 768",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        verticalalignment="bottom",
    )

    fig.tight_layout()
    save(fig, "fig1_drug_tsne_by_label.png")


# ---------------------------------------------------------------------------
# Figure 2 — t-SNE of drug embeddings coloured by target
# ---------------------------------------------------------------------------

def fig_drug_tsne_by_target(interactions, drug_emb, sample_n=5000, top_n_targets=10):
    print("[fig 2] Drug embedding t-SNE by target...")

    emb_cols = [c for c in drug_emb.columns if c.startswith("drug_emb_")]

    # Keep only positives so each drug has a clear target association
    pos = interactions[interactions["label"] == 1][["drug_id", "target_id"]]

    # Keep top N most frequent targets for readability
    top_targets = pos["target_id"].value_counts().head(top_n_targets).index.tolist()
    pos_top = pos[pos["target_id"].isin(top_targets)].copy()

    merged = pos_top.merge(drug_emb[["drug_id"] + emb_cols], on="drug_id", how="inner")
    merged = merged.drop_duplicates(subset=["drug_id"]).reset_index(drop=True)

    if len(merged) > sample_n:
        merged = merged.sample(sample_n, random_state=42).reset_index(drop=True)

    X = merged[emb_cols].to_numpy(dtype=np.float32)
    coords = run_tsne(X, perplexity=35)

    # Encode targets to colour indices
    le = LabelEncoder()
    target_idx = le.fit_transform(merged["target_id"])
    cmap = plt.cm.get_cmap("tab10", len(le.classes_))

    fig, ax = plt.subplots(figsize=(11, 8), facecolor=BG)
    ax.set_facecolor(BG)

    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=target_idx, cmap=cmap, vmin=0, vmax=len(le.classes_) - 1,
        alpha=0.5, s=10, linewidths=0,
    )

    # Shorten target IDs for legend
    handles = [
        mpatches.Patch(
            color=cmap(i),
            label=le.classes_[i]
        )
        for i in range(len(le.classes_))
    ]
    ax.legend(
        handles=handles, fontsize=8, framealpha=0.9,
        loc="upper right", title="Target (ChEMBL ID)",
        title_fontsize=9,
    )

    ax.set_title(
        f"t-SNE of ChemBERTa Drug Embeddings\nColoured by Primary Target (top {top_n_targets} targets, active pairs only)",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.tick_params(labelsize=9)

    ax.text(
        0.02, 0.02,
        f"n = {len(merged):,} drugs (sampled)\nActive interactions only",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        verticalalignment="bottom",
    )

    fig.tight_layout()
    save(fig, "fig2_drug_tsne_by_target.png")


# ---------------------------------------------------------------------------
# Figure 3 — t-SNE of protein embeddings
# ---------------------------------------------------------------------------

def fig_protein_tsne(protein_emb):
    print("[fig 3] Protein embedding t-SNE...")

    emb_cols = [c for c in protein_emb.columns if c.startswith("target_emb_")]
    X = protein_emb[emb_cols].to_numpy(dtype=np.float32)

    # With only 67 points, use low perplexity
    coords = run_tsne(X, perplexity=10, n_iter=2000)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
    ax.set_facecolor(BG)

    ax.scatter(
        coords[:, 0], coords[:, 1],
        color=PURPLE, alpha=0.8, s=120, zorder=3,
        edgecolors="white", linewidths=0.8,
    )

    # Label each point with target ID
    for i, row in protein_emb.reset_index(drop=True).iterrows():
        ax.annotate(
            row["target_id"],
            (coords[i, 0], coords[i, 1]),
            fontsize=5.5,
            xytext=(4, 4), textcoords="offset points",
            color="#1F2937",
            alpha=0.85,
        )

    ax.set_title(
        "t-SNE of ESM2 Protein Embeddings\n63 Parkinson's Disease Targets",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.tick_params(labelsize=9)

    ax.text(
        0.02, 0.02,
        "n = 63 targets\nModel: ESM2-35M\nDims: 480",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        verticalalignment="bottom",
    )

    fig.tight_layout()
    save(fig, "fig3_protein_tsne.png")


# ---------------------------------------------------------------------------
# Figure 4 — Model performance comparison
# ---------------------------------------------------------------------------

def fig_model_comparison():
    print("[fig 4] Model performance comparison...")

    import json

    if not DATA["baseline_metrics"].exists():
        print("  [skip] baseline metrics not found.")
        return
    if not DATA["timeslice_metrics"].exists():
        print("  [skip] timeslice metrics not found.")
        return

    b = json.loads(DATA["baseline_metrics"].read_text())
    t = json.loads(DATA["timeslice_metrics"].read_text())

    metrics = ["test_roc_auc", "test_pr_auc", "test_f1",
               "test_accuracy", "test_precision", "test_recall"]
    labels  = ["ROC-AUC", "PR-AUC", "F1", "Accuracy", "Precision", "Recall"]

    b_vals = [b[m] for m in metrics]
    t_vals = [t[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    ax.set_facecolor(BG)

    bars_b = ax.bar(x - width/2, b_vals, width, label="Baseline (random split)",
                    color=BLUE, alpha=0.85, zorder=3)
    bars_t = ax.bar(x + width/2, t_vals, width, label="Time-slice (cutoff 2018)",
                    color=ORANGE, alpha=0.85, zorder=3)

    # Value labels on bars
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=BLUE)
    for bar in bars_t:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=ORANGE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Random Forest Performance: Baseline vs Time-Slice Evaluation\n"
        "Drug-Target Interaction Prediction for Parkinson's Disease",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.legend(fontsize=11, framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Annotation showing AUC drop
    auc_drop = b_vals[0] - t_vals[0]
    ax.annotate(
        f"AUC drop: −{auc_drop:.3f}\n(optimism from random split)",
        xy=(0, t_vals[0]), xytext=(0.5, 0.55),
        textcoords="axes fraction",
        fontsize=9.5, color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", alpha=0.9),
    )

    fig.tight_layout()
    save(fig, "fig4_model_comparison.png")


# ---------------------------------------------------------------------------
# Figure 5 — Data overview (label dist + year dist)
# ---------------------------------------------------------------------------

def fig_data_overview(interactions):
    print("[fig 5] Data overview charts...")

    fig = plt.figure(figsize=(14, 5), facecolor=BG)
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

    # --- Panel A: Label distribution ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG)

    counts = interactions["label"].value_counts().sort_index()
    bars = ax1.bar(
        ["Inactive (0)", "Active (1)"],
        counts.values,
        color=[RED, BLUE], alpha=0.85, width=0.5, zorder=3,
    )
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                 f"{val:,}\n({100*val/counts.sum():.1f}%)",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax1.set_title("Label Distribution\nAfter Deduplication", fontsize=12,
                  fontweight="bold", pad=10)
    ax1.set_ylabel("Number of Drug-Target Pairs", fontsize=11)
    ax1.set_ylim(0, counts.max() * 1.2)
    ax1.yaxis.grid(True, alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)
    ax1.tick_params(labelsize=10)

    # --- Panel B: Year distribution ---
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG)

    year_counts = (
        interactions["year"].dropna().astype(int)
        .value_counts().sort_index()
    )
    ax2.bar(year_counts.index, year_counts.values,
            color=GREEN, alpha=0.75, zorder=3, width=0.85)

    # Cutoff line
    ax2.axvline(x=2018, color=RED, linewidth=2, linestyle="--", zorder=4,
                label="Train/test cutoff (2018)")
    ax2.fill_betweenx(
        [0, year_counts.max() * 1.15],
        2018, year_counts.index.max(),
        alpha=0.08, color=ORANGE, zorder=2,
    )
    ax2.text(2019.2, year_counts.max() * 0.85, "Test set\n(post-2018)",
             fontsize=9, color=ORANGE, fontweight="bold")
    ax2.text(2008, year_counts.max() * 0.85, "Training set\n(≤2018)",
             fontsize=9, color=GREEN, fontweight="bold")

    ax2.set_title("Interaction Records by Year\nTime-Slice Split at 2018",
                  fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylabel("Number of Raw Interactions", fontsize=11)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.yaxis.grid(True, alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)
    ax2.tick_params(labelsize=9)

    fig.suptitle(
        "ChEMBL Parkinson's Disease Interaction Dataset Overview",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig5_data_overview.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  Generating preliminary results figures")
    print(f"  Output directory: {OUT_DIR}")
    print("=" * 55)

    print("\n[loading] Interactions...")
    interactions = load_interactions()
    print(f"  {len(interactions):,} unique drug-target pairs")

    print("\n[loading] Drug embeddings...")
    drug_emb = load_drug_embeddings()
    print(f"  {len(drug_emb):,} drugs × {sum(1 for c in drug_emb.columns if c.startswith('drug_emb_'))} dims")

    print("\n[loading] Protein embeddings...")
    protein_emb = pd.read_csv(DATA["protein_emb"])
    print(f"  {len(protein_emb)} targets × {sum(1 for c in protein_emb.columns if c.startswith('target_emb_'))} dims")

    print()
    fig_data_overview(interactions)
    fig_drug_tsne_by_label(interactions, drug_emb, sample_n=5000)
    fig_drug_tsne_by_target(interactions, drug_emb, sample_n=5000)
    fig_protein_tsne(protein_emb)
    fig_model_comparison()

    print("\n" + "=" * 55)
    print("  All figures saved to reports/figures/")
    print("  Files:")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"    {f.name}")
    print("=" * 55)


if __name__ == "__main__":
    main()