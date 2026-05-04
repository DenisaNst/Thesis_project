"""
plot_target_similarity_heatmap.py
----------------------------------
Generates a side-by-side cosine similarity heatmap of the 63 PD targets
under two embedding types:
  Left:  ESM2 (sequence-based)
  Right: DRKG TransE (topology-based)

If the two heatmaps look similar → both embeddings encode the same
biological relationships (explains why AUC was the same).

If they look different → they encode different information, and
combining them might improve predictions.

Run from your project root:
    python src/visualisation/plot_target_similarity_heatmap.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR      = PROJECT_ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#F8FAFC"


# ---------------------------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------------------------

def load_esm2(path):
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith("target_emb_")]
    return df["target_id"].tolist(), df[emb_cols].to_numpy(dtype=np.float32)


def load_drkg(path):
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith("target_emb_")]
    return df["target_id"].tolist(), df[emb_cols].to_numpy(dtype=np.float32)


def cosine_similarity_matrix(embeddings):
    """Compute pairwise cosine similarity matrix."""
    normed = normalize(embeddings, norm="l2")
    return normed @ normed.T


def hierarchical_order(sim_matrix):
    """
    Returns index order from hierarchical clustering so that
    similar targets are grouped together in the heatmap.
    """
    dist = squareform(pdist(sim_matrix, metric="euclidean"))
    linkage_matrix = linkage(squareform(dist), method="average")
    dendro = dendrogram(linkage_matrix, no_plot=True)
    return dendro["leaves"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading ESM2 protein embeddings...")
    esm2_ids, esm2_emb = load_esm2(
        PROJECT_ROOT / "data/processed/protein_embeddings.csv"
    )
    print(f"  ESM2: {len(esm2_ids)} targets × {esm2_emb.shape[1]} dims")

    print("Loading DRKG TransE embeddings...")
    drkg_ids, drkg_emb = load_drkg(
        PROJECT_ROOT / "data/processed/drkg_target_embeddings.csv"
    )
    print(f"  DRKG: {len(drkg_ids)} targets × {drkg_emb.shape[1]} dims")

    # Find common targets between both embeddings
    common_ids = sorted(set(esm2_ids) & set(drkg_ids))
    print(f"  Common targets: {len(common_ids)}")

    # Align both to common targets in same order
    esm2_idx = [esm2_ids.index(t) for t in common_ids]
    drkg_idx = [drkg_ids.index(t) for t in common_ids]

    esm2_aligned = esm2_emb[esm2_idx]
    drkg_aligned = drkg_emb[drkg_idx]

    # Compute similarity matrices
    print("Computing cosine similarity matrices...")
    esm2_sim = cosine_similarity_matrix(esm2_aligned)
    drkg_sim  = cosine_similarity_matrix(drkg_aligned)

    # Hierarchical clustering order for ESM2 (reuse same order for DRKG
    # so the two heatmaps are directly comparable)
    order = hierarchical_order(esm2_sim)
    esm2_sim_ordered = esm2_sim[np.ix_(order, order)]
    drkg_sim_ordered  = drkg_sim[np.ix_(order, order)]
    labels_ordered   = [common_ids[i] for i in order]

    # Shorten labels for readability
    short_labels = [lbl.replace("CHEMBL", "") for lbl in labels_ordered]

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    n = len(common_ids)
    fontsize = max(3, min(7, 120 // n))  # auto-scale label size

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 8),
        facecolor=BG,
        gridspec_kw={"width_ratios": [1, 1, 0.05]},
    )
    fig.suptitle(
        "Pairwise Cosine Similarity of PD Targets\n"
        "ESM2 (Sequence) vs DRKG TransE (Topology)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    kwargs = dict(vmin=-0.2, vmax=1.0, aspect="auto")
    cmap   = "RdYlBu_r"

    # ── Left: ESM2 ──────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(BG)
    im1 = ax1.imshow(esm2_sim_ordered, cmap=cmap, **kwargs)

    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(short_labels, rotation=90, fontsize=fontsize)
    ax1.set_yticklabels(short_labels, fontsize=fontsize)
    ax1.set_title(
        f"ESM2 — Sequence Embeddings\n({esm2_emb.shape[1]} dims)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax1.set_xlabel("Target (ChEMBL ID, prefix removed)", fontsize=9)
    ax1.set_ylabel("Target (ChEMBL ID, prefix removed)", fontsize=9)

    # Annotation box
    ax1.text(
        0.02, 0.98,
        "Clusters reflect\nprotein sequence\nsimilarity\n(evolutionary families)",
        transform=ax1.transAxes, fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    # ── Right: DRKG TransE ──────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    im2 = ax2.imshow(drkg_sim_ordered, cmap=cmap, **kwargs)

    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(short_labels, rotation=90, fontsize=fontsize)
    ax2.set_yticklabels(short_labels, fontsize=fontsize)
    ax2.set_title(
        f"DRKG TransE — Topology Embeddings\n({drkg_emb.shape[1]} dims)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax2.set_xlabel("Target (ChEMBL ID, prefix removed)", fontsize=9)

    ax2.text(
        0.02, 0.98,
        "Clusters reflect\nbiological network\nposition\n(disease/pathway links)",
        transform=ax2.transAxes, fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    # ── Colorbar ─────────────────────────────────────────────────────────────
    ax3 = axes[2]
    cbar = fig.colorbar(im2, cax=ax3)
    cbar.set_label("Cosine Similarity", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # ── Compute and print correlation between the two matrices ───────────────
    upper_tri = np.triu_indices(n, k=1)
    esm2_vals = esm2_sim[upper_tri]
    drkg_vals = drkg_sim[upper_tri]
    correlation = np.corrcoef(esm2_vals, drkg_vals)[0, 1]
    print(f"\nCorrelation between ESM2 and DRKG similarity matrices: {correlation:.4f}")

    # Add correlation annotation to figure
    fig.text(
        0.5, -0.02,
        f"Pearson correlation between ESM2 and DRKG similarity matrices: r = {correlation:.3f}  "
        f"({'High' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Low'} overlap — "
        f"{'embeddings encode similar information' if abs(correlation) > 0.7 else 'embeddings encode different information'})",
        ha="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF3C7", alpha=0.9),
    )

    plt.tight_layout()
    out_path = OUT_DIR / "fig_target_similarity_heatmap.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")

    # ── Print interpretation ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  INTERPRETATION")
    print("=" * 55)
    if abs(correlation) > 0.7:
        print(f"  r = {correlation:.3f} → HIGH correlation")
        print("  ESM2 and DRKG encode similar biological relationships.")
        print("  This explains why RF AUC was nearly identical for both.")
        print("  For well-studied PD targets: sequence similarity ≈")
        print("  network position similarity (evolutionarily conserved).")
    elif abs(correlation) > 0.4:
        print(f"  r = {correlation:.3f} → MODERATE correlation")
        print("  ESM2 and DRKG encode partially different information.")
        print("  Combining both might improve predictions.")
    else:
        print(f"  r = {correlation:.3f} → LOW correlation")
        print("  ESM2 and DRKG encode very different information.")
        print("  This is surprising — investigate which targets differ most.")
    print("=" * 55)


if __name__ == "__main__":
    main()