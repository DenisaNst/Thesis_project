"""
How this script works:
1. Metric Loading: Initializes a dictionary containing Train/Test AUC scores
   for both ESM2 and DRKG embeddings across three distinct splitting strategies.
2. Figure Initialization: Sets up a grouped bar chart layout using Matplotlib.
3. Bar Generation: Plots four bars per evaluation condition to contrast the
   Test AUC (honest performance) and Train AUC (overfitting) for both models.
4. Annotation: Dynamically calculates and annotates the absolute difference (Δ)
   between ESM2 and DRKG Test AUCs to highlight their statistical similarity,
   along with visual indicators for the evaluation strategy drop-off.
5. Export: Renders the composite figure and saves it to the reports directory.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

results = {
    "Random Split\n(naive)": {
        "ESM2":  {"test": 0.8887, "train": 0.9996},
        "DRKG":  {"test": 0.8837, "train": 0.9992},
    },
    "Time-Slice\n(cutoff 2018)": {
        "ESM2":  {"test": 0.7599, "train": 0.9994},
        "DRKG":  {"test": 0.7579, "train": 0.9989},
    },
    "Time-Slice\n+ CV (regularised)": {
        "ESM2":  {"test": 0.7518, "train": 0.9681},
        "DRKG":  {"test": 0.7513, "train": 0.9674},
    },
}

ESM2_TEST_COL  = "#2563EB"
DRKG_TEST_COL  = "#EA580C"
ESM2_TRAIN_COL = "#93C5FD"
DRKG_TRAIN_COL = "#FCA37A"
BG             = "#F8FAFC"
RED_LINE       = "#DC2626"

conditions = list(results.keys())
n = len(conditions)
x = np.arange(n)
width = 0.18

fig, ax = plt.subplots(figsize=(13, 7), facecolor=BG)
ax.set_facecolor(BG)

offsets = [-1.5, -0.5, 0.5, 1.5]

bars_esm2_test = ax.bar(x + offsets[0]*width, [results[c]["ESM2"]["test"] for c in conditions], width, color=ESM2_TEST_COL, edgecolor="white", linewidth=0.5, zorder=3)
bars_drkg_test = ax.bar(x + offsets[1]*width, [results[c]["DRKG"]["test"] for c in conditions], width, color=DRKG_TEST_COL, edgecolor="white", linewidth=0.5, zorder=3)
bars_esm2_train = ax.bar(x + offsets[2]*width, [results[c]["ESM2"]["train"] for c in conditions], width, color=ESM2_TRAIN_COL, edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
bars_drkg_train = ax.bar(x + offsets[3]*width, [results[c]["DRKG"]["train"] for c in conditions], width, color=DRKG_TRAIN_COL, edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)

def label_bars(bars, fontsize=9, color="white"):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h - 0.025, f"{h:.3f}",
            ha="center", va="top", fontsize=fontsize, color=color, fontweight="bold",
        )

label_bars(bars_esm2_test, color="white")
label_bars(bars_drkg_test, color="white")
label_bars(bars_esm2_train, color="#1E3A5F", fontsize=8)
label_bars(bars_drkg_train, color="#7C2D12", fontsize=8)

for i, c in enumerate(conditions):
    esm2_auc = results[c]["ESM2"]["test"]
    drkg_auc = results[c]["DRKG"]["test"]
    delta = abs(esm2_auc - drkg_auc)

    mid_x = x[i] + (offsets[0] + offsets[1]) / 2 * width
    top_y = max(esm2_auc, drkg_auc) + 0.025

    ax.annotate(
        f"Δ = {delta:.4f}", xy=(mid_x, top_y),
        fontsize=9, ha="center", color=RED_LINE, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=RED_LINE, alpha=0.9),
    )

ax.axhline(y=0.5, color="gray", linewidth=1, linestyle=":", alpha=0.6, zorder=1)
ax.axhline(y=1.0, color="#6B7280", linewidth=0.5, linestyle="--", alpha=0.3, zorder=1)

finding_text = (
    "Key finding:\n"
    "Evaluation strategy changes AUC by ~0.129\n"
    "Embedding type (ESM2 vs DRKG) changes AUC by ~0.002\n"
    "→ RF cannot exploit DRKG's richer topology"
)
ax.text(
    0.98, 0.05, finding_text, transform=ax.transAxes,
    fontsize=9.5, va="bottom", ha="right",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7", edgecolor="#D97706", alpha=0.95),
)

ax.annotate(
    "", xy=(0.67, 0.760), xycoords=("axes fraction", "data"),
    xytext=(0.33, 0.8887), textcoords=("axes fraction", "data"),
    arrowprops=dict(arrowstyle="-|>", color=RED_LINE, lw=2, connectionstyle="arc3,rad=0.2"),
)
ax.text(
    0.50, 0.83, "−0.129\n(evaluation\nstrategy effect)",
    transform=ax.get_xaxis_transform(),
    fontsize=8.5, color=RED_LINE, ha="center", fontweight="bold",
)

ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=12)
ax.set_ylim(0.45, 1.08)
ax.set_ylabel("ROC-AUC", fontsize=12)
ax.set_title(
    "ESM2 vs DRKG TransE — Prediction Performance\n"
    "Across Evaluation Strategies (Random Forest on ChEMBL PD Interactions)",
    fontsize=13, fontweight="bold", pad=15,
)

ax.yaxis.grid(True, alpha=0.35, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_patches = [
    mpatches.Patch(color=ESM2_TEST_COL,  label="ESM2 — Test AUC  (honest)"),
    mpatches.Patch(color=DRKG_TEST_COL,  label="DRKG — Test AUC  (honest)"),
    mpatches.Patch(color=ESM2_TRAIN_COL, label="ESM2 — Train AUC (overfitting check)"),
    mpatches.Patch(color=DRKG_TRAIN_COL, label="DRKG — Train AUC (overfitting check)"),
    mpatches.Patch(color="gray",         label="Random chance (0.5)", alpha=0.6),
]
ax.legend(handles=legend_patches, fontsize=9.5, loc="upper right", framealpha=0.95, edgecolor="#D1D5DB")

plt.tight_layout()
out_path = FIG_DIR / "fig_esm2_vs_drkg_comparison.png"
plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()