"""
plot_timeslice_esm2_vs_drkg.py
-------------------------------
Simple bar chart comparing time-slice ESM2 vs DRKG TransE test AUC only.

Run from project root:
    python src/visualisation/plot_timeslice_esm2_vs_drkg.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BG = "#F8FAFC"

# ── Data ──────────────────────────────────────────────────────────────────────
experiments = [
    "Time-Slice\n(cutoff 2018)",
    "Time-Slice\n+ CV (regularised)",
]
esm2_aucs = [0.7599, 0.7518]
drkg_aucs = [0.7579, 0.7513]

x = np.arange(len(experiments))
width = 0.30

fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
ax.set_facecolor(BG)

# ── Bars ──────────────────────────────────────────────────────────────────────
bars_esm2 = ax.bar(
    x - width/2, esm2_aucs, width,
    color="#2563EB", label="ESM2 (sequence embeddings)",
    zorder=3, edgecolor="white", linewidth=0.5,
)
bars_drkg = ax.bar(
    x + width/2, drkg_aucs, width,
    color="#EA580C", label="DRKG TransE (topology embeddings)",
    zorder=3, edgecolor="white", linewidth=0.5,
)

# ── Value labels ──────────────────────────────────────────────────────────────
for bar in bars_esm2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h - 0.015,
            f"{h:.4f}", ha="center", va="top",
            fontsize=11, color="white", fontweight="bold")

for bar in bars_drkg:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h - 0.015,
            f"{h:.4f}", ha="center", va="top",
            fontsize=11, color="white", fontweight="bold")

# ── Delta annotations ─────────────────────────────────────────────────────────
for i in range(len(experiments)):
    delta = abs(esm2_aucs[i] - drkg_aucs[i])
    mid_x = x[i]
    top_y = max(esm2_aucs[i], drkg_aucs[i]) + 0.012
    ax.annotate(
        f"Δ = {delta:.4f}",
        xy=(mid_x, top_y),
        fontsize=10, ha="center", color="#DC2626", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#DC2626", alpha=0.9),
    )

# ── Reference line ────────────────────────────────────────────────────────────
ax.axhline(y=0.5, color="gray", linewidth=1, linestyle=":",
           alpha=0.6, label="Random chance (0.5)", zorder=1)

# ── Finding box ───────────────────────────────────────────────────────────────
ax.text(
    0.98, 0.08,
    "ESM2 and DRKG TransE give identical\n"
    "predictive performance under time-slice\n"
    "despite encoding different information\n"
    "(r = 0.293 between similarity matrices)\n"
    "→ RF uses embeddings as identifiers only",
    transform=ax.transAxes,
    fontsize=9, va="bottom", ha="right",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7",
              edgecolor="#D97706", alpha=0.95),
)

# ── Formatting ────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(experiments, fontsize=12)
ax.set_ylim(0.45, 0.85)
ax.set_ylabel("Test ROC-AUC", fontsize=12)
ax.set_title(
    "Time-Slice Evaluation — ESM2 vs DRKG TransE\n"
    "Test AUC Comparison (Answers RQ2)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.yaxis.grid(True, alpha=0.35, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10, loc="upper right", framealpha=0.95)

plt.tight_layout()
out = FIG_DIR / "fig_timeslice_esm2_vs_drkg.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out}")