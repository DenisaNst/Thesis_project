"""
Time-Slice Cutoff Sensitivity Analysis Plotter.

How this script works mechanically:
1. Metric Loading: Loads a pre-computed dictionary of evaluation metrics (Train/Test
   sizes and AUC scores) for every temporal cutoff year between 1995 and 2022.
   These results were copied from artifacts.
2. Data Extraction: Unpacks the dictionary into temporal arrays (years, test_aucs,
   train_aucs, gaps, train_rows, test_rows) for plotting.
3. Figure Initialization: Sets up a vertically stacked, 3-panel Matplotlib figure.
4. Panel 1 (Predictive Performance): Plots the Train AUC vs. Test AUC over time,
   highlighting the chosen 2018 cutoff and shading the unreliable post-2020 zone.
5. Panel 2 (Overfitting Gap): Plots the mathematical difference between the Train
   and Test AUC to visualize how structural memorization changes as the test set shrinks.
6. Panel 3 (Data Volume): Plots the raw number of training vs. testing pairs available
   at each temporal cutoff.
7. Export: Renders and saves the final composite image to the reports directory.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS = {
    1995: {"train": 2115,  "test": 94658, "train_auc": 1.0000, "test_auc": 0.5860},
    1996: {"train": 2980,  "test": 93793, "train_auc": 1.0000, "test_auc": 0.5725},
    1997: {"train": 3963,  "test": 92810, "train_auc": 1.0000, "test_auc": 0.5702},
    1998: {"train": 5258,  "test": 91515, "train_auc": 1.0000, "test_auc": 0.5627},
    1999: {"train": 7198,  "test": 89575, "train_auc": 1.0000, "test_auc": 0.5623},
    2000: {"train": 9799,  "test": 86974, "train_auc": 1.0000, "test_auc": 0.5957},
    2001: {"train": 11582, "test": 85191, "train_auc": 0.9999, "test_auc": 0.6138},
    2002: {"train": 14368, "test": 82405, "train_auc": 0.9999, "test_auc": 0.6198},
    2003: {"train": 16774, "test": 79999, "train_auc": 0.9999, "test_auc": 0.6209},
    2004: {"train": 19218, "test": 77555, "train_auc": 0.9999, "test_auc": 0.6498},
    2005: {"train": 21863, "test": 74910, "train_auc": 0.9998, "test_auc": 0.6623},
    2006: {"train": 25303, "test": 71470, "train_auc": 0.9998, "test_auc": 0.6601},
    2007: {"train": 30153, "test": 66620, "train_auc": 0.9998, "test_auc": 0.6772},
    2008: {"train": 35970, "test": 60803, "train_auc": 0.9999, "test_auc": 0.7074},
    2009: {"train": 41560, "test": 55213, "train_auc": 0.9998, "test_auc": 0.7132},
    2010: {"train": 47674, "test": 49099, "train_auc": 0.9997, "test_auc": 0.7177},
    2011: {"train": 52858, "test": 43915, "train_auc": 0.9997, "test_auc": 0.7230},
    2012: {"train": 57039, "test": 39734, "train_auc": 0.9997, "test_auc": 0.7309},
    2013: {"train": 61667, "test": 35106, "train_auc": 0.9997, "test_auc": 0.7411},
    2014: {"train": 68033, "test": 28740, "train_auc": 0.9996, "test_auc": 0.7427},
    2015: {"train": 72186, "test": 24587, "train_auc": 0.9996, "test_auc": 0.7413},
    2016: {"train": 76632, "test": 20141, "train_auc": 0.9995, "test_auc": 0.7460},
    2017: {"train": 80778, "test": 15995, "train_auc": 0.9995, "test_auc": 0.7306},
    2018: {"train": 84152, "test": 12621, "train_auc": 0.9994, "test_auc": 0.7599},
    2019: {"train": 87251, "test": 9522,  "train_auc": 0.9995, "test_auc": 0.7688},
    2020: {"train": 90776, "test": 5997,  "train_auc": 0.9995, "test_auc": 0.7719},
    2021: {"train": 93187, "test": 3586,  "train_auc": 0.9994, "test_auc": 0.7730},
    2022: {"train": 94794, "test": 1979,  "train_auc": 0.9994, "test_auc": 0.7576},
}

years      = sorted(RESULTS.keys())
test_aucs  = [RESULTS[y]["test_auc"]  for y in years]
train_aucs = [RESULTS[y]["train_auc"] for y in years]
gaps       = [RESULTS[y]["train_auc"] - RESULTS[y]["test_auc"] for y in years]
train_rows = [RESULTS[y]["train"]     for y in years]
test_rows  = [RESULTS[y]["test"]      for y in years]

BLUE   = "#2563EB"
ORANGE = "#EA580C"
RED    = "#DC2626"
GREEN  = "#16A34A"
GREY   = "#6B7280"
BG     = "#F8FAFC"

fig, axes = plt.subplots(3, 1, figsize=(13, 14), facecolor=BG)
fig.suptitle(
    "Time-Slice Cutoff Sensitivity Analysis\n"
    "Random Forest on ChEMBL PD Interactions (1995–2022)",
    fontsize=14, fontweight="bold", y=0.98,
)

# Panel 1
ax1 = axes[0]
ax1.set_facecolor(BG)
ax1.plot(years, train_aucs, color=BLUE,   linewidth=2.5,
         marker="o", markersize=4, label="Train AUC (~0.999 always)")
ax1.plot(years, test_aucs,  color=ORANGE, linewidth=2.5,
         marker="o", markersize=4, label="Test AUC (honest)")

# Highlight chosen cutoff
ax1.axvline(x=2018, color=RED, linewidth=2, linestyle="--", zorder=4)
ax1.annotate(
    "Chosen\ncutoff\n(2018)",
    xy=(2018, 0.7599), xytext=(2013.5, 0.72),
    fontsize=9, color=RED, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
)

ax1.axvspan(2020, 2023, alpha=0.08, color=RED,
            label="Unreliable (test < 6k rows)")
ax1.text(2020.3, 0.58, "Small\ntest set\n(unreliable)", fontsize=8,
         color=RED, alpha=0.8)

ax1.set_ylabel("ROC-AUC", fontsize=11)
ax1.set_ylim(0.55, 1.05)
ax1.legend(fontsize=10, framealpha=0.9)
ax1.yaxis.grid(True, alpha=0.4)
ax1.set_axisbelow(True)
ax1.tick_params(labelbottom=False)
ax1.set_title("Train vs Test AUC by Cutoff Year", fontsize=12, pad=8)

# Panel 2
ax2 = axes[1]
ax2.set_facecolor(BG)
ax2.fill_between(years, gaps, alpha=0.3, color=RED)
ax2.plot(years, gaps, color=RED, linewidth=2.5, marker="o", markersize=4)
ax2.axvline(x=2018, color=RED, linewidth=2, linestyle="--")
ax2.axhline(y=0.239, color=GREY, linewidth=1, linestyle=":",
            label="Gap at 2018 cutoff (0.239)")
ax2.set_ylabel("Overfitting Gap\n(Train AUC − Test AUC)", fontsize=11)
ax2.legend(fontsize=10, framealpha=0.9)
ax2.yaxis.grid(True, alpha=0.4)
ax2.set_axisbelow(True)
ax2.tick_params(labelbottom=False)
ax2.set_title("Overfitting Gap by Cutoff Year", fontsize=12, pad=8)

# Panel 3
ax3 = axes[2]
ax3.set_facecolor(BG)
ax3.fill_between(years, train_rows, alpha=0.4, color=BLUE, label="Train rows")
ax3.fill_between(years, test_rows,  alpha=0.4, color=ORANGE, label="Test rows")
ax3.plot(years, train_rows, color=BLUE,   linewidth=2)
ax3.plot(years, test_rows,  color=ORANGE, linewidth=2)
ax3.axvline(x=2018, color=RED, linewidth=2, linestyle="--",
            label="Chosen cutoff (2018)")

# Annotate 2018 sizes
ax3.annotate(
    f"Train: 84,152\nTest: 12,621",
    xy=(2018, 84152), xytext=(2010, 75000),
    fontsize=9, color=BLUE,
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2),
)

ax3.set_xlabel("Cutoff Year", fontsize=11)
ax3.set_ylabel("Number of Pairs", fontsize=11)
ax3.legend(fontsize=10, framealpha=0.9)
ax3.yaxis.grid(True, alpha=0.4)
ax3.set_axisbelow(True)
ax3.set_title("Train / Test Set Size by Cutoff Year", fontsize=12, pad=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = OUT_DIR / "fig_cutoff_sensitivity.png"
plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out_path}")