"""
analyse_predictions.py
-----------------------
Analyses FDA drug repositioning predictions from the RF model.

Reads target names from pd_targets_metadata.csv and known PD drugs
from pd_indications.csv (ChEMBL EFO ontology query — structured source).

Usage:
    python src/evaluation/analyse_predictions.py \
        --scores_csv          artifacts/rf_cv/fda_target_scores_all.csv \
        --targets_metadata    data/raw/pd_targets_metadata.csv \
        --pd_indications_csv  data/raw/pd_indications.csv \
        --out_dir             artifacts/rf_cv/prediction_analysis_all \
        --high_conf_threshold 0.9 \
        --memorisation_target "Glucocorticoid receptor"
"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"drug_id", "target_id", "score"} - set(df.columns)
    if missing:
        raise ValueError(f"scores CSV missing columns: {missing}")
    return (df.sort_values("score", ascending=False)
              .reset_index(drop=True)
              .assign(rank=lambda d: d.index + 1))


def load_target_names(metadata_csv: Path) -> dict:
    df = pd.read_csv(metadata_csv)
    df.columns = [c.lower().strip() for c in df.columns]
    id_col   = next((c for c in ["target_chembl_id", "target_id"] if c in df.columns), None)
    name_col = next((c for c in ["pref_name", "preferred_name", "name"] if c in df.columns), None)
    if not id_col or not name_col:
        raise ValueError(f"pd_targets_metadata.csv must have id+name cols. Found: {list(df.columns)}")
    mapping = dict(zip(df[id_col].astype(str), df[name_col].astype(str)))
    print(f"  Loaded {len(mapping)} target names from {metadata_csv.name}")
    return mapping


def load_known_pd_drugs(pd_indications_csv: Path) -> set:
    """
    Uses pd_indications.csv — saved by fetch_chembl_interactions.py from the
    ChEMBL EFO ontology query (efo_term=parkinson). Resolves molecule ChEMBL
    IDs to preferred drug names via the ChEMBL API, cached after first run.
    """
    import requests

    if not pd_indications_csv.exists():
        print(f"  Warning: {pd_indications_csv} not found.")
        return set()

    ind_df = pd.read_csv(pd_indications_csv)
    ind_df.columns = [c.lower().strip() for c in ind_df.columns]
    mol_id_col = next((c for c in ["molecule_chembl_id", "mol_chembl_id", "chembl_id"]
                       if c in ind_df.columns), None)
    if mol_id_col is None:
        print(f"  Warning: no molecule_chembl_id column. Cols: {list(ind_df.columns)}")
        return set()

    mol_ids = ind_df[mol_id_col].dropna().unique().tolist()
    print(f"  pd_indications.csv: {len(mol_ids)} PD-indicated molecules")

    cache_path = pd_indications_csv.parent / "pd_indication_names_cache.csv"
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
        print(f"  Name cache loaded ({len(cache)} entries)")
    else:
        print("  Fetching names from ChEMBL API (one-time, will be cached)...")
        rows = []
        for i, mol_id in enumerate(mol_ids):
            try:
                r = requests.get(
                    f"https://www.ebi.ac.uk/chembl/api/data/molecule/{mol_id}.json",
                    timeout=10)
                if r.status_code == 200:
                    pref = r.json().get("pref_name", "")
                    if pref:
                        rows.append({"molecule_chembl_id": mol_id, "pref_name": pref})
                time.sleep(0.15)
            except Exception:
                continue
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(mol_ids)} fetched...")
        cache = pd.DataFrame(rows)
        cache.to_csv(cache_path, index=False)
        print(f"  Cached {len(cache)} names to {cache_path.name}")

    known = set(cache["pref_name"].str.lower().str.strip().dropna().tolist())
    print(f"  Known PD drugs resolved: {len(known)}")
    return known


def flag_known_pd(df: pd.DataFrame, known_pd: set) -> pd.DataFrame:
    name_col = "drug_name" if "drug_name" in df.columns else "drug_id"
    df = df.copy()
    df["_lower"] = df[name_col].astype(str).str.lower().str.strip()
    df["is_known_pd"] = df["_lower"].apply(
        lambda n: any(pd_d in n or n in pd_d for pd_d in known_pd)
    )
    return df


# ---------------------------------------------------------------------------
# Analysis 1 — Target distribution
# ---------------------------------------------------------------------------

def analyse_target_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    print("\n=== ANALYSIS 1 — Target Distribution ===")
    counts = df["target_name"].value_counts()
    total  = len(df)
    for t, c in counts.items():
        flag = "  <- MEMORISATION?" if c / total > 0.30 else ""
        print(f"  {str(t):<55} {c:>6}  ({c/total*100:.1f}%){flag}")

    colors = ["#e74c3c" if c / total > 0.30 else "#2980b9" for c in counts.values]
    fig, ax = plt.subplots(figsize=(10, max(5, len(counts) * 0.38)))
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    ax.axvline(total / max(len(counts), 1), color="gray", linestyle="--",
               label=f"Equal share ({total // max(len(counts), 1):,} per target)")
    ax.set_xlabel("Number of predictions")
    ax.set_title("Target Distribution\n(red = >30% of predictions, memorisation risk)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "1_target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] 1_target_distribution.png")


# ---------------------------------------------------------------------------
# Analysis 2 — Score distribution histogram
# ---------------------------------------------------------------------------

def analyse_score_distribution(df: pd.DataFrame, threshold: float, out_dir: Path) -> None:
    print(f"\n=== ANALYSIS 2 — Score Distribution ===")

    scores = df["score"].values
    novel_scores  = df[~df["is_known_pd"]]["score"].values
    known_scores  = df[df["is_known_pd"]]["score"].values

    print(f"  All pairs:          n={len(scores):,}  mean={scores.mean():.3f}  median={np.median(scores):.3f}")
    print(f"  Novel pairs:        n={len(novel_scores):,}  mean={novel_scores.mean():.3f}")
    print(f"  Known PD pairs:     n={len(known_scores):,}  mean={known_scores.mean():.3f}")
    print(f"\n  Score buckets (all pairs):")
    bins_labels = [(0.0,0.3,"0.0-0.3"),(0.3,0.5,"0.3-0.5"),(0.5,0.7,"0.5-0.7"),
                   (0.7,0.8,"0.7-0.8"),(0.8,0.9,"0.8-0.9"),(0.9,1.01,"0.9-1.0")]
    for lo, hi, label in bins_labels:
        n = ((scores >= lo) & (scores < hi)).sum()
        print(f"    {label}:  {n:>7,}  ({n/len(scores)*100:.1f}%)")

    above = (scores >= threshold).sum()
    print(f"\n  Above threshold ({threshold}):  {above:,} pairs  ({above/len(scores)*100:.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left — full distribution split by known/novel
    bins = np.linspace(0, 1, 40)
    axes[0].hist(novel_scores, bins=bins, alpha=0.65, color="#2980b9",
                 label=f"Novel candidates ({len(novel_scores):,})")
    axes[0].hist(known_scores, bins=bins, alpha=0.65, color="#e74c3c",
                 label=f"Known PD drugs ({len(known_scores):,})")
    axes[0].axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                    label=f"Threshold = {threshold}")
    axes[0].set_xlabel("Predicted interaction probability")
    axes[0].set_ylabel("Number of drug-target pairs")
    axes[0].set_title("Score Distribution: Novel vs Known PD Drugs")
    axes[0].legend(fontsize=9)

    # Right — zoom into high-confidence region (> 0.7)
    high_novel = novel_scores[novel_scores >= 0.7]
    high_known = known_scores[known_scores >= 0.7]
    bins_zoom  = np.linspace(0.7, 1.0, 25)
    axes[1].hist(high_novel, bins=bins_zoom, alpha=0.65, color="#2980b9",
                 label=f"Novel (score ≥ 0.7): {len(high_novel):,}")
    axes[1].hist(high_known, bins=bins_zoom, alpha=0.65, color="#e74c3c",
                 label=f"Known PD (score ≥ 0.7): {len(high_known):,}")
    axes[1].axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                    label=f"Threshold = {threshold}")
    axes[1].set_xlabel("Predicted interaction probability")
    axes[1].set_ylabel("Number of drug-target pairs")
    axes[1].set_title(f"Zoom: High-Confidence Region (≥ 0.7)\n{above:,} pairs above {threshold}")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "2_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] 2_score_distribution.png")


# ---------------------------------------------------------------------------
# Analysis 3 — Known PD drug validation
# ---------------------------------------------------------------------------

def analyse_known_pd_drugs(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    print("\n=== ANALYSIS 3 — Known PD Drug Validation ===")
    name_col = "drug_name" if "drug_name" in df.columns else "drug_id"
    hits = df[df["is_known_pd"]].copy()

    # Best prediction per known PD drug (highest score across all targets)
    best = (hits.sort_values("score", ascending=False)
                .drop_duplicates(subset=["drug_id"])
                .reset_index(drop=True))

    print(f"  {best['drug_id'].nunique()} known PD drug(s) found in predictions")
    print(f"  Score distribution for known PD drugs:")
    print(f"    Mean:   {best['score'].mean():.3f}")
    print(f"    Median: {best['score'].median():.3f}")
    print(f"    Min:    {best['score'].min():.3f}")
    print(f"    Max:    {best['score'].max():.3f}")
    print(f"\n  Top 20 known PD drugs by best score:")
    print(f"  {'Drug':<30} {'Best Target':<45} {'Score':>7}  {'Rank':>6}")
    print("  " + "-"*90)
    for _, r in best.head(20).iterrows():
        print(f"  {str(r[name_col]):<30} {str(r['target_name']):<45} "
              f"{r['score']:>7.4f}  #{int(r['rank'])}")

    hits.to_csv(out_dir / "3_known_pd_drug_hits.csv", index=False)
    best.to_csv(out_dir / "3_known_pd_best_per_drug.csv", index=False)
    print(f"  [saved] 3_known_pd_drug_hits.csv  +  3_known_pd_best_per_drug.csv")
    return hits


# ---------------------------------------------------------------------------
# Analysis 4 — Novel high-confidence candidates
# ---------------------------------------------------------------------------

def analyse_novel_candidates(
    df: pd.DataFrame,
    threshold: float,
    memorisation_target: str,
    out_dir: Path
) -> pd.DataFrame:
    print(f"\n=== ANALYSIS 4 — Novel Repositioning Candidates (score ≥ {threshold}) ===")
    name_col = "drug_name" if "drug_name" in df.columns else "drug_id"

    # All novel pairs
    novel = df[~df["is_known_pd"]].copy()

    # High-confidence subset, excluding the memorisation target
    high_conf = novel[
        (novel["score"] >= threshold) &
        (novel["target_name"] != memorisation_target)
    ].copy()

    # Also save high-confidence INCLUDING memorisation target, for completeness
    high_conf_all = novel[novel["score"] >= threshold].copy()

    print(f"  Novel pairs total:                    {len(novel):,}")
    print(f"  Novel pairs score ≥ {threshold}:             {len(high_conf_all):,}")
    print(f"  Novel pairs score ≥ {threshold}, excl. GR:   {len(high_conf):,}")

    # Top 25 — one per drug, best target (excluding memorisation target)
    top25 = (high_conf.sort_values("score", ascending=False)
             .drop_duplicates(subset=["drug_id"])
             .head(25)
             .reset_index(drop=True))

    print(f"\n  Top 25 novel candidates (score ≥ {threshold}, excl. {memorisation_target}):\n")
    print(f"  {'#':<4} {'Drug':<30} {'Target':<45} {'Score':>7}")
    print("  " + "-"*90)
    for i, r in top25.iterrows():
        print(f"  #{i+1:<3} {str(r[name_col]):<30} {str(r['target_name']):<45} {r['score']:.4f}")

    novel.to_csv(out_dir / "4_novel_candidates_all.csv", index=False)
    high_conf_all.to_csv(out_dir / "4_novel_high_confidence.csv", index=False)
    high_conf.to_csv(out_dir / "4_novel_high_confidence_excl_gr.csv", index=False)
    top25.to_csv(out_dir / "4_top25_novel_candidates.csv", index=False)
    print(f"\n  [saved] 4_novel_candidates_all.csv")
    print(f"  [saved] 4_novel_high_confidence.csv          ({len(high_conf_all):,} pairs)")
    print(f"  [saved] 4_novel_high_confidence_excl_gr.csv  ({len(high_conf):,} pairs)")
    print(f"  [saved] 4_top25_novel_candidates.csv")
    return top25


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv",           type=Path,  default=Path("artifacts/rf_cv/fda_target_scores_all.csv"))
    parser.add_argument("--targets_metadata",     type=Path,  default=Path("data/raw/pd_targets_metadata.csv"))
    parser.add_argument("--pd_indications_csv",   type=Path,  default=Path("data/raw/pd_indications.csv"))
    parser.add_argument("--out_dir",              type=Path,  default=Path("artifacts/rf_cv/prediction_analysis_all"))
    parser.add_argument("--high_conf_threshold",  type=float, default=0.9,
                        help="Score threshold for high-confidence candidates (default 0.9)")
    parser.add_argument("--memorisation_target",  type=str,   default="Glucocorticoid receptor",
                        help="Target name to exclude from novel candidates (memorisation artifact)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_scores(args.scores_csv)
    print(f"Loaded {len(df):,} predictions  |  {df['target_id'].nunique()} targets  |  {df['drug_id'].nunique()} drugs")

    # Resolve names
    print("\n[1/2] Resolving target names...")
    tmap = load_target_names(args.targets_metadata)
    df["target_name"] = df["target_id"].map(tmap).fillna(df["target_id"])
    missing = df[df["target_name"] == df["target_id"]]["target_id"].unique()
    if len(missing):
        print(f"  Warning: {len(missing)} targets not in metadata: {missing}")

    print("\n[2/2] Loading known PD drugs...")
    known_pd = load_known_pd_drugs(args.pd_indications_csv)
    df = flag_known_pd(df, known_pd)

    # Run analyses
    analyse_target_distribution(df, args.out_dir)
    analyse_score_distribution(df, args.high_conf_threshold, args.out_dir)
    hits  = analyse_known_pd_drugs(df, args.out_dir)
    novel = analyse_novel_candidates(df, args.high_conf_threshold,
                                     args.memorisation_target, args.out_dir)

    # Summary
    above = (df["score"] >= args.high_conf_threshold).sum()
    above_excl_gr = (
        (df["score"] >= args.high_conf_threshold) &
        (~df["is_known_pd"]) &
        (df["target_name"] != args.memorisation_target)
    ).sum()

    summary = {
        "total_predictions":              len(df),
        "unique_targets":                 df["target_id"].nunique(),
        "unique_drugs":                   df["drug_id"].nunique(),
        "known_pd_hits":                  int(hits["drug_id"].nunique()),
        "known_pd_list_size":             len(known_pd),
        "pairs_above_threshold":          int(above),
        "novel_high_conf_excl_gr":        int(above_excl_gr),
        "threshold_used":                 args.high_conf_threshold,
        "memorisation_target_excluded":   args.memorisation_target,
        "mean_score":                     round(float(df["score"].mean()), 4),
        "top_target":                     df["target_name"].value_counts().index[0],
        "top_target_pct":                 round(df["target_name"].value_counts().iloc[0] / len(df) * 100, 1),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()