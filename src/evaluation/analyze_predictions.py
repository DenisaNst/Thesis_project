"""
Comprehensive analysis of FDA drug repositioning predictions from a Random Forest
model trained on drug-target interactions for Parkinson's Disease (PD). Validates
predictions against known PD drugs from ChEMBL and identifies novel high-confidence
repositioning candidates.

Key functionality:
  - Load and rank model predictions (drug-target pair scores)
  - Resolve target IDs to human-readable names from metadata
  - Load known PD-indicated drugs from ChEMBL EFO ontology query results
  - Flag predictions matching known PD drugs (validation of model quality)
  - Perform four complementary analyses:
    1. Target distribution — identify over-predicted targets (memorisation risk)
    2. Score distribution — compare scores for novel vs known PD drug pairs
    3. Known PD drug validation — check if model recovers known associations
    4. Novel candidates — extract high-confidence repositioning predictions

Output:
  - PNG visualizations: target distribution, score histograms with novel/known split
  - CSV files: known PD hits, best scores per drug, all novel candidates,
    high-confidence subset (with/without memorisation target exclusion),
    top 25 ranked novel candidates
  - JSON summary: overall statistics and threshold parameters

Dependencies:
  - pandas, numpy: Data manipulation and analysis
  - matplotlib: Visualization
  - requests: ChEMBL API access for drug name resolution
"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return (df.sort_values("score", ascending=False)
              .reset_index(drop=True)
              .assign(rank=lambda d: d.index + 1))


def load_target_names(metadata_csv: Path) -> dict:
    df = pd.read_csv(metadata_csv)
    df.columns = [c.lower().strip() for c in df.columns]
    id_col   = next((c for c in ["target_chembl_id", "target_id"] if c in df.columns), None)
    name_col = next((c for c in ["pref_name", "preferred_name", "name"] if c in df.columns), None)
    mapping = dict(zip(df[id_col].astype(str), df[name_col].astype(str)))
    return mapping


def load_known_pd_drugs(pd_indications_csv: Path) -> set:
    import requests

    ind_df = pd.read_csv(pd_indications_csv)
    ind_df.columns = [c.lower().strip() for c in ind_df.columns]
    mol_id_col = next((c for c in ["molecule_chembl_id", "mol_chembl_id", "chembl_id"]
                       if c in ind_df.columns), None)

    mol_ids = ind_df[mol_id_col].dropna().unique().tolist()
    print(f"  pd_indications.csv: {len(mol_ids)} PD-indicated molecules")

    cache_path = pd_indications_csv.parent / "pd_indication_names_cache.csv"
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
    else:
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
                print(f"    {i+1}/{len(mol_ids)} fetched")
        cache = pd.DataFrame(rows)
        cache.to_csv(cache_path, index=False)

    known = set(cache["pref_name"].str.lower().str.strip().dropna().tolist())
    return known


def flag_known_pd(df: pd.DataFrame, known_pd: set) -> pd.DataFrame:
    name_col = "drug_name" if "drug_name" in df.columns else "drug_id"
    df = df.copy()
    df["_lower"] = df[name_col].astype(str).str.lower().str.strip()
    df["is_known_pd"] = df["_lower"].apply(
        lambda n: any(pd_d in n or n in pd_d for pd_d in known_pd)
    )
    return df


def analyse_target_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    print("\n ANALYSIS 1 — Target Distribution")
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


def analyse_score_distribution(df: pd.DataFrame, threshold: float, out_dir: Path) -> None:
    print(f"\n ANALYSIS 2 — Score Distribution")

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


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
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


def analyse_known_pd_drugs(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    print("\n ANALYSIS 3 — Known PD Drug Validation")
    name_col = "drug_name" if "drug_name" in df.columns else "drug_id"
    hits = df[df["is_known_pd"]].copy()

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


def analyse_novel_candidates(
    df: pd.DataFrame,
    threshold: float,
    memorisation_target: str,
    out_dir: Path
) -> pd.DataFrame:
    print(f"\n ANALYSIS 4 — Novel Repositioning Candidates (score ≥ {threshold}) ")
    name_col = "drug_name" if "drug_name" in df.columns else "drug_id"

    novel = df[~df["is_known_pd"]].copy()

    high_conf = novel[
        (novel["score"] >= threshold) &
        (novel["target_name"] != memorisation_target)
    ].copy()

    high_conf_all = novel[novel["score"] >= threshold].copy()

    print(f"  Novel pairs total:                    {len(novel):,}")
    print(f"  Novel pairs score ≥ {threshold}:             {len(high_conf_all):,}")
    print(f"  Novel pairs score ≥ {threshold}, excl. GR:   {len(high_conf):,}")

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
    return top25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv",           type=Path,  default=Path("artifacts/rf_cv/fda_target_scores_all_esm2.csv"))
    parser.add_argument("--targets_metadata",     type=Path,  default=Path("data/raw/pd_targets_metadata.csv"))
    parser.add_argument("--pd_indications_csv",   type=Path,  default=Path("data/raw/pd_indications.csv"))
    parser.add_argument("--out_dir",              type=Path,  default=Path("artifacts/rf_cv/prediction_analysis_all"))
    parser.add_argument("--high_conf_threshold",  type=float, default=0.9,
                        help="Score threshold for high-confidence candidates (default 0.9)")
    parser.add_argument("--memorisation_target",  type=str,   default="Glucocorticoid receptor",
                        help="Target name to exclude from novel candidates (memorisation artifact)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_scores(args.scores_csv)

    tmap = load_target_names(args.targets_metadata)
    df["target_name"] = df["target_id"].map(tmap).fillna(df["target_id"])

    known_pd = load_known_pd_drugs(args.pd_indications_csv)
    df = flag_known_pd(df, known_pd)

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