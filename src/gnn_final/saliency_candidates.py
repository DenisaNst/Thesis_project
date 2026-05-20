"""
prepare_saliency_candidates.py
-------------------------------
Maps RF high-confidence FDA drug predictions (from BOTH ESM2 and DRKG
inference models) to DRKG node indices, producing three prioritised
candidate lists for GNN saliency map analysis.

Output candidate lists:
  1. BOTH models agree (score >= threshold in both)  <- highest priority
  2. ESM2 only high confidence
  3. DRKG only high confidence

Usage:
    python src/gnn_final/prepare_saliency_candidates.py
    python src/gnn_final/prepare_saliency_candidates.py --threshold 0.85
"""

from __future__ import annotations
from pathlib import Path
import sys
import argparse

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_drkg3 import build_pd_drkg_graph, PRED_SRC, PRED_DST

RF_ESM2   = PROJECT_ROOT / "artifacts" / "rf_inference_esm2" / "fda_target_scores_all.csv"
RF_DRKG   = PROJECT_ROOT / "artifacts" / "rf_inference_drkg" / "fda_target_scores_all.csv"
TARG_EMB  = PROJECT_ROOT / "data" / "processed" / "drkg_target_embeddings.csv"
TARG_META = PROJECT_ROOT / "data" / "raw" / "pd_targets_metadata.csv"
OUT_DIR   = PROJECT_ROOT / "artifacts" / "gnn_v2"


def _map_to_drkg(df, comp_map, gene_map, chembl_to_drkg, target_names):
    """
    Map a scored DataFrame to DRKG node indices.
    Input df must have columns: drug_id, drug_name, target_id, score
    Returns only rows where both drug and target exist in DRKG.
    """
    rows = []
    for _, row in df.iterrows():
        drug_id    = str(row["drug_id"])
        drug_name  = str(row.get("drug_name", drug_id))
        target_id  = str(row["target_id"])
        score      = float(row["score"])

        # DRKG stores DrugBank compounds as Compound::DBXXXXX (no DrugBank:: prefix)
        drkg_drug   = f"Compound::{drug_id}"
        drkg_target = chembl_to_drkg.get(target_id)
        tname       = target_names.get(target_id, target_id)

        if drkg_drug in comp_map and drkg_target and drkg_target in gene_map:
            rows.append({
                "drug_id":         drug_id,
                "drug_name":       drug_name,
                "target_id":       target_id,
                "target_name":     tname,
                "drkg_drug_key":   drkg_drug,
                "drkg_target_key": drkg_target,
                "drug_node_idx":   comp_map[drkg_drug],
                "target_node_idx": gene_map[drkg_target],
                "score":           score,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        "score", ascending=False).reset_index(drop=True)


def prepare_candidates(threshold: float = 0.9) -> dict[str, pd.DataFrame]:

    print(f"\n{'='*55}")
    print(f"  Saliency Candidate Preparation")
    print(f"  Score threshold : {threshold}")
    print(f"{'='*55}")

    # ------------------------------------------------------------------
    # 1. Load RF scores and apply threshold
    # ------------------------------------------------------------------
    esm2_all = pd.read_csv(RF_ESM2)
    drkg_all = pd.read_csv(RF_DRKG)

    esm2_high = esm2_all[esm2_all["score"] >= threshold].copy()
    drkg_high = drkg_all[drkg_all["score"] >= threshold].copy()

    print(f"\n  ESM2: {len(esm2_high):,} pairs >= {threshold} "
          f"({esm2_high['drug_id'].nunique():,} unique drugs)")
    print(f"  DRKG: {len(drkg_high):,} pairs >= {threshold} "
          f"({drkg_high['drug_id'].nunique():,} unique drugs)")

    # ------------------------------------------------------------------
    # 2. Cross-model overlap
    # ------------------------------------------------------------------
    esm2_pairs = set(zip(esm2_high["drug_id"], esm2_high["target_id"]))
    drkg_pairs = set(zip(drkg_high["drug_id"], drkg_high["target_id"]))

    both_pairs      = esm2_pairs & drkg_pairs
    esm2_only_pairs = esm2_pairs - drkg_pairs
    drkg_only_pairs = drkg_pairs - esm2_pairs

    print(f"\n  Cross-model overlap:")
    print(f"    Both models agree: {len(both_pairs):,} pairs  <- highest priority")
    print(f"    ESM2 only:         {len(esm2_only_pairs):,} pairs")
    print(f"    DRKG only:         {len(drkg_only_pairs):,} pairs")

    # ------------------------------------------------------------------
    # 3. Load DRKG graph and target mappings
    # ------------------------------------------------------------------
    print(f"\n  Loading DRKG graph ...")
    data, node_to_idx, idx_to_node, _, _ = build_pd_drkg_graph()
    comp_map = node_to_idx[PRED_SRC]
    gene_map = node_to_idx[PRED_DST]

    targ_df = pd.read_csv(TARG_EMB)
    chembl_to_drkg = dict(zip(targ_df["target_id"], targ_df["drkg_entity"]))

    target_names = {}
    if TARG_META.exists():
        meta = pd.read_csv(TARG_META)
        if "target_chembl_id" in meta.columns and "pref_name" in meta.columns:
            target_names = dict(zip(meta["target_chembl_id"], meta["pref_name"]))

    print(f"  Compound nodes:  {len(comp_map):,}")
    print(f"  Gene nodes:      {len(gene_map):,}")
    print(f"  Target mappings: {len(chembl_to_drkg):,}")

    # ------------------------------------------------------------------
    # 4. Build candidate DataFrames (each with clean drug_id, drug_name,
    #    target_id, score columns before mapping)
    # ------------------------------------------------------------------

    # BOTH — average of ESM2 and DRKG scores
    esm2_both = esm2_high[
        esm2_high.apply(
            lambda r: (r["drug_id"], r["target_id"]) in both_pairs, axis=1)
    ][["drug_id", "drug_name", "target_id", "score"]].copy()\
     .rename(columns={"score": "score_esm2"})

    drkg_both = drkg_high[
        drkg_high.apply(
            lambda r: (r["drug_id"], r["target_id"]) in both_pairs, axis=1)
    ][["drug_id", "target_id", "score"]].copy()\
     .rename(columns={"score": "score_drkg"})

    both_df = esm2_both.merge(drkg_both, on=["drug_id", "target_id"])
    both_df["score"] = (both_df["score_esm2"] + both_df["score_drkg"]) / 2

    # ESM2 only
    esm2_only_df = esm2_high[
        esm2_high.apply(
            lambda r: (r["drug_id"], r["target_id"]) in esm2_only_pairs, axis=1)
    ][["drug_id", "drug_name", "target_id", "score"]].copy()

    # DRKG only
    drkg_only_df = drkg_high[
        drkg_high.apply(
            lambda r: (r["drug_id"], r["target_id"]) in drkg_only_pairs, axis=1)
    ][["drug_id", "drug_name", "target_id", "score"]].copy()

    # ------------------------------------------------------------------
    # 5. Map each group to DRKG nodes
    # ------------------------------------------------------------------
    print(f"\n  Mapping to DRKG nodes ...")
    cands_both     = _map_to_drkg(both_df,     comp_map, gene_map,
                                  chembl_to_drkg, target_names)
    cands_esm2only = _map_to_drkg(esm2_only_df, comp_map, gene_map,
                                  chembl_to_drkg, target_names)
    cands_drkgonly = _map_to_drkg(drkg_only_df, comp_map, gene_map,
                                  chembl_to_drkg, target_names)

    # Add metadata columns
    if len(cands_both) > 0:
        cands_both["source"] = "both"
        lu = both_df.set_index(["drug_id", "target_id"])
        idx = pd.MultiIndex.from_frame(cands_both[["drug_id", "target_id"]])
        cands_both["score_esm2"] = lu["score_esm2"].reindex(idx).values
        cands_both["score_drkg"] = lu["score_drkg"].reindex(idx).values

    if len(cands_esm2only) > 0:
        cands_esm2only["source"] = "esm2_only"
    if len(cands_drkgonly) > 0:
        cands_drkgonly["source"] = "drkg_only"

    all_cands = pd.concat(
        [cands_both, cands_esm2only, cands_drkgonly],
        ignore_index=True)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  DRKG-mappable candidates")
    print(f"{'='*55}")
    print(f"  Both models agree: {len(cands_both):,}  <- use for saliency first")
    print(f"  ESM2 only:         {len(cands_esm2only):,}")
    print(f"  DRKG only:         {len(cands_drkgonly):,}")
    print(f"  Total:             {len(all_cands):,}")

    if len(cands_both) > 0:
        print(f"\n  Top 20 (both models):")
        print(f"  {'Drug':<25} {'Target':<35} {'ESM2':>6} {'DRKG':>6} {'Avg':>6}")
        print(f"  {'-'*25} {'-'*35} {'-'*6} {'-'*6} {'-'*6}")
        for _, r in cands_both.head(20).iterrows():
            print(f"  {r['drug_name']:<25} {r['target_name']:<35} "
                  f"{r['score_esm2']:>6.4f} {r['score_drkg']:>6.4f} "
                  f"{r['score']:>6.4f}")

        print(f"\n  By target:")
        tgt = cands_both.groupby(
            ["target_id", "target_name"]).size().reset_index(name="n")
        for _, r in tgt.sort_values("n", ascending=False).iterrows():
            print(f"    {r['target_name']:<40} {r['n']:>4} candidates")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cands_both.to_csv(    OUT_DIR / "saliency_candidates_both.csv",     index=False)
    cands_esm2only.to_csv(OUT_DIR / "saliency_candidates_esm2only.csv", index=False)
    cands_drkgonly.to_csv(OUT_DIR / "saliency_candidates_drkgonly.csv", index=False)
    all_cands.to_csv(     OUT_DIR / "saliency_candidates_all.csv",      index=False)

    print(f"\n  Saved to {OUT_DIR}/")
    print(f"    saliency_candidates_both.csv    <- start here for saliency maps")
    print(f"    saliency_candidates_esm2only.csv")
    print(f"    saliency_candidates_drkgonly.csv")
    print(f"    saliency_candidates_all.csv")

    return {
        "both":      cands_both,
        "esm2_only": cands_esm2only,
        "drkg_only": cands_drkgonly,
        "all":       all_cands,
    }


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.9)
    return p.parse_args()


if __name__ == "__main__":
    args       = _parse_args()
    candidates = prepare_candidates(args.threshold)
    both       = candidates["both"]
    print(f"\n  Ready: {len(both)} high-priority candidates mapped to DRKG")
    print(f"  Load with: pd.read_csv('artifacts/gnn_v2/saliency_candidates_both.csv')")