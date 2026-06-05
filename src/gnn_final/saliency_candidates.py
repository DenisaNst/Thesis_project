"""
Prepares high-confidence drug-target repositioning candidates for saliency map
generation. Combines predictions from two independent
Random Forest (RF) models (ESM2-based and DRKG-based), prioritizes consensus
predictions, and maps all candidates to DRKG graph node indices.

Key workflow:
  1. Load RF model predictions from both ESM2 and DRKG inference pipelines.
  2. Filter by confidence threshold (default: 0.9).
  3. Identify cross-model agreement: pairs predicted by both models, ESM2-only,
     and DRKG-only.
  4. Load DRKG knowledge graph and map drug-target pairs to node indices.
  5. Generate candidate sets with node indices for downstream graph attribution.
  6. Output prioritized candidate sets.

Usage:
  python src/gnn_final/saliency_candidates.py --threshold 0.9
"""

from __future__ import annotations
from pathlib import Path
import sys
import argparse

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_drkg import build_pd_drkg_graph, PRED_SRC, PRED_DST

RF_ESM2   = PROJECT_ROOT / "artifacts" / "rf_inference_esm2" / "fda_target_scores_all.csv"
RF_DRKG   = PROJECT_ROOT / "artifacts" / "rf_inference_drkg" / "fda_target_scores_all.csv"
TARG_EMB  = PROJECT_ROOT / "data" / "processed" / "drkg_target_embeddings.csv"
TARG_META = PROJECT_ROOT / "data" / "raw" / "pd_targets_metadata.csv"
OUT_DIR   = PROJECT_ROOT / "artifacts" / "gnn_3"


def _map_to_drkg(df, comp_map, gene_map, chembl_to_drkg, target_names):
    rows = []
    for _, row in df.iterrows():
        drug_id    = str(row["drug_id"])
        drug_name  = str(row.get("drug_name", drug_id))
        target_id  = str(row["target_id"])
        score      = float(row["score"])

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
    return pd.DataFrame(rows).sort_values(
        "score", ascending=False).reset_index(drop=True)


def prepare_candidates(threshold: float = 0.9) -> dict[str, pd.DataFrame]:
    esm2_all = pd.read_csv(RF_ESM2)
    drkg_all = pd.read_csv(RF_DRKG)

    esm2_high = esm2_all[esm2_all["score"] >= threshold].copy()
    drkg_high = drkg_all[drkg_all["score"] >= threshold].copy()

    esm2_pairs = set(zip(esm2_high["drug_id"], esm2_high["target_id"]))
    drkg_pairs = set(zip(drkg_high["drug_id"], drkg_high["target_id"]))

    both_pairs      = esm2_pairs & drkg_pairs
    esm2_only_pairs = esm2_pairs - drkg_pairs
    drkg_only_pairs = drkg_pairs - esm2_pairs

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


    cands_both     = _map_to_drkg(both_df,     comp_map, gene_map,
                                  chembl_to_drkg, target_names)
    cands_esm2only = _map_to_drkg(esm2_only_df, comp_map, gene_map,
                                  chembl_to_drkg, target_names)
    cands_drkgonly = _map_to_drkg(drkg_only_df, comp_map, gene_map,
                                  chembl_to_drkg, target_names)

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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cands_both.to_csv(    OUT_DIR / "saliency_candidates_both.csv",     index=False)
    cands_esm2only.to_csv(OUT_DIR / "saliency_candidates_esm2only.csv", index=False)
    cands_drkgonly.to_csv(OUT_DIR / "saliency_candidates_drkgonly.csv", index=False)
    all_cands.to_csv(     OUT_DIR / "saliency_candidates_all.csv",      index=False)

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