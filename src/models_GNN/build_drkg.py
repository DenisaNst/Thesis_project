"""
build_pd_drkg_graph.py
-----------------------
Builds a Parkinson's-specific heterogeneous graph from DRKG.

Key fix vs previous version:
  All node type and relation names are sanitized before being used as
  PyG HeteroData keys. DRKG names contain ::, +, -, spaces and other
  characters that cause PyG's to_hetero to mismap edge_index_dict entries,
  producing (None, None) edge indices and crashing SAGEConv.propagate.

  "Hetionet::CbG::Compound:Gene"  ->  "Hetionet__CbG__Compound_Gene"
  "Biological Process"            ->  "Biological_Process"
  "GNBR::A+::Compound:Gene"       ->  "GNBR__A___Compound_Gene"
"""

from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRKG_TSV     = PROJECT_ROOT / "data" / "raw"  / "drkg" / "drkg.tsv"
DRKG_EMB_NPY = PROJECT_ROOT / "data" / "raw"  / "drkg" / "embed" / "DRKG_TransE_l2_entity.npy"
DRKG_ENT_TSV = PROJECT_ROOT / "data" / "raw"  / "drkg" / "embed" / "entities.tsv"
DRUG_EMB_CSV = PROJECT_ROOT / "data" / "processed" / "chembl_drug_embeddings.csv"
TARG_EMB_CSV = PROJECT_ROOT / "data" / "processed" / "drkg_target_embeddings.csv"
INTER_CSV    = PROJECT_ROOT / "data" / "raw"  / "chembl_pd_interactions.csv"

CUTOFF_YEAR  = 2018

_RAW_KEEP = {"Compound", "Gene", "Disease",
             "Biological Process", "Molecular Function", "Pathway"}

def _san(s: str) -> str:
    """Replace any character that is not a letter, digit, or underscore."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)

KEEP_TYPES = {_san(t) for t in _RAW_KEEP}

_RAW_PRED_REL = "Hetionet::CbG::Compound:Gene"
PRED_REL  = _san(_RAW_PRED_REL)
PRED_SRC  = "Compound"
PRED_DST  = "Gene"
PRED_ETYPE = (PRED_SRC, PRED_REL, PRED_DST)


def _raw_node_type(entity: str) -> str:
    """Extract raw (unsanitized) node type from DRKG entity name."""
    return entity.split("::")[0]


def build_pd_drkg_graph() -> tuple[HeteroData, dict, dict, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    data         : HeteroData with sanitized node/edge type names
    node_to_idx  : {sanitized_ntype: {entity_name: int_idx}}
    idx_to_node  : {sanitized_ntype: {int_idx: entity_name}}
    train_edges  : np.ndarray (N, 3)  [src, dst, label]  pre-2018
    test_edges   : np.ndarray (N, 3)  [src, dst, label]  post-2018
    """

    print("Loading DRKG")
    df = pd.read_csv(DRKG_TSV, sep="\t", header=None,
                     names=["head", "relation", "tail"]).dropna()
    print(f"  Total triples: {len(df):,}")

    df["head_raw_type"] = df["head"].map(_raw_node_type)
    df["tail_raw_type"] = df["tail"].map(_raw_node_type)
    df = df[
        df["head_raw_type"].isin(_RAW_KEEP) &
        df["tail_raw_type"].isin(_RAW_KEEP)
    ].reset_index(drop=True)
    print(f"  After node-type filter: {len(df):,}")

    # Sanitize node types and relation names
    df["head_type"] = df["head_raw_type"].map(_san)
    df["tail_type"] = df["tail_raw_type"].map(_san)
    df["relation"]  = df["relation"].map(_san)


    print("Building node dictionaries")
    node_to_idx: dict[str, dict[str, int]] = {t: {} for t in KEEP_TYPES}

    for _, row in df[["head", "head_type", "tail", "tail_type"]].iterrows():
        for entity, ntype in ((row["head"], row["head_type"]),
                              (row["tail"], row["tail_type"])):
            if entity not in node_to_idx[ntype]:
                node_to_idx[ntype][entity] = len(node_to_idx[ntype])

    idx_to_node = {
        nt: {i: e for e, i in m.items()}
        for nt, m in node_to_idx.items()
    }
    for nt, m in node_to_idx.items():
        print(f"  {nt}: {len(m):,} nodes")

    print("Loading embeddings")
    emb_matrix  = np.load(str(DRKG_EMB_NPY))
    ent_df      = pd.read_csv(DRKG_ENT_TSV, sep="\t",
                              header=None, names=["entity", "idx"])
    ent_to_drkg = dict(zip(ent_df["entity"], ent_df["idx"].astype(int)))
    drkg_dim    = emb_matrix.shape[1]   # 400

    drug_emb_df   = pd.read_csv(DRUG_EMB_CSV).rename(
        columns={"molecule_chembl_id": "drug_id"})
    drug_emb_cols = [c for c in drug_emb_df.columns if c.startswith("drug_emb_")]
    chembl_dim    = len(drug_emb_cols)
    chembl_to_emb = {
        row["drug_id"]: row[drug_emb_cols].values.astype(np.float32)
        for _, row in drug_emb_df.iterrows()
    }

    targ_emb_df = pd.read_csv(TARG_EMB_CSV)
    chembl_to_drkg_entity = (
        dict(zip(targ_emb_df["target_id"], targ_emb_df["drkg_entity"]))
        if "drkg_entity" in targ_emb_df.columns else {}
    )

    print("Building node feature tensors")
    node_features: dict[str, torch.Tensor] = {}

    for ntype, mapping in node_to_idx.items():
        n = len(mapping)
        x = np.zeros((n, drkg_dim), dtype=np.float32)
        found = 0
        for entity, i in mapping.items():
            drkg_idx = ent_to_drkg.get(entity)
            if drkg_idx is not None:
                x[i] = emb_matrix[drkg_idx]
                found += 1
        print(f"  {ntype}: {found}/{n} have DRKG TransE embeddings")
        node_features[ntype] = torch.tensor(x, dtype=torch.float32)


    print(" Building HeteroData")
    data = HeteroData()
    for ntype, feat in node_features.items():
        data[ntype].x = feat

    for (htype, rel, ttype), grp in df.groupby(
            ["head_type", "relation", "tail_type"]):
        h_map = node_to_idx[htype]
        t_map = node_to_idx[ttype]
        src   = torch.tensor([h_map[e] for e in grp["head"]], dtype=torch.long)
        dst   = torch.tensor([t_map[e] for e in grp["tail"]], dtype=torch.long)
        data[(htype, rel, ttype)].edge_index          = torch.stack([src, dst])
        data[(ttype, "rev_" + rel, htype)].edge_index = torch.stack([dst, src])

    n_et = len(data.edge_types)
    n_e  = sum(data[e].edge_index.shape[1] for e in data.edge_types)
    print(f"  Edge types : {n_et}")
    print(f"  Total edges: {n_e:,}")

    # Verify prediction edge type exists
    if PRED_ETYPE not in data.edge_types:
        available = [et for et in data.edge_types
                     if et[0] == PRED_SRC and et[2] == PRED_DST]
        print(f"  [warn] {PRED_ETYPE} not found.")
        print(f"  Available Compound->Gene types: {available[:5]}")

    print("Building time-split prediction edges")
    inter = pd.read_csv(INTER_CSV).rename(columns={
        "molecule_chembl_id": "drug_id",
        "target_chembl_id":   "target_id",
    })
    if "label" not in inter.columns:
        inter["label"] = (
            pd.to_numeric(inter.get("pchembl_value"), errors="coerce") >= 6.0
        ).astype(int) if "pchembl_value" in inter.columns else 1
    inter["year"] = pd.to_numeric(
        inter.get("year", pd.Series(dtype=float)), errors="coerce")
    if "pchembl_value" in inter.columns:
        inter = inter.sort_values("pchembl_value", ascending=False, na_position="last")
    inter = inter.drop_duplicates(subset=["drug_id", "target_id"])

    comp_map = node_to_idx.get("Compound", {})
    gene_map = node_to_idx.get("Gene", {})

    def _make_edges(split_df: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in split_df.iterrows():
            d_key = f"Compound::{row['drug_id']}"
            t_key = chembl_to_drkg_entity.get(row["target_id"])
            if d_key in comp_map and t_key and t_key in gene_map:
                rows.append([comp_map[d_key], gene_map[t_key], int(row["label"])])
        return (np.array(rows, dtype=np.int64) if rows
                else np.zeros((0, 3), dtype=np.int64))

    train_edges = _make_edges(inter[inter["year"] <= CUTOFF_YEAR])
    test_edges  = _make_edges(inter[inter["year"] >  CUTOFF_YEAR])
    print(f"  Train edges: {len(train_edges):,}  |  Test edges: {len(test_edges):,}")

    return data, node_to_idx, idx_to_node, train_edges, test_edges


if __name__ == "__main__":
    data, node_to_idx, idx_to_node, train_edges, test_edges = build_pd_drkg_graph()
    print(f"\nDone. Train: {len(train_edges)}  Test: {len(test_edges)}")
    print(f"Prediction edge type: {PRED_ETYPE}")
    print(f"In graph: {PRED_ETYPE in data.edge_types}")