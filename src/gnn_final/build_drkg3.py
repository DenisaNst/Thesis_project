"""
build_drkg.py
-------------
Builds a heterogeneous graph from DRKG for GNN training.

Design:
  TRAINING signal  = ALL Compound-binds-Gene (CbG) edges in DRKG
                     This gives tens of thousands of drug-target
                     binding examples across all diseases, letting
                     the model learn general binding patterns.

  TEST signal      = ChEMBL PD-specific interactions
                     These are held out completely from training.
                     Label 1 = active (pChEMBL >= 6)
                     Label 0 = inactive (pChEMBL < 6)
                     Random 80/20 stratified split.

  LEAKAGE prevention:
                     Any CbG edge in DRKG that also appears in the
                     ChEMBL TEST set is removed from the training
                     graph before message passing. This ensures the
                     model cannot simply memorise test pairs from
                     the graph structure.

Why this is better than training on PD interactions only:
  - CbG edges in DRKG = tens of thousands of pairs (vs 725)
  - Model learns general binding patterns, not just PD
  - PD evaluation tests whether general knowledge transfers
  - Standard transfer learning approach used in literature

Returns
-------
data            : HeteroData  (graph used for message passing)
node_to_idx     : {ntype: {entity: int}}
idx_to_node     : {ntype: {int: entity}}
train_edges     : np.ndarray (N, 3) [src, dst, label]
                  derived from DRKG CbG edges (all label=1)
                  with random negatives sampled at call time
test_edges      : np.ndarray (N, 3) [src, dst, label]
                  ChEMBL PD interactions (label 0 or 1)
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

# Node types to keep
_RAW_KEEP = {"Compound", "Gene", "Disease",
             "Biological Process", "Molecular Function", "Pathway"}

# The DRKG relation that represents direct compound-gene binding
# This is what we train on — all binding edges across all diseases
_RAW_CbG_REL = "Hetionet::CbG::Compound:Gene"

def _san(s: str) -> str:
    """Sanitize string for PyG HeteroData keys."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)

KEEP_TYPES = {_san(t) for t in _RAW_KEEP}

# Prediction edge type (sanitized)
PRED_REL   = _san(_RAW_CbG_REL)
PRED_SRC   = "Compound"
PRED_DST   = "Gene"
PRED_ETYPE = (PRED_SRC, PRED_REL, PRED_DST)


def _raw_node_type(entity: str) -> str:
    return entity.split("::")[0]


def build_pd_drkg_graph() -> tuple[
        HeteroData, dict, dict, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    data        : HeteroData
    node_to_idx : dict
    idx_to_node : dict
    train_edges : np.ndarray (N, 3) — DRKG CbG positives, all label=1
    test_edges  : np.ndarray (N, 3) — ChEMBL PD interactions, label 0/1
    """

    # ------------------------------------------------------------------
    # 1. Load and filter DRKG to 6 node types
    # ------------------------------------------------------------------
    print("[1/7] Loading DRKG triples ...")
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

    df["head_type"] = df["head_raw_type"].map(_san)
    df["tail_type"] = df["tail_raw_type"].map(_san)
    df["relation"]  = df["relation"].map(_san)

    # ------------------------------------------------------------------
    # 2. Build node dictionaries
    # ------------------------------------------------------------------
    print("[2/7] Building node dictionaries ...")
    node_to_idx: dict[str, dict[str, int]] = {t: {} for t in KEEP_TYPES}

    for _, row in df[["head", "head_type",
                       "tail", "tail_type"]].iterrows():
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

    # ------------------------------------------------------------------
    # 3. Load ChEMBL PD interactions — build TEST SET first
    #    so we know which pairs to remove from training graph
    # ------------------------------------------------------------------
    print("[3/7] Loading ChEMBL PD interactions (test set) ...")
    inter = pd.read_csv(INTER_CSV).rename(columns={
        "molecule_chembl_id": "drug_id",
        "target_chembl_id":   "target_id",
    })

    # Label: 1 = active (pChEMBL >= 6), 0 = inactive
    inter["label"] = (
        pd.to_numeric(inter.get("pchembl_value"), errors="coerce") >= 6.0
    ).astype(int) if "pchembl_value" in inter.columns else 1

    if "pchembl_value" in inter.columns:
        inter = inter.sort_values("pchembl_value", ascending=False,
                                  na_position="last")
    inter = inter.drop_duplicates(subset=["drug_id", "target_id"])

    # Load target → DRKG entity mapping
    targ_emb_df = pd.read_csv(TARG_EMB_CSV)
    chembl_to_drkg_entity = (
        dict(zip(targ_emb_df["target_id"],
                 targ_emb_df["drkg_entity"]))
        if "drkg_entity" in targ_emb_df.columns else {}
    )

    comp_map = node_to_idx.get("Compound", {})
    gene_map = node_to_idx.get("Gene", {})

    # Build all ChEMBL PD pairs that map to DRKG nodes
    all_chembl_rows = []
    for _, row in inter.iterrows():
        d_key = f"Compound::{row['drug_id']}"
        t_key = chembl_to_drkg_entity.get(row["target_id"])
        if d_key in comp_map and t_key and t_key in gene_map:
            all_chembl_rows.append([
                comp_map[d_key],
                gene_map[t_key],
                int(row["label"]),
                d_key,   # keep entity names for leakage removal
                t_key,
            ])

    all_chembl = np.array(all_chembl_rows, dtype=object) \
        if all_chembl_rows else np.zeros((0, 5), dtype=object)

    # Stratified 80/20 split on ChEMBL pairs
    rng_split = np.random.default_rng(42)
    if len(all_chembl) > 0:
        labels     = all_chembl[:, 2].astype(int)
        pos_idx    = np.where(labels == 1)[0]
        neg_idx    = np.where(labels == 0)[0]
        rng_split.shuffle(pos_idx)
        rng_split.shuffle(neg_idx)

        n_pos_train = int(0.8 * len(pos_idx))
        n_neg_train = int(0.8 * len(neg_idx))

        # We only keep the test split from ChEMBL
        # train split is used only to define what NOT to leak
        chembl_train_pairs = set()
        for i in np.concatenate([pos_idx[:n_pos_train],
                                  neg_idx[:n_neg_train]]):
            chembl_train_pairs.add(
                (str(all_chembl[i, 3]), str(all_chembl[i, 4])))

        test_idx = np.concatenate([pos_idx[n_pos_train:],
                                   neg_idx[n_neg_train:]])
        test_chembl_pairs = set()
        for i in test_idx:
            test_chembl_pairs.add(
                (str(all_chembl[i, 3]), str(all_chembl[i, 4])))

        test_edges = all_chembl[test_idx][:, :3].astype(np.int64)
    else:
        test_edges         = np.zeros((0, 3), dtype=np.int64)
        test_chembl_pairs  = set()
        chembl_train_pairs = set()

    print(f"  ChEMBL PD pairs mapped to DRKG: {len(all_chembl):,}")
    print(f"  Test set: {len(test_edges)} pairs "
          f"(active={(test_edges[:,2]==1).sum() if len(test_edges)>0 else 0} "
          f"inactive={(test_edges[:,2]==0).sum() if len(test_edges)>0 else 0})")

    # ------------------------------------------------------------------
    # 4. Load embeddings
    # ------------------------------------------------------------------
    print("[4/7] Loading embeddings ...")
    emb_matrix  = np.load(str(DRKG_EMB_NPY))
    ent_df      = pd.read_csv(DRKG_ENT_TSV, sep="\t",
                              header=None, names=["entity", "idx"])
    ent_to_drkg = dict(zip(ent_df["entity"],
                           ent_df["idx"].astype(int)))
    drkg_dim    = emb_matrix.shape[1]  # 400

    # ------------------------------------------------------------------
    # 5. Build node feature tensors (TransE for all)
    # ------------------------------------------------------------------
    print("[5/7] Building node feature tensors ...")
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
        print(f"  {ntype}: {found}/{n} TransE embeddings")
        node_features[ntype] = torch.tensor(x, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 6. Build HeteroData
    #    Remove test CbG pairs from message passing graph
    #    to prevent leakage
    # ------------------------------------------------------------------
    print("[6/7] Building HeteroData ...")
    data = HeteroData()
    for ntype, feat in node_features.items():
        data[ntype].x = feat

    # Build set of test pairs to exclude from graph edges
    # (we exclude BOTH chembl train and test pairs to be safe —
    #  the graph should not contain any pair we evaluate on)
    all_chembl_entity_pairs = (
        test_chembl_pairs | chembl_train_pairs)

    for (htype, rel, ttype), grp in df.groupby(
            ["head_type", "relation", "tail_type"]):
        h_map = node_to_idx[htype]
        t_map = node_to_idx[ttype]

        # For the prediction edge type, remove ChEMBL pairs
        # to prevent data leakage into message passing
        if htype == PRED_SRC and ttype == PRED_DST:
            mask = [
                (row["head"], row["tail"])
                not in all_chembl_entity_pairs
                for _, row in grp[["head", "tail"]].iterrows()
            ]
            grp = grp[mask]
            print(f"  Removed {sum(~np.array(mask)):,} ChEMBL pairs "
                  f"from {rel} message passing edges")

        if len(grp) == 0:
            continue

        src = torch.tensor(
            [h_map[e] for e in grp["head"]], dtype=torch.long)
        dst = torch.tensor(
            [t_map[e] for e in grp["tail"]], dtype=torch.long)
        data[(htype, rel, ttype)].edge_index = (
            torch.stack([src, dst]))
        data[(ttype, "rev_" + rel, htype)].edge_index = (
            torch.stack([dst, src]))

    n_et = len(data.edge_types)
    n_e  = sum(data[e].edge_index.shape[1]
               for e in data.edge_types)
    print(f"  Edge types : {n_et}")
    print(f"  Total edges: {n_e:,}")

    # ------------------------------------------------------------------
    # 7. Build TRAINING edges from DRKG CbG edges
    #    These are ALL compound-binds-gene edges in DRKG
    #    minus any that appear in ChEMBL test set
    # ------------------------------------------------------------------
    print("[7/7] Building training edges from DRKG CbG ...")

    san_cbg = _san(_RAW_CbG_REL)
    cbg_df  = df[
        (df["head_type"] == PRED_SRC) &
        (df["relation"]  == san_cbg) &
        (df["tail_type"] == PRED_DST)
    ].copy()

    print(f"  Total CbG edges in DRKG: {len(cbg_df):,}")

    # Remove any pair that appears in ChEMBL test set
    # (to prevent the model being trained on its own test answers)
    train_rows = []
    skipped    = 0
    for _, row in cbg_df.iterrows():
        pair = (row["head"], row["tail"])
        if pair in test_chembl_pairs:
            skipped += 1
            continue
        h_idx = comp_map.get(row["head"])
        t_idx = gene_map.get(row["tail"])
        if h_idx is not None and t_idx is not None:
            train_rows.append([h_idx, t_idx, 1])

    train_edges = (np.array(train_rows, dtype=np.int64)
                   if train_rows
                   else np.zeros((0, 3), dtype=np.int64))

    print(f"  Removed {skipped} pairs overlapping with ChEMBL test")
    print(f"  Final training edges: {len(train_edges):,} "
          f"(all label=1, positives only)")
    print(f"  Test edges: {len(test_edges):,} "
          f"(active={(test_edges[:,2]==1).sum() if len(test_edges)>0 else 0} "
          f"inactive={(test_edges[:,2]==0).sum() if len(test_edges)>0 else 0})")
    print()
    print("  NOTE: Training edges are all positive (CbG from DRKG).")
    print("  Negatives are sampled randomly during training in")
    print("  train_gnn.py via sample_negatives().")

    return data, node_to_idx, idx_to_node, train_edges, test_edges


if __name__ == "__main__":
    data, node_to_idx, idx_to_node, train_edges, test_edges = \
        build_pd_drkg_graph()

    print(f"\nDone.")
    print(f"  Train edges: {len(train_edges):,}")
    print(f"  Test edges:  {len(test_edges):,}")
    print(f"  Prediction edge type: {PRED_ETYPE}")
    print(f"  In graph: {PRED_ETYPE in data.edge_types}")
    print(f"  Node types: {data.node_types}")