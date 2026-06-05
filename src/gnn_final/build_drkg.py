"""
This script constructs the heterogeneous graph structure required by the PDHeteroGNN
model, preparing the nodes, edges, and features for bidirectional message passing.

Design:
  TRAINING signal     = ALL Compound-binds-Gene (CbG) edges in DRKG, exclusing saliency candidate
                        edges.
  TEST signal         = High-confidence Random Forest consensus predictions
                        (saliency_candidates_all.csv).
  Node initialization = Pre-trained 400-dimensional TransE embeddings. Replaces
                        naive random vectors with topological context.
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
DRKG_ENTITIES_TSV = PROJECT_ROOT / "data" / "raw" / "drkg" / "embed" / "entities.tsv"
DRKG_EMBED_NPY    = PROJECT_ROOT / "data" / "raw" / "drkg" / "embed" / "DRKG_TransE_l2_entity.npy"

KEEP = {"Compound", "Gene", "Disease",
             "Biological Process", "Molecular Function", "Pathway",
             "Cellular Component", "Anatomy",
             "Symptom", "Side Effect", "Tax",
             "Pharmacologic Class", "Atc"}

_RAW_CbG_REL = "Hetionet::CbG::Compound:Gene"

def _san(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)

KEEP_TYPES = {_san(t) for t in KEEP}

PRED_REL   = _san(_RAW_CbG_REL)
PRED_SRC   = "Compound"
PRED_DST   = "Gene"
PRED_ETYPE = (PRED_SRC, PRED_REL, PRED_DST)


def _raw_node_type(entity: str) -> str:
    return entity.split("::")[0]


def build_pd_drkg_graph() -> tuple[
        HeteroData, dict, dict, np.ndarray, np.ndarray]:
    print("Loading DRKG triples")
    df = pd.read_csv(DRKG_TSV, sep="\t", header=None,
                     names=["head", "relation", "tail"]).dropna()

    df["head_raw_type"] = df["head"].map(_raw_node_type)
    df["tail_raw_type"] = df["tail"].map(_raw_node_type)
    df = df[
        df["head_raw_type"].isin(KEEP) &
        df["tail_raw_type"].isin(KEEP)
    ].reset_index(drop=True)

    df["head_type"] = df["head_raw_type"].map(_san)
    df["tail_type"] = df["tail_raw_type"].map(_san)
    df["relation"]  = df["relation"].map(_san)

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

    comp_map = node_to_idx.get("Compound", {})
    gene_map = node_to_idx.get("Gene", {})


# Random initialization
    # print("Building node feature tensors (random initialization)")
    # node_features: dict[str, torch.Tensor] = {}
    #
    # np.random.seed(42)
    # torch.manual_seed(42)
    #
    # for ntype, mapping in node_to_idx.items():
    #     n = len(mapping)
    #     x = np.random.randn(n, 128).astype(np.float32)
    #     node_features[ntype] = torch.tensor(x, dtype=torch.float32)

# TransE initialization
    print("Loading pre-trained DRKG TransE embeddings")
    entity_df = pd.read_csv(DRKG_ENTITIES_TSV, sep="\t", header=None, names=["drkg_idx", "entity"])
    entity_to_drkg_idx = dict(zip(entity_df["entity"], entity_df["drkg_idx"]))

    # Load the actual 400-dimensional TransE vectors
    drkg_embeds = np.load(DRKG_EMBED_NPY)
    embed_dim = drkg_embeds.shape[1]

    node_features: dict[str, torch.Tensor] = {}
    rng = np.random.default_rng(42)

    for ntype, mapping in node_to_idx.items():
        n = len(mapping)
        x = np.zeros((n, embed_dim), dtype=np.float32)
        missing = 0

        for entity, local_idx in mapping.items():
            if entity in entity_to_drkg_idx:
                drkg_idx = entity_to_drkg_idx[entity]
                x[local_idx] = drkg_embeds[drkg_idx]
            else:
                # If an entity is somehow missing, fallback to random noise
                x[local_idx] = rng.standard_normal(embed_dim)
                missing += 1
        node_features[ntype] = torch.tensor(x, dtype=torch.float32)

    data = HeteroData()
    for ntype, feat in node_features.items():
        data[ntype].x = feat

    # Load saliency candidates to exclude from graph
    SALIENCY_CANDIDATES = PROJECT_ROOT / "artifacts" / "gnn_v2" / "saliency_candidates_all.csv"
    saliency_pairs = set()
    if SALIENCY_CANDIDATES.exists():
        sal_df = pd.read_csv(SALIENCY_CANDIDATES)
        for _, row in sal_df.iterrows():
            saliency_pairs.add((str(row["drkg_drug_key"]), str(row["drkg_target_key"])))
        print(f"Loaded {len(saliency_pairs):,} saliency pairs to exclude from graph")

    for (htype, rel, ttype), grp in df.groupby(
            ["head_type", "relation", "tail_type"]):
        h_map = node_to_idx[htype]
        t_map = node_to_idx[ttype]

        if htype == PRED_SRC and ttype == PRED_DST:
            mask = [
                (row["head"], row["tail"]) not in saliency_pairs
                for _, row in grp[["head", "tail"]].iterrows()
            ]
            grp = grp[mask]
        src = torch.tensor(
            [h_map[e] for e in grp["head"]], dtype=torch.long)
        dst = torch.tensor(
            [t_map[e] for e in grp["tail"]], dtype=torch.long)
        data[(htype, rel, ttype)].edge_index = torch.stack([src, dst])
        data[(ttype, "rev_" + rel, htype)].edge_index = torch.stack([dst, src])

    n_et = len(data.edge_types)
    n_e  = sum(data[e].edge_index.shape[1] for e in data.edge_types)
    print(f"Edge types: {n_et}")
    print(f"Total edges: {n_e:,}")

    # Build training edges from DRKG CbG
    san_cbg = _san(_RAW_CbG_REL)
    cbg_df  = df[
        (df["head_type"] == PRED_SRC) &
        (df["relation"]  == san_cbg) &
        (df["tail_type"] == PRED_DST)
    ].copy()

    train_rows = []
    for _, row in cbg_df.iterrows():
        h_idx = comp_map.get(row["head"])
        t_idx = gene_map.get(row["tail"])
        if h_idx is not None and t_idx is not None:
            train_rows.append([h_idx, t_idx, 1])

    train_edges = (np.array(train_rows, dtype=np.int64)
                   if train_rows
                   else np.zeros((0, 3), dtype=np.int64))

    test_edges = np.zeros((0, 3), dtype=np.int64)  # Empty, used saliency in train_gnn.py
    print(f"Final training edges: {len(train_edges):,}")

    return data, node_to_idx, idx_to_node, train_edges, test_edges


if __name__ == "__main__":
    data, node_to_idx, idx_to_node, train_edges, test_edges = \
        build_pd_drkg_graph()