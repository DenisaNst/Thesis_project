"""
build_drkg_pd.py
----------------
IMPROVED version of build_drkg.py with two key changes:

Change 1 — PD-specific subgraph filtering
    Instead of using the full 9.8M edge DRKG, we keep only nodes
    within 2 hops of the Parkinson's disease node. This reduces
    oversmoothing because each compound's neighbourhood becomes
    distinctive rather than a blur of the entire biological database.

Change 2 — ChEMBL Molecular Transformer embeddings for Compound nodes
    Compound nodes use your pre-computed ChEMBL MT embeddings instead
    of TransE. MT embeddings encode chemical structure, which is
    directly relevant for drug-target prediction. All other node types
    still use TransE.

To compare both approaches run:
    python src/models_GNN/train_gnn.py          # original (full DRKG + TransE)
    python src/models_GNN/train_gnn_pd.py       # improved (PD subgraph + MT emb)
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

# Parkinson's disease node in DRKG
PD_DISEASE_NODE = "Disease::MESH:D010300"

# Raw DRKG node type names
_RAW_KEEP = {"Compound", "Gene", "Disease",
             "Biological Process", "Molecular Function", "Pathway"}

def _san(s: str) -> str:
    """Sanitize string for use as PyG HeteroData key."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)

KEEP_TYPES = {_san(t) for t in _RAW_KEEP}

# Prediction edge type
_RAW_PRED_REL = "Hetionet::CbG::Compound:Gene"
PRED_REL   = _san(_RAW_PRED_REL)
PRED_SRC   = "Compound"
PRED_DST   = "Gene"
PRED_ETYPE = (PRED_SRC, PRED_REL, PRED_DST)


def _raw_node_type(entity: str) -> str:
    return entity.split("::")[0]


# ---------------------------------------------------------------------------
# NEW: PD subgraph filter
# ---------------------------------------------------------------------------

def _filter_to_pd_subgraph(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only nodes within 2 hops of the Parkinson's disease node.

    Hop 1: all triples directly involving the PD disease node.
            This gives us all compounds and genes directly linked to PD.
    Hop 2: all triples involving any node from hop 1.
            This gives us the biological context around those compounds
            and genes — their pathways, biological processes, etc.

    Why 2 hops and not more?
        On a dense biological graph, 3+ hops pulls in most of DRKG
        and defeats the purpose of filtering. 2 hops gives enough
        context while keeping the graph PD-specific.
    """
    print("  [PD filter] Filtering to 2-hop neighbourhood of PD node ...")

    # Check PD node exists
    pd_rows = df[(df["head"] == PD_DISEASE_NODE) |
                 (df["tail"] == PD_DISEASE_NODE)]
    if len(pd_rows) == 0:
        print(f"  [warn] PD node '{PD_DISEASE_NODE}' not found in DRKG.")
        print("  [warn] Skipping PD filter — using full graph.")
        return df

    # Hop 1: nodes directly connected to PD
    hop1_nodes = set(pd_rows["head"]) | set(pd_rows["tail"])
    print(f"  [PD filter] Hop-1 nodes: {len(hop1_nodes):,}")

    # Hop 2: nodes connected to any hop-1 node
    hop2_mask = df["head"].isin(hop1_nodes) | df["tail"].isin(hop1_nodes)
    hop2_df   = df[hop2_mask]
    hop2_nodes = set(hop2_df["head"]) | set(hop2_df["tail"])
    print(f"  [PD filter] Hop-2 nodes: {len(hop2_nodes):,}")

    # Keep triples where BOTH head and tail are in the 2-hop neighbourhood
    filtered = df[
        df["head"].isin(hop2_nodes) &
        df["tail"].isin(hop2_nodes)
    ].reset_index(drop=True)

    print(f"  [PD filter] Triples: {len(df):,} → {len(filtered):,} "
          f"({100*len(filtered)/len(df):.1f}% of original)")
    return filtered


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_pd_drkg_graph() -> tuple[HeteroData, dict, dict, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    data         : HeteroData (PD-specific subgraph, mixed embeddings)
    node_to_idx  : {sanitized_ntype: {entity_name: int_idx}}
    idx_to_node  : {sanitized_ntype: {int_idx: entity_name}}
    train_edges  : np.ndarray (N, 3)  [src, dst, label]  pre-2018
    test_edges   : np.ndarray (N, 3)  [src, dst, label]  post-2018
    node_feature_dims : {ntype: feature_dim}  for PDHeteroGNN input projections
    """

    # ------------------------------------------------------------------
    # 1. Load and filter DRKG to 6 node types
    # ------------------------------------------------------------------
    print("[1/6] Loading DRKG triples ...")
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

    # NEW: filter to PD-specific 2-hop subgraph
    df = _filter_to_pd_subgraph(df)

    # Sanitize names for PyG
    df["head_type"] = df["head_raw_type"].map(_san)
    df["tail_type"] = df["tail_raw_type"].map(_san)
    df["relation"]  = df["relation"].map(_san)

    # ------------------------------------------------------------------
    # 2. Build node dictionaries
    # ------------------------------------------------------------------
    print("[2/6] Building node dictionaries ...")
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

    # ------------------------------------------------------------------
    # 3. Load embeddings
    # ------------------------------------------------------------------
    print("[3/6] Loading embeddings ...")

    # TransE for all non-Compound node types
    emb_matrix  = np.load(str(DRKG_EMB_NPY))
    ent_df      = pd.read_csv(DRKG_ENT_TSV, sep="\t",
                              header=None, names=["entity", "idx"])
    ent_to_drkg = dict(zip(ent_df["entity"], ent_df["idx"].astype(int)))
    drkg_dim    = emb_matrix.shape[1]   # 400

    # NEW: ChEMBL Molecular Transformer embeddings for Compound nodes
    drug_emb_df   = pd.read_csv(DRUG_EMB_CSV).rename(
        columns={"molecule_chembl_id": "drug_id"})
    drug_emb_cols = [c for c in drug_emb_df.columns if c.startswith("drug_emb_")]
    chembl_dim    = len(drug_emb_cols)
    chembl_to_emb = {
        row["drug_id"]: row[drug_emb_cols].values.astype(np.float32)
        for _, row in drug_emb_df.iterrows()
    }
    print(f"  ChEMBL MT embeddings: {len(chembl_to_emb):,} compounds ({chembl_dim}d)")
    print(f"  DRKG TransE embeddings: {emb_matrix.shape[0]:,} entities ({drkg_dim}d)")

    # Target mapping for ChEMBL interactions
    targ_emb_df = pd.read_csv(TARG_EMB_CSV)
    chembl_to_drkg_entity = (
        dict(zip(targ_emb_df["target_id"], targ_emb_df["drkg_entity"]))
        if "drkg_entity" in targ_emb_df.columns else {}
    )

    # ------------------------------------------------------------------
    # 4. Build node feature tensors
    #    NEW: Compound → ChEMBL MT embeddings (chembl_dim)
    #         All others → DRKG TransE embeddings (drkg_dim)
    # ------------------------------------------------------------------
    print("[4/6] Building node feature tensors ...")
    node_features: dict[str, torch.Tensor] = {}
    node_feature_dims: dict[str, int] = {}  # for PDHeteroGNN input projections

    for ntype, mapping in node_to_idx.items():
        n = len(mapping)

        if ntype == "Compound":
            # Use ChEMBL Molecular Transformer embeddings
            # These encode chemical structure — directly relevant for
            # drug-target prediction, unlike TransE which encodes
            # graph proximity only
            x = np.zeros((n, chembl_dim), dtype=np.float32)
            found = 0
            for entity, i in mapping.items():
                # Entity format: "Compound::CHEMBL123456"
                chembl_id = entity.split("::")[-1]
                if chembl_id in chembl_to_emb:
                    x[i] = chembl_to_emb[chembl_id]
                    found += 1
            print(f"  Compound: {found}/{n} have ChEMBL MT embeddings ({chembl_dim}d)")
            feat_dim = chembl_dim

        else:
            # All other node types use DRKG TransE
            x = np.zeros((n, drkg_dim), dtype=np.float32)
            found = 0
            for entity, i in mapping.items():
                drkg_idx = ent_to_drkg.get(entity)
                if drkg_idx is not None:
                    x[i] = emb_matrix[drkg_idx]
                    found += 1
            print(f"  {ntype}: {found}/{n} have DRKG TransE embeddings ({drkg_dim}d)")
            feat_dim = drkg_dim

        node_features[ntype] = torch.tensor(x, dtype=torch.float32)
        node_feature_dims[ntype] = feat_dim

    # ------------------------------------------------------------------
    # 5. Build HeteroData with sanitized names + reverse edges
    # ------------------------------------------------------------------
    print("[5/6] Building HeteroData ...")
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

    if PRED_ETYPE not in data.edge_types:
        available = [et for et in data.edge_types
                     if et[0] == PRED_SRC and et[2] == PRED_DST]
        print(f"  [warn] {PRED_ETYPE} not found.")
        print(f"  Available Compound->Gene types: {available[:5]}")

    # ------------------------------------------------------------------
    # 6. Time-split prediction edges from ChEMBL interactions
    # ------------------------------------------------------------------
    print("[6/6] Building time-split prediction edges ...")
    inter = pd.read_csv(INTER_CSV).rename(columns={
        "molecule_chembl_id": "drug_id",
        "target_chembl_id":   "target_id",
    })

    # Label: 1 = active (pChEMBL >= 6), 0 = inactive (pChEMBL < 6)
    if "label" not in inter.columns:
        inter["label"] = (
            pd.to_numeric(inter.get("pchembl_value"), errors="coerce") >= 6.0
        ).astype(int) if "pchembl_value" in inter.columns else 1

    inter["year"] = pd.to_numeric(
        inter.get("year", pd.Series(dtype=float)), errors="coerce")
    if "pchembl_value" in inter.columns:
        inter = inter.sort_values("pchembl_value", ascending=False,
                                  na_position="last")
    inter = inter.drop_duplicates(subset=["drug_id", "target_id"])

    comp_map = node_to_idx.get("Compound", {})
    gene_map = node_to_idx.get("Gene", {})

    def _make_edges(split_df: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in split_df.iterrows():
            d_key = f"Compound::{row['drug_id']}"
            t_key = chembl_to_drkg_entity.get(row["target_id"])
            if d_key in comp_map and t_key and t_key in gene_map:
                rows.append([comp_map[d_key], gene_map[t_key],
                             int(row["label"])])
        return (np.array(rows, dtype=np.int64) if rows
                else np.zeros((0, 3), dtype=np.int64))

    train_edges = _make_edges(inter[inter["year"] <= CUTOFF_YEAR])
    test_edges  = _make_edges(inter[inter["year"] >  CUTOFF_YEAR])
    print(f"  Train edges: {len(train_edges):,}  |  "
          f"Test edges: {len(test_edges):,}")
    print(f"  Train active/inactive: "
          f"{(train_edges[:,2]==1).sum()}/{(train_edges[:,2]==0).sum()}")
    print(f"  Test  active/inactive: "
          f"{(test_edges[:,2]==1).sum()}/{(test_edges[:,2]==0).sum()}")

    return data, node_to_idx, idx_to_node, train_edges, test_edges, node_feature_dims


if __name__ == "__main__":
    data, node_to_idx, idx_to_node, train_edges, test_edges, feat_dims = \
        build_pd_drkg_graph()

    print(f"\nDone.")
    print(f"  Train: {len(train_edges)}  Test: {len(test_edges)}")
    print(f"  Prediction edge type: {PRED_ETYPE}")
    print(f"  In graph: {PRED_ETYPE in data.edge_types}")
    print(f"  Node feature dims: {feat_dims}")
    print(f"  Node types: {data.node_types}")
    print(f"  Edge types: {len(data.edge_types)}")