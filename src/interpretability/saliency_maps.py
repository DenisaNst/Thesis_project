"""
This loads the trained Heterogeneous GraphSAGE model (PDHeteroGNN), utilizing 400-dimensional
TransE embeddings, and performs input gradient analysis on the Drug Repurposing Knowledge Graph (DRKG).
Because the candidate edges were explicitly removed from the graph prior to training to prevent data leakage,
this script successfully forces the GNN to reveal the secondary "biological bridges" connecting a drug to a target.

Key functionality:
  - Gradient-based Saliency: Computes the derivative of the GNN prediction score with
    respect to all 97,238 node embeddings to determine mathematical importance.
  - Subgraph Extraction: Isolates the top-K most influential nodes and their connecting
    edges to form an explanatory subgraph, 15 in this script.
  - Accessible Saliency Visualization: Generates directed network diagrams and bar charts
    mapping node size to gradient importance.
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_final.build_drkg import build_pd_drkg_graph, PRED_SRC, PRED_DST
from gnn_final.GNN_pd import PDHeteroGNN

# Paths
MODEL_DIR    = PROJECT_ROOT / "artifacts" / "gnn_3"
CAND_CSV     = MODEL_DIR / "saliency_candidates_both.csv"
OUT_DIR      = MODEL_DIR / "saliency_maps"


def compute_saliency(model: PDHeteroGNN,
                     data,
                     drug_idx: int,
                     target_idx: int,
                     device: torch.device) -> dict[str, np.ndarray]:
    model.eval()

    x_dict_grad = {}
    for ntype, x in data.x_dict.items():
        x_grad = x.detach().clone().requires_grad_(True)
        x_dict_grad[ntype] = x_grad

    z_dict = model.encode(x_dict_grad, data.edge_index_dict)

    src_idx = torch.tensor([drug_idx],   dtype=torch.long, device=device)
    dst_idx = torch.tensor([target_idx], dtype=torch.long, device=device)
    edge_idx = torch.stack([src_idx, dst_idx])

    score = model.score_pairs(z_dict, edge_idx, PRED_SRC, PRED_DST)
    score.backward()

    saliency = {}
    for ntype, x_grad in x_dict_grad.items():
        if x_grad.grad is not None:
            saliency[ntype] = x_grad.grad.abs().mean(dim=1)\
                .detach().cpu().numpy()
        else:
            saliency[ntype] = np.zeros(x_grad.shape[0])
    return saliency


def get_top_nodes(saliency: dict[str, np.ndarray],
                  idx_to_node: dict[str, dict[int, str]],
                  drug_idx: int,
                  target_idx: int,
                  top_k: int = 15,
                  exclude_self: bool = True) -> list[dict]:
    all_nodes = []

    for ntype, scores in saliency.items():
        for node_idx, importance in enumerate(scores):
            if exclude_self:
                if ntype == PRED_SRC and node_idx == drug_idx:
                    continue
                if ntype == PRED_DST and node_idx == target_idx:
                    continue
            entity_name = idx_to_node.get(ntype, {}).get(
                node_idx, f"{ntype}_node_{node_idx}")
            all_nodes.append({
                "ntype":       ntype,
                "node_idx":    node_idx,
                "entity_name": entity_name,
                "importance":  float(importance),
            })

    all_nodes.sort(key=lambda x: x["importance"], reverse=True)
    return all_nodes[:top_k]


def extract_subgraph_around_nodes(data,
                                  node_to_idx: dict[str, dict[str, int]],
                                  idx_to_node: dict[str, dict[int, str]],
                                  drug_idx: int,
                                  drug_name: str,
                                  target_idx: int,
                                  target_name: str,
                                  top_nodes: list[dict]) -> dict:

    include_node_ids = set()
    include_node_ids.add((PRED_SRC, drug_idx))
    include_node_ids.add((PRED_DST, target_idx))

    for n in top_nodes:
        include_node_ids.add((n["ntype"], n["node_idx"]))
    subgraph_edges = []
    node_type_map = {}

    for edge_type, edge_idx in data.edge_index_dict.items():
        src_type, rel_type, dst_type = edge_type
        src_indices = edge_idx[0].cpu().numpy()
        dst_indices = edge_idx[1].cpu().numpy()

        for src_idx, dst_idx in zip(src_indices, dst_indices):
            src_node_id = (src_type, int(src_idx))
            dst_node_id = (dst_type, int(dst_idx))

            if src_node_id in include_node_ids and dst_node_id in include_node_ids:
                subgraph_edges.append({
                    "src": src_node_id,
                    "dst": dst_node_id,
                    "relation": rel_type,
                })
                node_type_map[src_node_id] = src_type
                node_type_map[dst_node_id] = dst_type
    node_importance = {}
    for n in top_nodes:
        node_importance[(n["ntype"], n["node_idx"])] = n["importance"]

    node_importance[(PRED_SRC, drug_idx)] = 1.0
    node_importance[(PRED_DST, target_idx)] = 1.0

    subgraph_nodes = []
    for node_id in include_node_ids:
        ntype, idx = node_id
        entity_name = idx_to_node.get(ntype, {}).get(
            idx, f"{ntype}_node_{idx}")
        importance = node_importance.get(node_id, 0.0)
        subgraph_nodes.append({
            "node_id": node_id,
            "ntype": ntype,
            "entity_name": entity_name,
            "importance": importance,
            "is_query_drug": (ntype == PRED_SRC and idx == drug_idx),
            "is_query_target": (ntype == PRED_DST and idx == target_idx),
        })

    return {
        "nodes": subgraph_nodes,
        "edges": subgraph_edges,
    }


def visualize_subgraph(subgraph_data: dict,
                       drug_name: str,
                       target_name: str,
                       rf_score: float,
                       output_path: Path) -> None:
    G = nx.DiGraph()

    node_color_map = {}
    node_size_map = {}

    for node_info in subgraph_data["nodes"]:
        node_id = node_info["node_id"]
        ntype = node_info["ntype"]
        entity = node_info["entity_name"]
        importance = node_info["importance"]
        if node_info["is_query_drug"]:
            clean_name = drug_name
        elif node_info["is_query_target"]:
            clean_name = target_name
        else:
            clean_name = entity
            for middle_tag in [
                "::9606::", "::DrugBank::", "::CHEMBL::", "::MESH:",
                "::GO:", "::UBERON:", "::UMLS:C", "::UMLS:",
                "::EPC:", "::MOA:", "::PE:", "::ATC:", "::TAX:"
            ]:
                clean_name = clean_name.replace(middle_tag, "::")
        if len(clean_name) > 35:
            clean_name = clean_name[:32] + "..."
        G.add_node(node_id, label=clean_name, importance=importance)

        if node_info["is_query_drug"]:
            node_color_map[node_id] = "#FF6B6B"
            node_size_map[node_id] = 3000
        elif node_info["is_query_target"]:
            node_color_map[node_id] = "#4ECDC4"
            node_size_map[node_id] = 3000
        else:
            type_colors = {
                "Compound": "#0072B2",
                "Biological_Process": "#D55E00",
                "Gene": "#E69F00",
                "Disease": "#009E73",
                "Pathway": "#56B4E9",
                "Molecular_Function": "#CC79A7",
                "Cellular_Component": "#F0E442",
                "Anatomy": "#7F7F7F",
                "Symptom": "#8C564B",
                "Side_Effect": "#17BECF",
                "Pharmacologic_Class": "#001C7F",
                "Atc": "#CAB2D6",
                "Tax": "#B2DF8A",
            }
            node_color_map[node_id] = type_colors.get(ntype, "#777777")
            node_size_map[node_id] = 500 + importance * 2000

    # Add edges
    edge_type_colors = {}
    for i, edge in enumerate(subgraph_data["edges"]):
        src = edge["src"]
        dst = edge["dst"]
        rel = edge["relation"]

        G.add_edge(src, dst, relation=rel)
        if rel not in edge_type_colors:
            colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]
            edge_type_colors[rel] = colors[len(edge_type_colors) % len(colors)]

    # Layout: spring layout for readability
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    for src, dst, rel_data in G.edges(data=True):
        rel = rel_data.get("relation", "unknown")
        color = edge_type_colors.get(rel, "#CCCCCC")
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(src, dst)],
            ax=ax,
            edge_color=color,
            edge_cmap=None,
            width=2,
            alpha=0.6,
            arrowsize=15,
            arrows=True,
            connectionstyle="arc3,rad=0.1",
        )
    nodes = list(G.nodes())
    colors = [node_color_map.get(n, "#777777") for n in nodes]
    sizes = [node_size_map.get(n, 500) for n in nodes]

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodes,
        node_color=colors,
        node_size=sizes,
        ax=ax,
        alpha=0.9,
    )

    labels = {n: G.nodes[n]["label"] for n in nodes}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=8,
        font_weight="bold",
        ax=ax,
    )

    ax.set_title(
        f"Explanatory Subgraph: {drug_name} → {target_name}\nRF Score: {rf_score:.4f}",
        fontsize=14, fontweight="bold", pad=20
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FF6B6B", label="Query Drug"),
        Patch(facecolor="#4ECDC4", label="Query Target"),
        Patch(facecolor="#0072B2", label="Compound"),
        Patch(facecolor="#D55E00", label="Biological Process"),
        Patch(facecolor="#E69F00", label="Gene"),
        Patch(facecolor="#009E73", label="Disease"),
        Patch(facecolor="#56B4E9", label="Pathway"),
        Patch(facecolor="#CC79A7", label="Molecular Function"),
        Patch(facecolor="#F0E442", label="Cellular Component"),
        Patch(facecolor="#7F7F7F", label="Anatomy"),
        Patch(facecolor="#8C564B", label="Symptom"),
        Patch(facecolor="#17BECF", label="Side Effect"),
        Patch(facecolor="#001C7F", label="Pharm. Class"),
        Patch(facecolor="#CAB2D6", label="ATC"),
        Patch(facecolor="#B2DF8A", label="Taxonomy"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_saliency_bars(top_nodes: list[dict],
                       drug_name: str,
                       target_name: str,
                       rf_score: float,
                       output_path: Path) -> None:
    labels = []
    for n in top_nodes:
        name = n["entity_name"]
        for prefix in ["Gene::9606::", "Compound::DrugBank::",
                        "Compound::CHEMBL::", "Disease::MESH:",
                        "Biological_Process::", "Molecular_Function::",
                        "Pathway::"]:
            name = name.replace(prefix, "")
        label = f"[{n['ntype'][:3].upper()}] {name[:40]}"
        labels.append(label)
    values = [n["importance"] for n in top_nodes]

    color_map = {
        "Compound": "#0072B2",
        "Biological_Process": "#D55E00",
        "Gene": "#E69F00",
        "Disease": "#009E73",
        "Pathway": "#56B4E9",
        "Molecular_Function": "#CC79A7",
        "Cellular_Component": "#F0E442",
        "Anatomy": "#7F7F7F",
        "Symptom": "#8C564B",
        "Side_Effect": "#17BECF",
        "Pharmacologic_Class": "#001C7F",
        "Atc": "#CAB2D6",
        "Tax": "#B2DF8A",
    }
    colors = [color_map.get(n["ntype"], "#777777") for n in top_nodes]
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.45)))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |Gradient| (importance)", fontsize=11)
    ax.set_title(
        f"GNN Saliency: {drug_name} → {target_name}\n"
        f"RF score: {rf_score:.4f}",
        fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_saliency_analysis(top_candidates: int = 20,
                          top_k_nodes: int = 15,
                          out_dir: Path = OUT_DIR) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates = pd.read_csv(CAND_CSV)
    candidates = candidates.sort_values(
        "score", ascending=False).head(top_candidates)

    data, node_to_idx, idx_to_node, _, _ = build_pd_drkg_graph()
    data = data.to(device)

    model_path    = MODEL_DIR / "gnn_model.pt"
    metadata_path = MODEL_DIR / "gnn_metadata.json"

    with open(metadata_path) as f:
        metadata = json.load(f)

    params = metadata["best_params"]
    model  = PDHeteroGNN(
        metadata=data.metadata(),
        hidden_channels=params["hidden_channels"],
        out_channels=params["out_channels"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    with torch.no_grad():
        model.encode(data.x_dict, data.edge_index_dict)

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print(f"\n  Computing saliency for {len(candidates)} candidates")
    results = []

    for i, row in candidates.iterrows():
        drug_name   = row["drug_name"]
        target_name = row["target_name"]
        drug_idx    = int(row["drug_node_idx"])
        target_idx  = int(row["target_node_idx"])
        rf_score    = float(row["score"])

        print(f"\n  [{i+1}/{len(candidates)}] "
              f"{drug_name} → {target_name} "
              f"(RF score: {rf_score:.4f})")

        try:
            saliency = compute_saliency(
                model, data, drug_idx, target_idx, device)

            top_nodes = get_top_nodes(
                saliency, idx_to_node,
                drug_idx, target_idx,
                top_k=top_k_nodes,
                exclude_self=True)

            subgraph = extract_subgraph_around_nodes(
                data, node_to_idx, idx_to_node,
                drug_idx, drug_name,
                target_idx, target_name,
                top_nodes)

            safe_name = f"{drug_name}_{target_name}"\
                .replace(" ", "_").replace("/", "_")[:60]

            # Generate subgraph visualization
            subgraph_path = out_dir / f"subgraph_{safe_name}.png"
            visualize_subgraph(
                subgraph, drug_name, target_name,
                rf_score, subgraph_path)

            # Also generate bar chart for comparison
            bar_path = out_dir / f"importance_{safe_name}.png"
            plot_saliency_bars(
                top_nodes, drug_name, target_name,
                rf_score, bar_path)

            for rank, n in enumerate(top_nodes):
                results.append({
                    "drug_name":    drug_name,
                    "target_name":  target_name,
                    "drug_id":      row["drug_id"],
                    "target_id":    row["target_id"],
                    "rf_score":     rf_score,
                    "rank":         rank + 1,
                    "ntype":        n["ntype"],
                    "entity_name":  n["entity_name"],
                    "importance":   n["importance"],
                })
        except Exception as e:
            print(f"    [ERROR] {drug_name} → {target_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    results_df = pd.DataFrame(results)
    out_csv    = out_dir / "saliency_results.csv"
    results_df.to_csv(out_csv, index=False)

    print(f"  Saliency analysis complete")
    print(f"  Candidates explained: "
          f"{results_df['drug_name'].nunique()}")
    print(f"  Results saved to: {out_csv}")
    print(f"  Subgraph visualization: {out_dir}/subgraph_*.png")
    print(f"  Importance bar charts:  {out_dir}/importance_*.png")

    if len(results_df) > 0:
        print(f"\n  Most commonly influential node types:")
        type_counts = results_df[results_df["rank"] <= 5]\
            .groupby("ntype")["entity_name"].count()\
            .sort_values(ascending=False)
        for ntype, count in type_counts.items():
            print(f"    {ntype:<25} {count:>4} appearances in top-5")

        print(f"\n  Most commonly influential entities (top-5 across all):")
        entity_counts = results_df[results_df["rank"] <= 5]\
            .groupby("entity_name")["drug_name"].count()\
            .sort_values(ascending=False).head(10)
        for entity, count in entity_counts.items():
            short = entity[:60]
            print(f"    {short:<60} {count:>4}×")

    return results_df


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--top_candidates", type=int, default=20,
                   help="Number of RF candidates to explain (default 20)")
    p.add_argument("--top_k", type=int, default=15,
                   help="Top-K influential nodes per prediction (default 15)")
    p.add_argument("--out_dir", type=Path, default=OUT_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    result = run_saliency_analysis(
        top_candidates=args.top_candidates,
        top_k_nodes=args.top_k,
        out_dir=args.out_dir)