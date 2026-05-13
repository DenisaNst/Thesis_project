"""
gnn_saliency.py
----------------
Gradient-based saliency and subgraph extraction for PDHeteroGNN.

WHY THE ORIGINAL compute_gnn_saliency WAS BROKEN
-------------------------------------------------
Bug 1 — Wrong API:
    Called model(x, edge_index, batch=batch).
    PDHeteroGNN has no forward() method — only encode() and score_pairs().

Bug 2 — Wrong input types:
    Passed a flat tensor x and a single edge_index.
    PDHeteroGNN expects dicts: x_dict and edge_index_dict.

Bug 3 — Spurious batch argument:
    PDHeteroGNN never accepts batch — PyG handles it via HeteroData.

Functions
---------
compute_hetero_gnn_saliency     — node-level gradient saliency (fixed)
compute_hetero_edge_saliency    — edge-level saliency via endpoint proxy
extract_explanatory_subgraph    — filters graph to salient nodes + edges
visualize_explanatory_subgraph  — NetworkX plot coloured by saliency
"""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None  # type: ignore


def compute_hetero_gnn_saliency(
    model,
    data: "HeteroData",
    drug_idx: int,
    target_idx: int,
) -> dict[str, np.ndarray]:
    """
    Gradient-based node saliency for PDHeteroGNN.

    Enables gradients on all node feature tensors, runs a forward pass
    through encode() + score_pairs(), back-propagates the prediction logit,
    and returns the absolute gradient magnitude (summed over features)
    as a saliency score per node — normalised to [0, 1] per node type.

    Parameters
    ----------
    model      : PDHeteroGNN in eval mode
    data       : HeteroData with x_dict and edge_index_dict populated
    drug_idx   : integer index of the drug node to explain
    target_idx : integer index of the target node to explain

    Returns
    -------
    dict of {node_type: np.ndarray of shape (num_nodes,)}
    """
    if torch is None:
        raise ImportError("PyTorch is required for GNN saliency.")

    model.eval()

    x_dict_grad = {
        ntype: data[ntype].x.clone().detach().requires_grad_(True)
        for ntype in data.node_types
    }

    z_dict = model.encode(x_dict_grad, data.edge_index_dict)

    edge_label_index = torch.tensor(
        [[drug_idx], [target_idx]],
        dtype=torch.long,
        device=z_dict["drug"].device,
    )
    score = model.score_pairs(z_dict, edge_label_index)
    score.backward()

    saliency: dict[str, np.ndarray] = {}
    for ntype, x_t in x_dict_grad.items():
        if x_t.grad is not None:
            sal = x_t.grad.abs().sum(dim=1).detach().cpu().numpy()
        else:
            sal = np.zeros(x_t.size(0), dtype=np.float32)

        max_val = sal.max()
        saliency[ntype] = sal / (max_val + 1e-8) if max_val > 0 else sal

    return saliency


def compute_hetero_edge_saliency(
    model,
    data: "HeteroData",
    drug_idx: int,
    target_idx: int,
    edge_type: tuple[str, str, str] = ("drug", "binds_to", "target"),
) -> np.ndarray:
    """
    Edge-level saliency via endpoint node proxy.

    Scores each edge as the mean saliency of its two endpoint nodes.
    Useful for highlighting which drug-target edges drive a prediction.

    Returns
    -------
    np.ndarray of shape (num_edges,) with values in [0, 1]
    """
    node_saliency = compute_hetero_gnn_saliency(model, data, drug_idx, target_idx)

    src_type, _, dst_type = edge_type
    edge_index = data[edge_type].edge_index

    src_sal = node_saliency.get(src_type, np.zeros(data[src_type].x.size(0)))
    dst_sal = node_saliency.get(dst_type, np.zeros(data[dst_type].x.size(0)))

    src_scores = src_sal[edge_index[0].cpu().numpy()]
    dst_scores = dst_sal[edge_index[1].cpu().numpy()]

    edge_sal = (src_scores + dst_scores) / 2.0
    max_val = edge_sal.max()
    return edge_sal / (max_val + 1e-8) if max_val > 0 else edge_sal


def extract_explanatory_subgraph(
    data: "HeteroData",
    saliency_dict: dict[str, np.ndarray],
    drug_idx: int,
    target_idx: int,
    saliency_threshold: float = 0.3,
    top_k_nodes: int = 10,
) -> dict:
    """
    Extracts the explanatory subgraph for a (drug, target) prediction.

    The anchor pair is always kept. For all other nodes, only those above
    saliency_threshold are retained, capped at top_k_nodes per type.
    Only edges connecting retained nodes are included.

    Returns
    -------
    dict with keys:
      "nodes"  -> {node_type: list of (node_idx, saliency_score)}
      "edges"  -> {edge_type_str: list of (src_idx, dst_idx)}
      "anchor" -> {"drug": drug_idx, "target": target_idx}
    """
    retained_nodes: dict[str, list[tuple[int, float]]] = {}

    for ntype, sal in saliency_dict.items():
        sorted_idx = np.argsort(sal)[::-1]

        if ntype == "drug":
            others = [
                (int(i), float(sal[i]))
                for i in sorted_idx
                if sal[i] >= saliency_threshold and i != drug_idx
            ][:top_k_nodes]
            retained_nodes["drug"] = [(drug_idx, float(sal[drug_idx]))] + others

        elif ntype == "target":
            others = [
                (int(i), float(sal[i]))
                for i in sorted_idx
                if sal[i] >= saliency_threshold and i != target_idx
            ][:top_k_nodes]
            retained_nodes["target"] = [(target_idx, float(sal[target_idx]))] + others

        else:
            above = [
                (int(i), float(sal[i]))
                for i in sorted_idx
                if sal[i] >= saliency_threshold
            ][:top_k_nodes]
            if above:
                retained_nodes[ntype] = above

    retained_sets = {
        ntype: {idx for idx, _ in pairs}
        for ntype, pairs in retained_nodes.items()
    }

    retained_edges: dict[str, list[tuple[int, int]]] = {}
    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        src_set = retained_sets.get(src_type, set())
        dst_set = retained_sets.get(dst_type, set())

        if not src_set or not dst_set:
            continue

        edge_index = data[edge_type].edge_index.cpu().numpy()
        mask = (
            np.isin(edge_index[0], list(src_set)) &
            np.isin(edge_index[1], list(dst_set))
        )
        if mask.any():
            edge_key = f"{src_type}__{rel}__{dst_type}"
            retained_edges[edge_key] = list(
                zip(edge_index[0][mask].tolist(), edge_index[1][mask].tolist())
            )

    return {
        "nodes": retained_nodes,
        "edges": retained_edges,
        "anchor": {"drug": drug_idx, "target": target_idx},
    }


def visualize_explanatory_subgraph(
    subgraph: dict,
    node_id_map: Optional[dict[str, dict[int, str]]] = None,
    title: str = "Explanatory Subgraph",
    output_path: Optional[str] = None,
):
    """
    Draws the explanatory subgraph coloured by node type and saliency score.

    Parameters
    ----------
    subgraph     : output of extract_explanatory_subgraph
    node_id_map  : optional {node_type: {int_index: human_readable_name}}
                   e.g. {"drug": {0: "Levodopa"}, "target": {0: "LRRK2"}}
    title        : plot title
    output_path  : saves to this path if given, otherwise calls plt.show()
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("networkx and matplotlib are required for visualization.")

    TYPE_COLORS = {
        "drug": "#3B82F6",
        "target": "#10B981",
        "phenotype": "#F59E0B",
    }

    def _label(ntype: str, idx: int) -> str:
        if node_id_map and ntype in node_id_map:
            return node_id_map[ntype].get(idx, f"{ntype}_{idx}")
        return f"{ntype}_{idx}"

    G = nx.DiGraph()

    for ntype, pairs in subgraph["nodes"].items():
        for idx, sal in pairs:
            is_anchor = (
                (ntype == "drug"   and idx == subgraph["anchor"]["drug"]) or
                (ntype == "target" and idx == subgraph["anchor"]["target"])
            )
            G.add_node(_label(ntype, idx), node_type=ntype, saliency=sal, is_anchor=is_anchor)

    for edge_key, edge_list in subgraph["edges"].items():
        parts = edge_key.split("__")
        src_type, dst_type = parts[0], parts[2]
        for src_idx, dst_idx in edge_list:
            src_name = _label(src_type, src_idx)
            dst_name = _label(dst_type, dst_idx)
            if G.has_node(src_name) and G.has_node(dst_name):
                G.add_edge(src_name, dst_name)

    if not G.nodes:
        print("[warn] No nodes to visualize.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")
    pos = nx.spring_layout(G, seed=42, k=2.5)

    for ntype, color in TYPE_COLORS.items():
        nodes = [n for n in G.nodes if G.nodes[n].get("node_type") == ntype]
        if not nodes:
            continue
        sizes = [1800 if G.nodes[n]["is_anchor"] else 400 + 800 * G.nodes[n]["saliency"] for n in nodes]
        borders = [3 if G.nodes[n]["is_anchor"] else 1 for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color,
                               node_size=sizes, edgecolors="black",
                               linewidths=borders, alpha=0.9, ax=ax, label=ntype)

    nx.draw_networkx_edges(G, pos, edge_color="#94A3B8", arrows=True,
                           arrowstyle="-|>", arrowsize=15, width=1.5, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(scatterpoints=1, loc="upper left", fontsize=9)
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Saliency Score", fraction=0.03, pad=0.04)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {output_path}")
    else:
        plt.show()
    plt.close(fig)