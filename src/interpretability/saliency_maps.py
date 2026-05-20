"""
saliency_maps.py
-----------------
Generates saliency maps for RF top drug repositioning candidates
using the trained PDHeteroGNN model on the DRKG knowledge graph.

Purpose:
  The RF identified high-confidence FDA drug → PD target candidates
  using molecular embeddings. The GNN explains WHY those predictions
  make sense from a biological graph structure perspective.

  For each candidate (drug, target):
    → gradient of GNN score w.r.t. input node features
    → identifies which DRKG nodes most influenced the prediction
    → produces an explanatory subgraph: genes, pathways, diseases
      that biologically connect the drug to the target

Connection to RF:
  RF candidates come from saliency_candidates_both.csv
  (pairs where BOTH ESM2 and DRKG RF models gave score >= 0.9)
  These are the highest-confidence repositioning predictions.
  Saliency maps provide the biological narrative for each.

Usage:
    python src/gnn_final/saliency_maps.py
    python src/gnn_final/saliency_maps.py --top_k 10 --top_candidates 20
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_final.build_drkg3 import build_pd_drkg_graph, PRED_SRC, PRED_DST
from gnn_final.GNN_pd import PDHeteroGNN

# Paths
MODEL_DIR    = PROJECT_ROOT / "artifacts" / "gnn_v2"
CAND_CSV     = MODEL_DIR / "saliency_candidates_both.csv"
OUT_DIR      = MODEL_DIR / "saliency_maps"


# ---------------------------------------------------------------------------
# Core saliency computation
# ---------------------------------------------------------------------------

def compute_saliency(model: PDHeteroGNN,
                     data,
                     drug_idx: int,
                     target_idx: int,
                     device: torch.device) -> dict[str, np.ndarray]:
    """
    Compute input gradient saliency for a single drug-target pair.

    How it works:
      1. Enable requires_grad on input node features (x_dict)
         — this is where gradients must flow INTO, not the output
      2. Forward pass: x_dict → GNN encoder → z_dict (node embeddings)
      3. Score = dot product of drug and target embeddings
      4. Backpropagate score through the GNN
      5. Gradient magnitude at each node's input features
         = how much that node's features influenced the prediction
      6. Aggregate across embedding dimensions → one importance score
         per node

    Returns
    -------
    saliency : dict {ntype: np.ndarray of shape (n_nodes,)}
        Importance score for every node in the graph.
        Higher = more influential for this prediction.
    """
    model.eval()

    # Step 1: enable gradients on input features
    # Create new tensors with requires_grad=True so gradients flow back
    x_dict_grad = {}
    for ntype, x in data.x_dict.items():
        x_grad = x.detach().clone().requires_grad_(True)
        x_dict_grad[ntype] = x_grad

    # Step 2: forward pass with gradient-enabled inputs
    z_dict = model.encode(x_dict_grad, data.edge_index_dict)

    # Step 3: score the specific drug-target pair
    src_idx = torch.tensor([drug_idx],   dtype=torch.long, device=device)
    dst_idx = torch.tensor([target_idx], dtype=torch.long, device=device)
    edge_idx = torch.stack([src_idx, dst_idx])

    score = model.score_pairs(z_dict, edge_idx, PRED_SRC, PRED_DST)

    # Step 4: backpropagate
    score.backward()

    # Step 5: collect gradient magnitudes per node type
    saliency = {}
    for ntype, x_grad in x_dict_grad.items():
        if x_grad.grad is not None:
            # Mean absolute gradient across embedding dimensions
            # Shape: (n_nodes,) — one importance score per node
            saliency[ntype] = x_grad.grad.abs().mean(dim=1)\
                .detach().cpu().numpy()
        else:
            saliency[ntype] = np.zeros(x_grad.shape[0])

    return saliency


def get_top_nodes(saliency: dict[str, np.ndarray],
                  idx_to_node: dict[str, dict[int, str]],
                  drug_idx: int,
                  target_idx: int,
                  top_k: int = 10,
                  exclude_self: bool = True) -> list[dict]:
    """
    Extract the top-K most important nodes from saliency scores.

    Parameters
    ----------
    exclude_self : bool
        If True, exclude the drug and target nodes themselves —
        they are always important by definition and dominate the ranking.
        Set False to include them for completeness.

    Returns
    -------
    List of dicts: {ntype, node_idx, entity_name, importance}
    Sorted by importance descending.
    """
    all_nodes = []

    for ntype, scores in saliency.items():
        for node_idx, importance in enumerate(scores):
            # Optionally skip the query drug and target themselves
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


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_saliency(top_nodes: list[dict],
                  drug_name: str,
                  target_name: str,
                  rf_score: float,
                  output_path: Path) -> None:
    """
    Bar chart of top-K influential DRKG nodes for one drug-target pair.
    """
    if not top_nodes:
        return

    labels = []
    for n in top_nodes:
        # Shorten entity names for display
        name = n["entity_name"]
        # Strip DRKG prefix noise
        for prefix in ["Gene::9606::", "Compound::DrugBank::",
                        "Compound::CHEMBL::", "Disease::MESH:",
                        "Biological_Process::", "Molecular_Function::",
                        "Pathway::"]:
            name = name.replace(prefix, "")
        # Truncate long names
        label = f"[{n['ntype'][:3].upper()}] {name[:40]}"
        labels.append(label)

    values = [n["importance"] for n in top_nodes]

    # Color by node type
    color_map = {
        "Compound":           "#4C72B0",
        "Gene":               "#DD8452",
        "Disease":            "#55A868",
        "Biological_Process": "#C44E52",
        "Molecular_Function": "#8172B2",
        "Pathway":            "#937860",
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_saliency_analysis(top_candidates: int = 20,
                          top_k_nodes: int = 15,
                          out_dir: Path = OUT_DIR) -> pd.DataFrame:
    """
    Run saliency analysis for the top RF-identified drug repositioning
    candidates that are mappable to DRKG.

    Parameters
    ----------
    top_candidates : int
        Number of RF candidates to explain (sorted by RF score desc)
    top_k_nodes : int
        Number of top influential DRKG nodes to report per prediction
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load RF candidates
    # ------------------------------------------------------------------
    if not CAND_CSV.exists():
        raise FileNotFoundError(
            f"Candidates file not found: {CAND_CSV}\n"
            f"Run prepare_saliency_candidates.py first.")

    candidates = pd.read_csv(CAND_CSV)
    candidates = candidates.sort_values(
        "score", ascending=False).head(top_candidates)
    print(f"\n  RF candidates to explain: {len(candidates)}")
    print(f"  Score range: "
          f"{candidates['score'].min():.4f} – "
          f"{candidates['score'].max():.4f}")

    # ------------------------------------------------------------------
    # 2. Load DRKG graph
    # ------------------------------------------------------------------
    print("\n  Loading DRKG graph ...")
    data, node_to_idx, idx_to_node, _, _ = build_pd_drkg_graph()
    data = data.to(device)

    # ------------------------------------------------------------------
    # 3. Load trained GNN model
    # ------------------------------------------------------------------
    model_path    = MODEL_DIR / "gnn_model.pt"
    metadata_path = MODEL_DIR / "gnn_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train the GNN first with train_gnn_v2.py")

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

    model.load_state_dict(
        torch.load(model_path, map_location=device))
    model.eval()
    print(f"  Model loaded from {model_path}")
    print(f"  Params: hidden={params['hidden_channels']} "
          f"out={params['out_channels']} "
          f"layers={params['num_layers']}")

    # ------------------------------------------------------------------
    # 4. Compute saliency for each candidate
    # ------------------------------------------------------------------
    print(f"\n  Computing saliency for {len(candidates)} candidates ...")
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
            # Compute saliency
            saliency = compute_saliency(
                model, data, drug_idx, target_idx, device)

            # Get top-K influential nodes
            top_nodes = get_top_nodes(
                saliency, idx_to_node,
                drug_idx, target_idx,
                top_k=top_k_nodes,
                exclude_self=True)

            # Print top nodes
            print(f"    Top influential nodes:")
            for j, n in enumerate(top_nodes[:5]):
                print(f"      {j+1}. [{n['ntype'][:4]}] "
                      f"{n['entity_name'][:60]} "
                      f"  importance={n['importance']:.6f}")

            # Save visualisation
            safe_name = f"{drug_name}_{target_name}"\
                .replace(" ", "_").replace("/", "_")[:60]
            plot_path = out_dir / f"saliency_{safe_name}.png"
            plot_saliency(
                top_nodes, drug_name, target_name,
                rf_score, plot_path)

            # Save top nodes to results
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
            continue

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    out_csv    = out_dir / "saliency_results.csv"
    results_df.to_csv(out_csv, index=False)

    print(f"\n{'='*55}")
    print(f"  Saliency analysis complete")
    print(f"  Candidates explained: "
          f"{results_df['drug_name'].nunique()}")
    print(f"  Results saved to: {out_csv}")
    print(f"  Plots saved to:   {out_dir}/")
    print(f"{'='*55}")

    # Summary: most commonly influential node types
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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