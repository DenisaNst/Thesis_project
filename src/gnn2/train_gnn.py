"""
train_gnn_pd.py  —  Trains PDHeteroGNN on PD-SPECIFIC DRKG subgraph.

Approach:
  - PD-specific subgraph (2-hop neighbourhood of Parkinson's node)
  - ChEMBL MT embeddings for Compound nodes (768d)
  - DRKG TransE embeddings for all other node types (400d)
  - Per-type input projection layers (maps each type to hidden_channels)
  - Dot product predictor
  - BCE loss
  - Real ChEMBL inactives as negatives (pChEMBL < 6),
    supplemented with random pairs if not enough
  - Time-split evaluation (pre/post 2018)
  - Random hyperparameter search + early stopping
  - Hits@5 and Hits@10

Imports from:
  - build_drkg_pd.py   (PD subgraph — returns 6 values including node_feature_dims)
  - GNN_pd.py          (PDHeteroGNN WITH input projections)

Results saved to:
  - artifacts/gnn_pd/

Run:
    python src/models_GNN/train_gnn_pd.py --n_trials 3 --max_epochs 50
    python src/models_GNN/train_gnn_pd.py
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from torch_geometric.data import HeteroData
from src.gnn_final.GNN_pd import PDHeteroGNN                                      # with input projections
from build_drkg_pd import build_pd_drkg_graph, PRED_SRC, PRED_DST  # PD subgraph, 6 return values

OUT_DIR = PROJECT_ROOT / "artifacts" / "gnn_pd"

SEARCH_SPACE = {
    "hidden_channels": [64, 128, 256],
    "out_channels":    [32, 64],
    "lr":              [1e-3, 5e-4, 3e-4],
    "dropout":         [0.1, 0.2, 0.3],
    "weight_decay":    [1e-4, 1e-5],
    "neg_k":           [3, 5],
    "num_layers":      [1, 2],
}


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def sample_negatives(pos_edges: np.ndarray, n_src: int, n_dst: int,
                     k: int, rng: np.random.Generator) -> np.ndarray:
    """k random negatives per positive by corrupting the tail node."""
    pos_set = set(map(tuple, pos_edges[:, :2].tolist()))
    negs = []
    for src, _ in pos_edges[:, :2]:
        count = 0
        while count < k:
            dst = int(rng.integers(0, n_dst))
            if (int(src), dst) not in pos_set:
                negs.append([int(src), dst])
                count += 1
    return np.array(negs, dtype=np.int64)


# ---------------------------------------------------------------------------
# Hits@K
# ---------------------------------------------------------------------------

def hits_at_k(model: PDHeteroGNN, data: HeteroData,
              pos_edges: np.ndarray, device: torch.device,
              k: int = 10, n_neg: int = 99) -> float:
    """
    For each positive pair, rank it against n_neg random negatives.
    Returns fraction that rank in the top-k.
    """
    model.eval()
    rng   = np.random.default_rng(0)
    n_dst = data[PRED_DST].x.shape[0]
    hits  = 0

    with torch.no_grad():
        z = model.encode(data.x_dict, data.edge_index_dict)

    for src, dst, _ in pos_edges:
        neg_dst = rng.integers(0, n_dst, size=n_neg * 2)
        neg_dst = neg_dst[neg_dst != dst][:n_neg]
        if len(neg_dst) < n_neg:
            continue

        candidates = np.concatenate([[dst], neg_dst])
        cand_idx   = torch.tensor(
            np.stack([[src] * len(candidates), candidates]),
            dtype=torch.long, device=device)

        with torch.no_grad():
            scores = model.score_pairs(
                z, cand_idx, PRED_SRC, PRED_DST).cpu().numpy()

        rank = int(np.sum(scores > scores[0])) + 1
        if rank <= k:
            hits += 1

    return hits / len(pos_edges) if len(pos_edges) > 0 else 0.0


# ---------------------------------------------------------------------------
# Single training trial with early stopping
# ---------------------------------------------------------------------------

def run_trial(params: dict, data: HeteroData,
              train_edges: np.ndarray, val_edges: np.ndarray,
              node_feature_dims: dict,
              device: torch.device, max_epochs: int = 300,
              patience: int = 50,
              verbose: bool = False) -> tuple[float, PDHeteroGNN]:
    """
    Train one hyperparameter configuration.
    Returns (best_val_auc, model_with_best_weights).
    """
    rng  = np.random.default_rng(42)
    k    = params["neg_k"]
    data = data.to(device)

    # GNN_pd.PDHeteroGNN — requires node_feature_dims for input projections
    model = PDHeteroGNN(
        metadata=data.metadata(),
        hidden_channels=params["hidden_channels"],
        out_channels=params["out_channels"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        node_feature_dims=node_feature_dims,
    ).to(device)

    with torch.no_grad():
        model.encode(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"])

    n_src = data[PRED_SRC].x.shape[0]
    n_dst = data[PRED_DST].x.shape[0]

    train_pos = train_edges[train_edges[:, 2] == 1]
    train_neg = train_edges[train_edges[:, 2] == 0]
    val_pos   = val_edges[val_edges[:, 2] == 1]
    val_neg   = val_edges[val_edges[:, 2] == 0]

    best_val_loss = float("inf")
    best_val_auc  = 0.0
    best_state    = None
    no_improve    = 0

    for epoch in range(1, max_epochs + 1):

        # -----------------------------------------------------------
        # Training step
        # -----------------------------------------------------------
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x_dict, data.edge_index_dict)

        pos_idx = torch.tensor(
            train_pos[:, :2].T, dtype=torch.long, device=device)
        pos_sc  = model.score_pairs(z, pos_idx, PRED_SRC, PRED_DST)

        n_needed = len(train_pos) * k
        if len(train_neg) >= n_needed:
            chosen  = train_neg[
                rng.choice(len(train_neg), n_needed, replace=False)]
            neg_arr = chosen[:, :2]
        else:
            rand    = sample_negatives(
                train_pos[:, :2], n_src, n_dst, k, rng)
            neg_arr = np.concatenate([train_neg[:, :2], rand])

        neg_idx = torch.tensor(neg_arr.T, dtype=torch.long, device=device)
        neg_sc  = model.score_pairs(z, neg_idx, PRED_SRC, PRED_DST)

        train_scores = torch.cat([pos_sc, neg_sc])
        train_labels = torch.cat([
            torch.ones(len(pos_sc),  device=device),
            torch.zeros(len(neg_sc), device=device),
        ])
        loss = F.binary_cross_entropy_with_logits(train_scores, train_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # -----------------------------------------------------------
        # Validation step
        # -----------------------------------------------------------
        model.eval()
        with torch.no_grad():
            z_v = model.encode(data.x_dict, data.edge_index_dict)

            vi_pos = torch.tensor(
                val_pos[:, :2].T, dtype=torch.long, device=device)
            vi_neg = torch.tensor(
                val_neg[:, :2].T, dtype=torch.long, device=device)

            v_sc_pos = model.score_pairs(
                z_v, vi_pos, PRED_SRC, PRED_DST).cpu()
            v_sc_neg = model.score_pairs(
                z_v, vi_neg, PRED_SRC, PRED_DST).cpu()

        # Validation loss — uses validation scores only
        v_loss = float(F.binary_cross_entropy_with_logits(
            torch.cat([v_sc_pos, v_sc_neg]),
            torch.cat([torch.ones(len(v_sc_pos)),
                       torch.zeros(len(v_sc_neg))])
        ).item())

        all_v_sc   = np.concatenate([v_sc_pos.numpy(), v_sc_neg.numpy()])
        all_v_lab  = np.concatenate([np.ones(len(v_sc_pos)),
                                     np.zeros(len(v_sc_neg))])
        all_v_prob = torch.sigmoid(torch.tensor(all_v_sc)).numpy()

        v_auc = (float(roc_auc_score(all_v_lab, all_v_prob))
                 if len(np.unique(all_v_lab)) == 2 else 0.0)

        if verbose and epoch % 10 == 0:
            print(f"      epoch {epoch:3d} | "
                  f"loss={loss.item():.4f} | "
                  f"v_loss={v_loss:.4f} | "
                  f"v_auc={v_auc:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_val_auc  = v_auc
            best_state    = deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return best_val_auc, model


# ---------------------------------------------------------------------------
# Random hyperparameter search
# ---------------------------------------------------------------------------

def random_search(n_trials: int, data: HeteroData,
                  train_edges: np.ndarray, val_edges: np.ndarray,
                  node_feature_dims: dict,
                  device: torch.device, max_epochs: int,
                  patience: int) -> tuple[dict, pd.DataFrame]:

    rng = random.Random(42)
    best_params, best_auc, results = None, -1.0, []

    print(f"\n{'='*55}\n  Random search  ({n_trials} trials)\n{'='*55}")

    for trial in range(1, n_trials + 1):
        params = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        if params["out_channels"] > params["hidden_channels"]:
            params["out_channels"] = params["hidden_channels"]

        print(f"\n  Trial {trial:2d}/{n_trials}  "
              f"hidden={params['hidden_channels']} "
              f"out={params['out_channels']} "
              f"layers={params['num_layers']} "
              f"lr={params['lr']:.0e} "
              f"dropout={params['dropout']} "
              f"wd={params['weight_decay']:.0e} "
              f"neg_k={params['neg_k']}")

        val_auc, _ = run_trial(
            params, data, train_edges, val_edges,
            node_feature_dims, device,
            max_epochs, patience,
            verbose=False)

        print(f"    -> val AUC: {val_auc:.4f}")
        results.append({**params, "val_roc_auc": val_auc})

        if val_auc > best_auc:
            best_auc, best_params = val_auc, deepcopy(params)

    print(f"\n{'='*55}\n  Best val AUC={best_auc:.4f}")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    return best_params, pd.DataFrame(results).sort_values(
        "val_roc_auc", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation on test set
# ---------------------------------------------------------------------------

def evaluate(model: PDHeteroGNN, data: HeteroData,
             edges: np.ndarray, device: torch.device) -> dict:
    model.eval()

    pos_edges = edges[edges[:, 2] == 1]
    neg_edges = edges[edges[:, 2] == 0]

    with torch.no_grad():
        z = model.encode(data.x_dict, data.edge_index_dict)

        pos_sc = model.score_pairs(
            z,
            torch.tensor(pos_edges[:, :2].T, dtype=torch.long, device=device),
            PRED_SRC, PRED_DST).cpu().numpy()

        neg_sc = model.score_pairs(
            z,
            torch.tensor(neg_edges[:, :2].T, dtype=torch.long, device=device),
            PRED_SRC, PRED_DST).cpu().numpy()

    all_scores = np.concatenate([pos_sc, neg_sc])
    all_labels = np.concatenate([np.ones(len(pos_sc)),
                                 np.zeros(len(neg_sc))])
    probs = torch.sigmoid(torch.tensor(all_scores)).numpy()
    preds = (probs >= 0.5).astype(int)

    if len(np.unique(all_labels)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                "f1": float("nan"), "hits5": float("nan"),
                "hits10": float("nan")}

    return {
        "roc_auc": float(roc_auc_score(all_labels, probs)),
        "pr_auc":  float(average_precision_score(all_labels, probs)),
        "f1":      float(f1_score(all_labels, preds, zero_division=0)),
        "hits5":   hits_at_k(model, data, pos_edges, device, k=5),
        "hits10":  hits_at_k(model, data, pos_edges, device, k=10),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_trials: int = 15, max_epochs: int = 300, patience: int = 50,
         val_fraction: float = 0.1, device_str: str = "cpu",
         out_dir: Path = OUT_DIR):

    device = torch.device(
        device_str if (device_str == "cpu" or
                       torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    # build_drkg_pd.py returns exactly 6 values (includes node_feature_dims)
    (data, node_to_idx, idx_to_node,
     train_edges, test_edges,
     node_feature_dims) = build_pd_drkg_graph()

    if len(train_edges) == 0:
        raise RuntimeError("No train edges found.")

    print(f"\n  Node feature dims: {node_feature_dims}")
    print(f"\n  Train edges: {len(train_edges)} "
          f"(active={(train_edges[:,2]==1).sum()} "
          f"inactive={(train_edges[:,2]==0).sum()})")
    print(f"  Test edges:  {len(test_edges)} "
          f"(active={(test_edges[:,2]==1).sum()} "
          f"inactive={(test_edges[:,2]==0).sum()})")

    rng   = np.random.default_rng(42)
    idx   = rng.permutation(len(train_edges))
    n_val = max(1, int(val_fraction * len(train_edges)))
    s_val   = train_edges[idx[:n_val]]
    s_train = train_edges[idx[n_val:]]
    print(f"\n  Search split -> train: {len(s_train):,}  val: {len(s_val):,}")

    best_params, results_df = random_search(
        n_trials, data, s_train, s_val,
        node_feature_dims, device,
        max_epochs, patience)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "search_results.csv", index=False)

    print(f"\n{'='*55}\n  Final training (verbose)\n{'='*55}")
    idx2    = rng.permutation(len(train_edges))
    f_val   = train_edges[idx2[:n_val]]
    f_train = train_edges[idx2[n_val:]]

    _, model = run_trial(
        best_params, data, f_train, f_val,
        node_feature_dims, device,
        max_epochs, patience,
        verbose=True)

    metrics = evaluate(model, data, test_edges, device)

    print(f"\n{'='*55}")
    print(f"  Test ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"  Test PR-AUC  : {metrics['pr_auc']:.4f}")
    print(f"  Test F1      : {metrics['f1']:.4f}")
    print(f"  Hits@5       : {metrics['hits5']:.4f}")
    print(f"  Hits@10      : {metrics['hits10']:.4f}")
    print(f"  (RF time-slice baseline: AUC ~0.76)")
    print(f"{'='*55}")

    torch.save(model.state_dict(), out_dir / "gnn_model.pt")
    (out_dir / "gnn_metadata.json").write_text(json.dumps({
        "best_params":       best_params,
        "pred_src":          PRED_SRC,
        "pred_dst":          PRED_DST,
        "node_types":        list(data.node_types),
        "node_feature_dims": node_feature_dims,
        "approach":          "PD subgraph + ChEMBL-MT embeddings + input projections + BCE loss",
    }, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n  [saved] -> {out_dir}")

    return metrics, model, data, node_to_idx, idx_to_node


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_trials",   type=int,  default=15)
    p.add_argument("--max_epochs", type=int,  default=300)
    p.add_argument("--patience",   type=int,  default=50)
    p.add_argument("--device",     type=str,  default="cpu",
                   dest="device_str")
    p.add_argument("--out_dir",    type=Path, default=OUT_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(n_trials=args.n_trials,
         max_epochs=args.max_epochs,
         patience=args.patience,
         device_str=args.device_str,
         out_dir=args.out_dir)