"""
How this script works :
1. Graph Construction: Loads the DRKG network, where nodes are pre-initialized
   with 400-dimensional TransE embeddings, and high-confidence saliency candidates
   have been explicitly removed to prevent data leakage.
2. Edge Routing: Uses all remaining physical Compound-binds-Gene (CbG) edges
   in DRKG as positive training examples (Label = 1).
3. Negative Sampling: Dynamically samples disconnected node pairs during
   training to act as negative examples (Label = 0).
4. Model Training: Trains the PDHeteroGNN using Binary Cross-Entropy loss
   with PyTorch Geometric lazy initialization.
5. Evaluation: Computes final testing metrics (ROC-AUC, PR-AUC, Hits@k)
   and saves the model weights (gnn_model.pt) for downstream interpretability analysis.
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
from gnn_final.GNN_pd import PDHeteroGNN
from build_drkg import build_pd_drkg_graph, PRED_SRC, PRED_DST

OUT_DIR = PROJECT_ROOT / "artifacts" / "gnn_3"

BEST_PARAMS = {
    "hidden_channels": 128,
    "out_channels":    64,
    "lr":              1e-3,
    "dropout":         0.2,
    "weight_decay":    1e-5,
    "neg_k":           5,
    "num_layers":      1,
}


def sample_negatives(pos_edges: np.ndarray, n_src: int, n_dst: int,
                     k: int, rng: np.random.Generator) -> np.ndarray:
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


def hits_at_k(model: PDHeteroGNN, data: HeteroData,
              pos_edges: np.ndarray, device: torch.device,
              k: int = 5, n_neg: int = 99) -> float:
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


def run_trial(params: dict, data: HeteroData,
              train_edges: np.ndarray, val_edges: np.ndarray,
              device: torch.device, max_epochs: int = 150,
              patience: int = 15,
              verbose: bool = False) -> tuple[float, float, PDHeteroGNN]:
    rng  = np.random.default_rng(42)
    k    = params["neg_k"]
    data = data.to(device)

    model = PDHeteroGNN(
        metadata=data.metadata(),
        hidden_channels=params["hidden_channels"],
        out_channels=params["out_channels"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    with torch.no_grad():
        model.encode(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"])

    n_src = data[PRED_SRC].x.shape[0]
    n_dst = data[PRED_DST].x.shape[0]

    train_pos = train_edges
    val_pos   = val_edges

    best_val_loss  = float("inf")
    best_val_auc   = 0.0
    best_train_auc = 0.0
    best_state     = None
    no_improve     = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x_dict, data.edge_index_dict)

        pos_idx = torch.tensor(
            train_pos[:, :2].T, dtype=torch.long, device=device)
        pos_sc  = model.score_pairs(z, pos_idx, PRED_SRC, PRED_DST)

        neg_arr = sample_negatives(
            train_pos[:, :2], n_src, n_dst, k, rng)
        neg_idx = torch.tensor(
            neg_arr.T, dtype=torch.long, device=device)
        neg_sc  = model.score_pairs(z, neg_idx, PRED_SRC, PRED_DST)

        train_scores = torch.cat([pos_sc, neg_sc])
        train_labels = torch.cat([
            torch.ones(len(pos_sc),  device=device),
            torch.zeros(len(neg_sc), device=device),
        ])
        loss = F.binary_cross_entropy_with_logits(
            train_scores, train_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z_v = model.encode(data.x_dict, data.edge_index_dict)

            vi_pos = torch.tensor(
                val_pos[:, :2].T, dtype=torch.long, device=device)
            v_sc_pos = model.score_pairs(
                z_v, vi_pos, PRED_SRC, PRED_DST).cpu()

            val_neg_arr = sample_negatives(
                val_pos[:, :2], n_src, n_dst, 1, rng)
            vi_neg = torch.tensor(
                val_neg_arr.T, dtype=torch.long, device=device)
            v_sc_neg = model.score_pairs(
                z_v, vi_neg, PRED_SRC, PRED_DST).cpu()

        v_loss = float(F.binary_cross_entropy_with_logits(
            torch.cat([v_sc_pos, v_sc_neg]),
            torch.cat([torch.ones(len(v_sc_pos)),
                       torch.zeros(len(v_sc_neg))])
        ).item())

        all_v_sc   = np.concatenate([
            v_sc_pos.numpy(), v_sc_neg.numpy()])
        all_v_lab  = np.concatenate([
            np.ones(len(v_sc_pos)), np.zeros(len(v_sc_neg))])
        all_v_prob = torch.sigmoid(
            torch.tensor(all_v_sc)).numpy()

        v_auc = (float(roc_auc_score(all_v_lab, all_v_prob))
                 if len(np.unique(all_v_lab)) == 2 else 0.0)

        with torch.no_grad():
            tr_pos_sc = model.score_pairs(
                z_v, pos_idx, PRED_SRC, PRED_DST).cpu().numpy()
            tr_neg_arr = sample_negatives(
                train_pos[:, :2], n_src, n_dst, 1, rng)
            tr_neg_idx = torch.tensor(
                tr_neg_arr.T, dtype=torch.long, device=device)
            tr_neg_sc = model.score_pairs(
                z_v, tr_neg_idx, PRED_SRC, PRED_DST).cpu().numpy()

        tr_sc    = np.concatenate([tr_pos_sc, tr_neg_sc])
        tr_lab   = np.concatenate([
            np.ones(len(tr_pos_sc)), np.zeros(len(tr_neg_sc))])
        tr_prob  = torch.sigmoid(torch.tensor(tr_sc)).numpy()
        train_auc = float(roc_auc_score(tr_lab, tr_prob)) \
            if len(np.unique(tr_lab)) == 2 else 0.0

        if v_loss < best_val_loss:
            best_val_loss  = v_loss
            best_val_auc   = v_auc
            best_train_auc = train_auc
            best_state     = deepcopy(model.state_dict())
            no_improve     = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return best_val_auc, best_train_auc, model


def evaluate(model: PDHeteroGNN, data: HeteroData,
             pos_edges: np.ndarray, device: torch.device) -> dict:
    model.eval()

    n_src = data[PRED_SRC].x.shape[0]
    n_dst = data[PRED_DST].x.shape[0]
    rng = np.random.default_rng(42)

    neg_edges_arr = sample_negatives(pos_edges, n_src, n_dst, 1, rng)

    with torch.no_grad():
        z = model.encode(data.x_dict, data.edge_index_dict)
        pos_sc = model.score_pairs(
            z,
            torch.tensor(pos_edges[:, :2].T,
                         dtype=torch.long, device=device),
            PRED_SRC, PRED_DST).cpu().numpy()
        neg_sc = model.score_pairs(
            z,
            torch.tensor(neg_edges_arr.T,
                         dtype=torch.long, device=device),
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
        "pr_auc": float(average_precision_score(all_labels, probs)),
        "f1": float(f1_score(all_labels, preds, zero_division=0)),
        "hits5": hits_at_k(model, data, pos_edges, device, k=5),
        "hits10": hits_at_k(model, data, pos_edges, device, k=10),
    }


def main(n_trials: int = 1, max_epochs: int = 150,
         patience: int = 15, val_fraction: float = 0.1,
         device_str: str = "cpu", out_dir: Path = OUT_DIR):

    device = torch.device(
        device_str if (device_str == "cpu" or
                       torch.cuda.is_available()) else "cpu")

    data, node_to_idx, idx_to_node, train_edges, test_edges = \
        build_pd_drkg_graph()

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_edges))

    n_val = int(0.1 * len(train_edges))
    n_test = int(0.1 * len(train_edges))

    f_val = train_edges[idx[:n_val]]
    f_test = train_edges[idx[n_val:n_val + n_test]]
    f_train = train_edges[idx[n_val + n_test:]]

    print(f"  DRKG split: train={len(f_train):,}  val={len(f_val):,}  test={len(f_test):,}")
    best_val_auc, train_auc, model = run_trial(
        BEST_PARAMS, data, f_train, f_val,
        device, max_epochs, patience,
        verbose=True)

    metrics = evaluate(model, data, f_test, device)
    metrics["train_roc_auc"] = train_auc
    metrics["val_roc_auc"]   = best_val_auc
    metrics["overfit_gap"]   = train_auc - metrics["roc_auc"]

    print(f"  Train ROC-AUC : {train_auc:.4f}")
    print(f"  Val ROC-AUC   : {best_val_auc:.4f}")
    print(f"  Test ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  Overfit gap   : {metrics['overfit_gap']:.4f}")
    print(f"  Test PR-AUC   : {metrics['pr_auc']:.4f}")
    print(f"  Test F1       : {metrics['f1']:.4f}")
    print(f"  Hits@5        : {metrics['hits5']:.4f}")
    print(f"  Hits@10       : {metrics['hits10']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "gnn_model.pt")
    (out_dir / "gnn_metadata.json").write_text(json.dumps({
        "best_params": BEST_PARAMS,
        "pred_src":    PRED_SRC,
        "pred_dst":    PRED_DST,
        "node_types":  list(data.node_types),
        "approach":    "train on DRKG CbG (80%), val on DRKG (10%), test on saliency_candidates_all",
    }, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n  [saved] -> {out_dir}")

    return metrics, model, data, node_to_idx, idx_to_node


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_trials",   type=int,  default=10)
    p.add_argument("--max_epochs", type=int,  default=150)
    p.add_argument("--patience",   type=int,  default=15)
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