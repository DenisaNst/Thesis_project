"""
train_gnn.py
-------------
Trains and evaluates PDHeteroGNN on the Parkinson's drug-target interaction
dataset, using the same time-split (cutoff 2018) and evaluation protocol
as the Random Forest baselines.

Usage
-----
    python src/models_GNN/train_gnn.py                  # default paths
    python src/models_GNN/train_gnn.py --epochs 300 --hidden 256

Outputs (written to artifacts/gnn/)
------------------------------------
    gnn_model.pt          — trained model weights (torch.save)
    gnn_metadata.json     — node/edge types, index maps, hyperparams
    metrics.json          — ROC-AUC, PR-AUC, F1 on test set
    top_predictions.csv   — ranked drug-target scores on the test set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.transforms import RandomLinkSplit
except ImportError as exc:
    raise ImportError(
        "torch-geometric is required.\n"
        "  pip install torch-geometric"
    ) from exc

from GNN import PDHeteroGNN, build_pd_graph  # noqa: E402

# ---------------------------------------------------------------------------
# Default file paths  (mirror the RF baseline convention)
# ---------------------------------------------------------------------------
INTERACTIONS_CSV   = PROJECT_ROOT / "data" / "raw"       / "chembl_pd_interactions.csv"
DRUG_EMB_CSV       = PROJECT_ROOT / "data" / "processed" / "chembl_drug_embeddings.csv"
TARGET_EMB_CSV     = PROJECT_ROOT / "data" / "processed" / "drkg_target_embeddings.csv"
OUT_DIR            = PROJECT_ROOT / "artifacts" / "gnn"
CUTOFF_YEAR        = 2018


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _normalise_ids(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "molecule_chembl_id" in df.columns and "drug_id" not in df.columns:
        rename["molecule_chembl_id"] = "drug_id"
    if "target_chembl_id" in df.columns and "target_id" not in df.columns:
        rename["target_chembl_id"] = "target_id"
    return df.rename(columns=rename) if rename else df


def load_data(
    interactions_csv: Path = INTERACTIONS_CSV,
    drug_emb_csv: Path     = DRUG_EMB_CSV,
    target_emb_csv: Path   = TARGET_EMB_CSV,
    cutoff_year: int        = CUTOFF_YEAR,
):
    """
    Returns
    -------
    train_edges : np.ndarray shape (N_train, 3)  — [drug_int_idx, target_int_idx, label]
    test_edges  : np.ndarray shape (N_test,  3)
    drug_x      : torch.Tensor  (n_drugs,  drug_emb_dim)
    target_x    : torch.Tensor  (n_targets, target_emb_dim)
    drug_to_idx : dict[str, int]   ChEMBL drug id → row index in drug_x
    target_to_idx: dict[str, int]  ChEMBL target id → row index in target_x
    """
    print("[1/4] Loading interactions …")
    interactions = _normalise_ids(pd.read_csv(interactions_csv))

    # Standardise label column
    if "label" not in interactions.columns:
        if "pchembl_value" in interactions.columns:
            interactions["label"] = (
                pd.to_numeric(interactions["pchembl_value"], errors="coerce") >= 6.0
            ).astype(int)
        else:
            interactions["label"] = 1

    interactions["year"] = pd.to_numeric(interactions.get("year", np.nan), errors="coerce")

    # Deduplicate — keep the row with the highest pChEMBL if present
    if "pchembl_value" in interactions.columns:
        interactions = (
            interactions.sort_values("pchembl_value", ascending=False, na_position="last")
            .drop_duplicates(subset=["drug_id", "target_id"])
        )
    else:
        interactions = interactions.drop_duplicates(subset=["drug_id", "target_id"])

    train_int = interactions[interactions["year"] <= cutoff_year].copy()
    test_int  = interactions[interactions["year"] >  cutoff_year].copy()
    print(f"  Train pairs: {len(train_int):,}  |  Test pairs: {len(test_int):,}")
    print(f"  Train label counts:\n{train_int['label'].value_counts().to_string()}")

    print("[2/4] Loading drug embeddings …")
    drug_emb = _normalise_ids(pd.read_csv(drug_emb_csv))
    drug_emb_cols = sorted(c for c in drug_emb.columns if c.startswith("drug_emb_"))
    if not drug_emb_cols:
        raise ValueError(f"No 'drug_emb_*' columns found in {drug_emb_csv}")

    print("[3/4] Loading target (DRKG TransE) embeddings …")
    target_emb = _normalise_ids(pd.read_csv(target_emb_csv))
    target_emb_cols = sorted(c for c in target_emb.columns if c.startswith("target_emb_"))
    if not target_emb_cols:
        raise ValueError(f"No 'target_emb_*' columns found in {target_emb_csv}")

    print("[4/4] Building integer-indexed node tensors …")

    # Keep only drugs / targets that appear in the embedding files
    all_drug_ids   = sorted(set(drug_emb["drug_id"])   & set(interactions["drug_id"]))
    all_target_ids = sorted(set(target_emb["target_id"]) & set(interactions["target_id"]))

    drug_to_idx   = {d: i for i, d in enumerate(all_drug_ids)}
    target_to_idx = {t: i for i, t in enumerate(all_target_ids)}

    # Build node feature matrices in a consistent order
    drug_emb_ord   = drug_emb.set_index("drug_id").loc[all_drug_ids, drug_emb_cols]
    target_emb_ord = target_emb.set_index("target_id").loc[all_target_ids, target_emb_cols]

    drug_x   = torch.tensor(drug_emb_ord.values,   dtype=torch.float32)
    target_x = torch.tensor(target_emb_ord.values, dtype=torch.float32)

    print(f"  Drug nodes:   {drug_x.shape[0]:,} × {drug_x.shape[1]} dims")
    print(f"  Target nodes: {target_x.shape[0]:,} × {target_x.shape[1]} dims")

    def _make_edges(df: pd.DataFrame):
        """Convert a DataFrame of (drug_id, target_id, label) to integer edges."""
        df = df[
            df["drug_id"].isin(drug_to_idx) & df["target_id"].isin(target_to_idx)
        ].copy()
        df["drug_int"]   = df["drug_id"].map(drug_to_idx)
        df["target_int"] = df["target_id"].map(target_to_idx)
        return df[["drug_int", "target_int", "label"]].to_numpy(dtype=np.int64)

    train_edges = _make_edges(train_int)
    test_edges  = _make_edges(test_int)
    print(f"  Matched train edges: {len(train_edges):,}  |  test edges: {len(test_edges):,}")

    return train_edges, test_edges, drug_x, target_x, drug_to_idx, target_to_idx


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_training_graph(
    train_edges: np.ndarray,
    drug_x: torch.Tensor,
    target_x: torch.Tensor,
    device: torch.device,
) -> HeteroData:
    """
    Builds the HeteroData graph from POSITIVE training edges only.
    The edge_index represents the known drug-target interaction topology
    that the GNN encoder uses for message passing.

    Negative edges are generated on-the-fly during training (see training loop).
    """
    pos_mask    = train_edges[:, 2] == 1
    pos_edges   = train_edges[pos_mask]

    drug_idx_t  = torch.tensor(pos_edges[:, 0], dtype=torch.long)
    target_idx_t = torch.tensor(pos_edges[:, 1], dtype=torch.long)
    edge_index  = torch.stack([drug_idx_t, target_idx_t], dim=0)

    data = build_pd_graph(
        drug_x=drug_x.to(device),
        target_x=target_x.to(device),
        drug_target_edge_index=edge_index.to(device),
        add_reverse_edges=True,
    )
    return data


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def sample_negatives(
    pos_edges: np.ndarray,
    n_drugs: int,
    n_targets: int,
    ratio: float = 1.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Uniform random negative sampling.
    Returns array of shape (n_neg, 2) with [drug_int, target_int].
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pos_set = set(map(tuple, pos_edges[:, :2].tolist()))
    n_neg   = int(len(pos_edges) * ratio)
    negs    = []

    while len(negs) < n_neg:
        d = rng.integers(0, n_drugs)
        t = rng.integers(0, n_targets)
        if (d, t) not in pos_set:
            negs.append([d, t])

    return np.array(negs, dtype=np.int64)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    epochs: int          = 200,
    hidden_channels: int = 256,
    out_channels: int    = 128,
    lr: float            = 1e-3,
    weight_decay: float  = 1e-5,
    neg_ratio: float     = 1.0,
    dropout: float       = 0.2,
    device_str: str      = "cpu",
    interactions_csv: Path = INTERACTIONS_CSV,
    drug_emb_csv: Path     = DRUG_EMB_CSV,
    target_emb_csv: Path   = TARGET_EMB_CSV,
    cutoff_year: int        = CUTOFF_YEAR,
    out_dir: Path           = OUT_DIR,
) -> dict:

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    print(f"\nDevice: {device}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data --------------------------------------------------------
    train_edges, test_edges, drug_x, target_x, drug_to_idx, target_to_idx = load_data(
        interactions_csv, drug_emb_csv, target_emb_csv, cutoff_year
    )

    n_drugs   = drug_x.shape[0]
    n_targets = target_x.shape[0]

    # ---- Build the topology graph (positive train edges only) -------------
    data = build_training_graph(train_edges, drug_x, target_x, device)

    # ---- Initialise model -------------------------------------------------
    # NOTE: to_hetero uses lazy initialisation.  We must do one forward pass
    # before the optimizer can see all parameters.  We use a tiny dummy call.
    model = PDHeteroGNN(
        metadata=data.metadata(),
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    ).to(device)

    # Warm-up pass to initialise lazy layers
    with torch.no_grad():
        dummy_edge = torch.zeros(2, 1, dtype=torch.long, device=device)
        _ = model.encode(data.x_dict, data.edge_index_dict)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    rng       = np.random.default_rng(42)
    pos_train = train_edges[train_edges[:, 2] == 1]
    neg_train = train_edges[train_edges[:, 2] == 0]   # real negatives if any

    # ---- Training ---------------------------------------------------------
    print(f"\nTraining for {epochs} epochs …")
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Encode the full graph
        z_dict = model.encode(data.x_dict, data.edge_index_dict)

        # ---- Positive edges ----
        pos_idx = torch.tensor(pos_train[:, :2].T, dtype=torch.long, device=device)
        pos_scores = model.score_pairs(z_dict, pos_idx)
        pos_labels = torch.ones(pos_scores.shape[0], device=device)

        # ---- Negative edges (sample fresh each epoch) ----
        negs = sample_negatives(pos_train, n_drugs, n_targets, neg_ratio, rng)
        neg_idx = torch.tensor(negs.T, dtype=torch.long, device=device)
        neg_scores = model.score_pairs(z_dict, neg_idx)
        neg_labels = torch.zeros(neg_scores.shape[0], device=device)

        # ---- Loss ----
        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([pos_labels, neg_labels])
        loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs}  loss={loss.item():.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if loss.item() < best_val_loss:
            best_val_loss = loss.item()
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ---- Evaluation -------------------------------------------------------
    print("\nEvaluating on test set …")
    model.eval()
    metrics = evaluate(model, data, test_edges, device)
    print(f"  ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC  : {metrics['pr_auc']:.4f}")
    print(f"  F1      : {metrics['f1']:.4f}")

    # ---- Save artefacts ---------------------------------------------------
    model_path = out_dir / "gnn_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n  [saved] Model → {model_path}")

    # Metadata needed to rebuild the model and map indices back to IDs
    metadata_path = out_dir / "gnn_metadata.json"
    metadata = {
        "hidden_channels": hidden_channels,
        "out_channels":    out_channels,
        "drug_to_idx":     drug_to_idx,
        "target_to_idx":   target_to_idx,
        "cutoff_year":     cutoff_year,
        "node_types":      data.node_types,
        "edge_types":      [list(e) for e in data.edge_types],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"  [saved] Metadata → {metadata_path}")

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  [saved] Metrics  → {metrics_path}")

    # Top predictions on the test set
    _save_top_predictions(model, data, test_edges, drug_to_idx, target_to_idx, device, out_dir)

    return metrics


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(
    model,
    data: HeteroData,
    edges: np.ndarray,
    device: torch.device,
) -> dict:
    """
    Evaluates the model on a pre-built edge array.

    Parameters
    ----------
    edges : np.ndarray, shape (N, 3)  — [drug_int, target_int, label]
    """
    model.eval()
    with torch.no_grad():
        z_dict = model.encode(data.x_dict, data.edge_index_dict)
        edge_idx = torch.tensor(edges[:, :2].T, dtype=torch.long, device=device)
        scores   = model.score_pairs(z_dict, edge_idx).cpu().numpy()

    labels    = edges[:, 2].astype(int)
    probs     = torch.sigmoid(torch.tensor(scores)).numpy()
    preds     = (probs >= 0.5).astype(int)

    if len(np.unique(labels)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "f1": float("nan"),
                "warning": "Only one class present in test labels."}

    return {
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc":  float(average_precision_score(labels, probs)),
        "f1":      float(f1_score(labels, preds, zero_division=0)),
    }


def _save_top_predictions(
    model, data, test_edges, drug_to_idx, target_to_idx, device, out_dir, top_k=200
):
    idx_to_drug   = {v: k for k, v in drug_to_idx.items()}
    idx_to_target = {v: k for k, v in target_to_idx.items()}

    model.eval()
    with torch.no_grad():
        z_dict   = model.encode(data.x_dict, data.edge_index_dict)
        edge_idx = torch.tensor(test_edges[:, :2].T, dtype=torch.long, device=device)
        scores   = model.score_pairs(z_dict, edge_idx).cpu().numpy()

    probs = torch.sigmoid(torch.tensor(scores)).numpy()
    order = np.argsort(probs)[::-1][:top_k]

    rows = []
    for i in order:
        d_idx = int(test_edges[i, 0])
        t_idx = int(test_edges[i, 1])
        rows.append({
            "drug_id":   idx_to_drug.get(d_idx, f"drug_{d_idx}"),
            "target_id": idx_to_target.get(t_idx, f"target_{t_idx}"),
            "score":     float(probs[i]),
            "label":     int(test_edges[i, 2]),
        })

    out_csv = out_dir / "top_predictions.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  [saved] Top predictions → {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train PDHeteroGNN for PD drug-target prediction.")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--hidden",       type=int,   default=256,  dest="hidden_channels")
    p.add_argument("--out_channels", type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--neg_ratio",    type=float, default=1.0,
                   help="Ratio of negative to positive edges per epoch.")
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--device",       type=str,   default="cpu",  dest="device_str")
    p.add_argument("--cutoff_year",  type=int,   default=2018)
    p.add_argument("--interactions", type=Path,  default=INTERACTIONS_CSV,
                   dest="interactions_csv")
    p.add_argument("--drug_emb",     type=Path,  default=DRUG_EMB_CSV,
                   dest="drug_emb_csv")
    p.add_argument("--target_emb",   type=Path,  default=TARGET_EMB_CSV,
                   dest="target_emb_csv")
    p.add_argument("--out_dir",      type=Path,  default=OUT_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    train(**vars(args))