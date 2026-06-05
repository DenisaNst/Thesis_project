"""
Comprehensive evaluation of Random Forest model predictions on labeled drug-target
interaction data. Computes standard classification metrics, generates visualizations,
and performs top-k enrichment analysis to assess model quality and ranking ability.

Key functionality:
  - Load trained RF model and merge interaction labels with drug and protein embeddings
  - Generate probabilistic predictions for drug-target pairs
  - Compute classification metrics: accuracy, F1, ROC-AUC, PR-AUC, Brier, Log Loss
  - Visualize score distributions: known positives vs negatives (histogram)
  - Plot ROC and Precision-Recall curves with AUC scores
  - Compute top-k enrichment: precision, recall, enrichment factor at k={10,25,50,...,1000}
  - Normalize column names across different data sources (ChEMBL vs DrugBank IDs)

Output:
  - CSV: evaluation_scores_esm2_time.csv with per-pair predictions and probabilities
  - CSV: topk_enrichment_similarity_esm2.csv with precision/recall/enrichment at each k
  - PNG: score_histograms_esm2_time.png showing separation of positives vs negatives
  - PNG: roc_pr_curves_similarity_esm2.png showing ROC and Precision-Recall performance
  - PNG: calibration_curve_similarity_esm2.png showing model reliability
  - JSON: metrics.json with summary statistics (AUC, F1, positive rate, etc.)
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}

    if "molecule_chembl_id" in out.columns and "drug_id" not in out.columns:
        rename_map["molecule_chembl_id"] = "drug_id"

    if "target_chembl_id" in out.columns and "target_id" not in out.columns:
        rename_map["target_chembl_id"] = "target_id"

    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def prepare_matrix(interactions_df, drug_embeddings_df, protein_embeddings_df):
    interactions_df = normalize_id_columns(interactions_df)
    drug_embeddings_df = normalize_id_columns(drug_embeddings_df)
    protein_embeddings_df = normalize_id_columns(protein_embeddings_df)

    drug_emb_cols = [c for c in drug_embeddings_df.columns if c.startswith("drug_emb_")]
    prot_emb_cols = [c for c in protein_embeddings_df.columns if c.startswith("target_emb_")]

    overlap_cols = [c for c in interactions_df.columns if c.startswith("drug_emb_") or c.startswith("target_emb_")]
    if overlap_cols:
        interactions_df = interactions_df.drop(columns=overlap_cols)

    d_clean = drug_embeddings_df[["drug_id"] + drug_emb_cols]
    p_clean = protein_embeddings_df[["target_id"] + prot_emb_cols]

    merged = interactions_df.merge(d_clean, on="drug_id", how="inner")
    merged = merged.merge(p_clean, on="target_id", how="inner")

    feature_cols = drug_emb_cols + prot_emb_cols

    X = merged[feature_cols].to_numpy(dtype=np.float32)
    y = merged["label"].to_numpy(dtype=int)

    return merged, X, y, feature_cols


def get_positive_class_probabilities(clf, X: np.ndarray) -> np.ndarray:
    classes = list(clf.classes_)
    if 1 in classes:
        pos_idx = classes.index(1)
    return clf.predict_proba(X)[:, pos_idx]


def compute_topk_enrichment(
        df: pd.DataFrame,
        score_col: str = "probability",
        label_col: str = "label",
        ks: list[int] | None = None,
) -> pd.DataFrame:
    if ks is None:
        ks = [5, 10, 15, 25, 50, 100, 250, 500, 1000]

    total_n = len(df)
    total_pos = int(df[label_col].sum())
    base_rate = total_pos / total_n if total_n else 0.0

    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    rows = []
    for k in ks:
        k = min(k, total_n)
        topk = ranked.head(k)
        hits = int(topk[label_col].sum())
        precision_at_k = hits / k if k else 0.0
        recall_at_k = hits / total_pos if total_pos else 0.0
        enrichment_factor = (precision_at_k / base_rate) if base_rate > 0 else np.nan
        rows.append({
            "k": k,
            "hits": hits,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "base_rate": base_rate,
            "enrichment_factor": enrichment_factor,
        })
    return pd.DataFrame(rows)


def plot_score_histograms(df: pd.DataFrame, out_path: Path) -> None:
    pos = df[df["label"] == 1]["probability"].astype(float)
    neg = df[df["label"] == 0]["probability"].astype(float)

    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 30)
    plt.hist(neg, bins=bins, alpha=0.65, label="Negative / non-interaction", color="steelblue")
    plt.hist(pos, bins=bins, alpha=0.65, label="Positive / interaction", color="darkorange")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Score Distribution: Known Positives vs Negatives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc_pr_curves(df: pd.DataFrame, out_path: Path) -> dict:
    y_true = df["label"].to_numpy(dtype=int)
    y_score = df["probability"].to_numpy(dtype=float)

    roc_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="navy", lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, color="darkred", lw=2, label=f"PR AUC = {pr_auc:.4f}")
    base_rate = y_true.mean() if len(y_true) else 0.0
    axes[1].axhline(base_rate, color="gray", linestyle="--", lw=1, label=f"Baseline = {base_rate:.4f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)}


def plot_calibration_curve_chart(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path):
    """Plot the calibration curve (reliability diagram) for the model."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "s-", label="Random Forest", color="#0072B2", linewidth=2, markersize=8)

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("True Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Evaluate RF predictions on labeled interaction data."
    )
    parser.add_argument(
        "--interactions_csv",
        type=Path,
        default=project_root / "data" / "raw" / "chembl_pd_interactions.csv",
    )
    parser.add_argument(
        "--drug_embeddings_csv",
        type=Path,
        default=project_root / "data" / "processed" / "chembl_drug_embeddings.csv",
    )
    parser.add_argument(
        "--drkg_target_embeddings_csv",
        type=Path,
        default=project_root / "data" / "processed" / "drkg_target_embeddings.csv",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=project_root / "artifacts" / "rf_timeslice_esm2" / "rf_model.pkl",
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        default=project_root / "artifacts" / "rf_timeslice_esm2" / "rf_metadata.json",
        help="Path to metadata.json saved during training (contains feature_cols).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=project_root / "artifacts" / "rf_evaluation",
    )
    parser.add_argument(
        "--label_threshold",
        type=float,
        default=6.0,
        help="pChEMBL threshold for positive label if label column is absent.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and metadata...")
    clf = load_model(args.model_path)

    if args.metadata_path.exists():
        metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
        print(f"  Metadata loaded from: {args.metadata_path}")

    interactions = pd.read_csv(args.interactions_csv)
    drug_emb = pd.read_csv(args.drug_embeddings_csv)
    protein_emb = pd.read_csv(args.drkg_target_embeddings_csv)

    merged, X, y, feature_cols_from_data = prepare_matrix(interactions, drug_emb, protein_emb)

    probs = get_positive_class_probabilities(clf, X)
    preds = (probs >= 0.5).astype(int)

    eval_df = merged[["drug_id", "target_id", "label"]].copy()
    if "drug_name" in merged.columns:
        eval_df["drug_name"] = merged["drug_name"]
    if "target_name" in merged.columns:
        eval_df["target_name"] = merged["target_name"]
    eval_df["probability"] = probs
    eval_df["predicted_label"] = preds

    eval_path = args.out_dir / "evaluation_scores_esm2_time.csv"
    eval_df.to_csv(eval_path, index=False)

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, probs)) if len(np.unique(y)) > 1 else float("nan"),
        "brier_score": float(brier_score_loss(y, probs)),
        "log_loss": float(log_loss(y, probs)),
        "positive_rate": float(y.mean()),
        "n_rows": int(len(y)),
    }

    hist_path = args.out_dir / "score_histograms_esm2_time.png"
    rocpr_path = args.out_dir / "roc_pr_curves_similarity_esm2.png"
    plot_score_histograms(eval_df, hist_path)
    curve_metrics = plot_roc_pr_curves(eval_df, rocpr_path)
    metrics.update(curve_metrics)

    calib_path = args.out_dir / "calibration_curve_similarity_esm2.png"
    plot_calibration_curve_chart(y, probs, calib_path)

    topk_df = compute_topk_enrichment(eval_df, ks=[5, 10, 15, 25, 50, 100, 250, 500, 1000])
    topk_path = args.out_dir / "topk_enrichment_similarity_esm2.csv"
    topk_df.to_csv(topk_path, index=False)

    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()