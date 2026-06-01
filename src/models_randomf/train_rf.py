"""
How this script works:
1. Data Loading: Imports drug-target interactions, drug embeddings (ChemBERTa),
   and target embeddings (ESM2 or DRKG), merging them on their respective IDs.
2. Naive Random Split: Uses a standard 80/20 random split (`train_test_split`).
   Unlike the time-slice scripts, this randomly scrambles all data regardless of
   publication year, mixing past and future discoveries together.
3. Model Training: Trains a Random Forest Classifier using fixed, pre-selected
   hyperparameters (n_estimators=300, max_depth=15, etc.) rather than searching
   for the best ones.
4. Evaluation & Export: Calculates standard metrics (ROC-AUC, Precision-Recall AUC)
   for both the training and testing sets to highlight the overfitting gap. It then
   saves the model, metrics, and metadata to disk.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

try:
    from src.evaluation import evaluation_protocol as eval_protocol
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluation import evaluation_protocol as eval_protocol  # type: ignore


def _normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ChEMBL-style ID columns to the standard names used throughout
    the pipeline (drug_id, target_id)."""
    out = df.copy()
    rename_map = {}
    if "molecule_chembl_id" in out.columns and "drug_id" not in out.columns:
        rename_map["molecule_chembl_id"] = "drug_id"
    if "target_chembl_id" in out.columns and "target_id" not in out.columns:
        rename_map["target_chembl_id"] = "target_id"
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _prepare_matrix(
    interactions_df: pd.DataFrame,
    drug_embeddings_df: pd.DataFrame,
    protein_embeddings_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:

    # Normalise column names in all three dataframes.
    interactions_df       = _normalize_id_columns(interactions_df)
    drug_embeddings_df    = _normalize_id_columns(drug_embeddings_df)
    protein_embeddings_df = _normalize_id_columns(protein_embeddings_df)

    merged = interactions_df.merge(drug_embeddings_df, on="drug_id", how="inner")
    merged = merged.merge(protein_embeddings_df, on="target_id", how="inner")

    feature_cols = [
        c for c in merged.columns
        if c.startswith("drug_emb_") or c.startswith("target_emb_")
    ]

    X = merged[feature_cols].to_numpy(dtype=np.float32)
    y = merged["label"].to_numpy(dtype=int)
    return merged, X, y, feature_cols


def train_and_evaluate(
    interactions_csv: Path,
    drug_embeddings_csv: Path,
    protein_embeddings_csv: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, dict, list[str], pd.DataFrame, pd.DataFrame]:
    interactions = eval_protocol.load_and_standardize_interactions(str(interactions_csv))
    print(f"  Unique drug-target pairs after deduplication: {len(interactions)}")
    print(f"  Label counts:\n{interactions['label'].value_counts().to_string()}")

    drug_emb    = _normalize_id_columns(pd.read_csv(drug_embeddings_csv))
    protein_emb = _normalize_id_columns(pd.read_csv(protein_embeddings_csv))

    print(f"  Drug embeddings:    {len(drug_emb)} rows  "
          f"| id col: {'drug_id' if 'drug_id' in drug_emb.columns else '*** MISSING drug_id ***'}")
    print(f"  Protein embeddings: {len(protein_emb)} rows  "
          f"| id col: {'target_id' if 'target_id' in protein_emb.columns else '*** MISSING target_id ***'}")

    merged, X, y, feature_cols = _prepare_matrix(interactions, drug_emb, protein_emb)

    print(f"  Rows after merge:   {len(merged)}")
    print(f"  Feature dimensions: {X.shape[1]}  "
          f"(drug: {sum(1 for c in feature_cols if c.startswith('drug_emb_'))} dims + "
          f"target: {sum(1 for c in feature_cols if c.startswith('target_emb_'))} dims)")
    print(f"  Label counts after merge:\n{pd.Series(y).value_counts().to_string()}")

    if len(merged) == 0:
        raise ValueError(
            "Merge produced zero rows. drug_id or target_id values do not "
            "match between interactions and embeddings."
        )
    if len(np.unique(y)) < 2:
        raise ValueError(
            "Only one class present after merging. Cannot train a classifier."
        )

    idx = np.arange(len(merged))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    train_df = merged.iloc[idx_train].reset_index(drop=True)
    test_df  = merged.iloc[idx_test].reset_index(drop=True)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        min_samples_split=30,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    class_to_idx = {cls: i for i, cls in enumerate(clf.classes_)}
    pos_idx = class_to_idx.get(1, None)

    y_pred_test = clf.predict(X_test)
    y_prob_test = (
        clf.predict_proba(X_test)[:, pos_idx]
        if pos_idx is not None else np.zeros(len(y_test))
    )

    y_pred_train = clf.predict(X_train)
    y_prob_train = (
        clf.predict_proba(X_train)[:, pos_idx]
        if pos_idx is not None else np.zeros(len(y_train))
    )

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()

    metrics = {
        "total_pairs_after_merge": int(len(merged)),
        "train_rows":              int(len(X_train)),
        "test_rows":               int(len(X_test)),
        "n_features":              int(X_train.shape[1]),
        "positive_rate_train":     float(np.mean(y_train)),
        "positive_rate_test":      float(np.mean(y_test)),
        "test_roc_auc":   _safe_auc(y_test, y_prob_test),
        "test_pr_auc":    float(average_precision_score(y_test, y_prob_test)),
        "test_f1":        float(f1_score(y_test, y_pred_test, zero_division=0)),
        "test_accuracy":  float(accuracy_score(y_test, y_pred_test)),
        "test_precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "test_recall":    float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "test_tn": int(tn), "test_fp": int(fp),
        "test_fn": int(fn), "test_tp": int(tp),
        "train_roc_auc":  _safe_auc(y_train, y_prob_train),
        "train_pr_auc":   float(average_precision_score(y_train, y_prob_train)),
        "train_f1":       float(f1_score(y_train, y_pred_train, zero_division=0)),
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
    }
    return clf, metrics, feature_cols, train_df, test_df


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Baseline RF: random 80/20 split."
        )
    )
    parser.add_argument(
        "--interactions_csv", type=Path,
        default=project_root / "data" / "raw" / "chembl_pd_interactions.csv",
    )
    parser.add_argument(
        "--drug_embeddings_csv", type=Path,
        default=project_root / "data" / "processed" / "chembl_drug_embeddings.csv",
    )
    parser.add_argument(
        "--protein_embeddings_csv", type=Path,
        default=project_root / "data" / "processed" / "protein_embeddings.csv",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction held out for testing (default: 0.2).",
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--artifacts_dir", type=Path,
        default=project_root / "random_forest" / "rf_baseline",
        help="Where to save model, metadata, and metrics.",
    )
    args = parser.parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    clf, metrics, feature_cols, train_df, test_df = train_and_evaluate(
        interactions_csv=args.interactions_csv,
        drug_embeddings_csv=args.drug_embeddings_csv,
        protein_embeddings_csv=args.protein_embeddings_csv,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model_path    = args.artifacts_dir / "rf_model.pkl"
    metadata_path = args.artifacts_dir / "rf_metadata.json"
    metrics_path  = args.artifacts_dir / "metrics.json"
    test_csv_path = args.artifacts_dir / "test_set.csv"

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    metadata = {
        "mode": "baseline_random_split",
        "feature_cols": feature_cols,
        "train_target_ids": sorted(train_df["target_id"].astype(str).unique().tolist()),
        "uses_phenotype_features": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    test_df.to_csv(test_csv_path, index=False)

    print("  BASELINE RF — RESULTS SUMMARY")
    print("\n  Held-out test set:")
    for k in ["test_roc_auc", "test_pr_auc", "test_f1",
              "test_accuracy", "test_precision", "test_recall"]:
        print(f"    {k}: {metrics[k]:.4f}")
    print(f"\n  Confusion matrix (test set):")
    print(f"    TN={metrics['test_tn']}  FP={metrics['test_fp']}")
    print(f"    FN={metrics['test_fn']}  TP={metrics['test_tp']}")
    print(f"\n  Train set:")
    for k in ["train_roc_auc", "train_pr_auc", "train_f1", "train_accuracy"]:
        print(f"    {k}: {metrics[k]:.4f}")
    print(f"\n  Saved to: {args.artifacts_dir}")


if __name__ == "__main__":
    main()