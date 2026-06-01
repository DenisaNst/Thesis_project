"""
What this script does differently from train_rf.py (baseline):
  - Splits data by year (cutoff=2018) instead of randomly.
    Train = interactions discovered up to and including 2018.
    Test  = interactions discovered after 2018 (2019-2024).
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

try:
    from src.evaluation import evaluation_protocol as eval_protocol
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluation import evaluation_protocol as eval_protocol  # type: ignore


def _normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _prepare_matrix(
        interactions_df: pd.DataFrame,
        drug_embeddings_df: pd.DataFrame,
        protein_embeddings_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    interactions_df = _normalize_id_columns(interactions_df)
    drug_embeddings_df = _normalize_id_columns(drug_embeddings_df)
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


def _evaluate(clf, X: np.ndarray, y: np.ndarray) -> dict:
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    return {
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "pr_auc": float(average_precision_score(y, y_prob)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train_and_evaluate_timeslice(
        interactions_csv: Path,
        drug_embeddings_csv: Path,
        protein_embeddings_csv: Path,
        cutoff_year: int = 2018,
) -> tuple:
    interactions = eval_protocol.load_and_standardize_interactions(str(interactions_csv))
    print(f"  Total unique drug-target pairs: {len(interactions)}")
    print(f"  Label counts:\n{interactions['label'].value_counts().to_string()}")

    train_df, test_df = eval_protocol.split_by_date(
        interactions, cutoff_year=cutoff_year
    )
    print(f"  Train (year <= {cutoff_year}): {len(train_df)} pairs")
    print(f"  Test  (year >  {cutoff_year}): {len(test_df)} pairs")

    drug_emb = _normalize_id_columns(pd.read_csv(drug_embeddings_csv))
    protein_emb = _normalize_id_columns(pd.read_csv(protein_embeddings_csv))

    train_merged, X_train, y_train, feature_cols = _prepare_matrix(
        train_df, drug_emb, protein_emb
    )
    test_merged, X_test, y_test, _ = _prepare_matrix(
        test_df, drug_emb, protein_emb
    )

    print(f"  Train: {len(X_train)} rows | "
          f"{X_train.shape[1]} features | "
          f"positives: {y_train.sum()} ({100 * y_train.mean():.1f}%)")
    print(f"  Test:  {len(X_test)} rows  | "
          f"positives: {y_test.sum()} ({100 * y_test.mean():.1f}%)")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        min_samples_split=30,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    test_metrics = _evaluate(clf, X_test, y_test)
    train_metrics = _evaluate(clf, X_train, y_train)

    metrics = {
        "mode": "timeslice_only",
        "cutoff_year": cutoff_year,
        "train_rows": int(len(X_train)),
        "test_rows_initial": int(len(test_df)),
        "test_rows_after_merge": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "positive_rate_train": float(np.mean(y_train)),
        "positive_rate_test": float(np.mean(y_test)),
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_f1": test_metrics["f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_tn": test_metrics["tn"], "test_fp": test_metrics["fp"],
        "test_fn": test_metrics["fn"], "test_tp": test_metrics["tp"],
        "train_roc_auc": train_metrics["roc_auc"],
        "train_pr_auc": train_metrics["pr_auc"],
        "train_f1": train_metrics["f1"],
        "train_accuracy": train_metrics["accuracy"],
    }
    return clf, metrics, feature_cols, train_merged, test_merged


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Time-slice RF training"
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
        "--cutoff_year", type=int, default=2018,
        help="Train on data up to and including this year (default: 2018).",
    )
    parser.add_argument(
        "--artifacts_dir", type=Path,
        default=project_root / "random_forest" / "rf_timeslice",
        help="Where to save model, metadata, and metrics.",
    )
    args = parser.parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    clf, metrics, feature_cols, train_df, test_df = train_and_evaluate_timeslice(
        interactions_csv=args.interactions_csv,
        drug_embeddings_csv=args.drug_embeddings_csv,
        protein_embeddings_csv=args.protein_embeddings_csv,
        cutoff_year=args.cutoff_year,
    )

    model_path = args.artifacts_dir / "rf_model.pkl"
    metadata_path = args.artifacts_dir / "rf_metadata.json"
    metrics_path = args.artifacts_dir / "metrics.json"
    test_csv_path = args.artifacts_dir / "test_set.csv"

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    metadata = {
        "mode": metrics["mode"],
        "cutoff_year": args.cutoff_year,
        "feature_cols": feature_cols,
        "train_target_ids": sorted(train_df["target_id"].astype(str).unique().tolist()),
        "uses_phenotype_features": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    test_df.to_csv(test_csv_path, index=False)

    print("  TIME-SLICE RF — RESULTS SUMMARY")
    print(f"  Cutoff year: {metrics['cutoff_year']}")
    print(f"  Train rows:  {metrics['train_rows']}")
    print(f"  Test rows (after merge): {metrics['test_rows_after_merge']}")

    print(f"\n  Confusion matrix (test set):")
    print(f"    TN={metrics['test_tn']}  FP={metrics['test_fp']}")
    print(f"    FN={metrics['test_fn']}  TP={metrics['test_tp']}")

    print(f"\n  Train metrics:")
    for k in ["train_roc_auc", "train_pr_auc", "train_f1", "train_accuracy"]:
        print(f"    {k}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()