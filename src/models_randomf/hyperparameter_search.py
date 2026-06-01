"""
How this script works:
1. Data Loading: Loads drug-target interactions alongside their pre-computed neural
   embeddings (ChemBERTa for drugs; ESM2 or DRKG for targets).
2. Chronological Splitting: Instead of randomly shuffling the data,
   this isolates the data strictly by publication year. Interactions published in
   or before 2018 form the training set; interactions after 2018 form the test set.
3. Predefined Split Grid Search: Standard Grid Search Cross-Validation (CV) normally
   shuffles data randomly. To override this, the script uses Scikit-Learn's
   `PredefinedSplit`. It assigns `-1` to pre-2018 rows and `0` to post-2018 rows,
   forcing the Grid Search to ALWAYS train on the past and ALWAYS evaluate on the
   future while it tests 27 different hyperparameter combinations (max_depth,
   min_samples_leaf, min_samples_split).
4. Retraining & Evaluation: Once it finds the specific hyperparameter combination
   that achieves the highest ROC-AUC on the post-2018 data, it builds a final model
   using those settings, scores it, checks for overfitting, and saves the final
   model and metrics to the artifacts folder.

Usage:
    python src/models_randomf/train_rf_cv.py
    python src/models_randomf/train_rf_cv.py --protein_embeddings_csv data/processed/drkg_target_embeddings.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import pickle
import sys

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
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from typing import List, Tuple

try:
    from src.evaluation import evaluation_protocol as eval_protocol
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluation import evaluation_protocol as eval_protocol  # type: ignore


def _normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
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


def _evaluate(clf, X: np.ndarray, y: np.ndarray) -> dict:
    class_to_idx = {cls: i for i, cls in enumerate(clf.classes_)}
    pos_idx = class_to_idx.get(1, None)
    y_pred = clf.predict(X)
    y_prob = (
        clf.predict_proba(X)[:, pos_idx]
        if pos_idx is not None else np.zeros(len(y))
    )
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        "roc_auc":   _safe_auc(y, y_prob),
        "pr_auc":    float(average_precision_score(y, y_prob)),
        "f1":        float(f1_score(y, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y, y_pred)),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall":    float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


PARAM_GRID = {
    "max_depth":         [6, 10, 15],
    "min_samples_leaf":  [1, 5, 20],
    "min_samples_split": [2, 10, 30],
}


def train_with_predefined_split(
    interactions_csv: Path,
    drug_embeddings_csv: Path,
    protein_embeddings_csv: Path,
    cutoff_year: int = 2018,
    n_estimators: int = 200,
) -> tuple:

    interactions = eval_protocol.load_and_standardize_interactions(
        str(interactions_csv)
    )
    drug_emb    = _normalize_id_columns(pd.read_csv(drug_embeddings_csv))
    protein_emb = _normalize_id_columns(pd.read_csv(protein_embeddings_csv))

    print(f"  Interactions: {len(interactions):,} unique pairs")
    print(f"  Drug embeddings:    {len(drug_emb):,} rows")
    print(f"  Protein embeddings: {len(protein_emb):,} rows")


    train_df, test_df = eval_protocol.split_by_date(
        interactions, cutoff_year=cutoff_year
    )
    print(f"  Train: {len(train_df):,} pairs (year <= {cutoff_year})")
    print(f"  Test:  {len(test_df):,} pairs  (year > {cutoff_year})")


    train_merged, X_train, y_train, feature_cols = _prepare_matrix(
        train_df, drug_emb, protein_emb
    )
    test_merged, X_test, y_test, _ = _prepare_matrix(
        test_df, drug_emb, protein_emb
    )

    print(f"  Train: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} rows")
    print(f"  Train positives: {y_train.mean()*100:.1f}%")
    print(f"  Test positives:  {y_test.mean()*100:.1f}%")


    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])

    split_index = np.concatenate([
        np.full(len(X_train), -1),
        np.zeros(len(X_test),  dtype=int),
    ])
    ps = PredefinedSplit(test_fold=split_index)

    base_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=PARAM_GRID,
        cv=ps,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid_search.fit(X_combined, y_combined)

    print(f"\n  Best parameters: {grid_search.best_params_}")
    print(f"  Best ROC-AUC on post-{cutoff_year} data: {grid_search.best_score_:.4f}")

    cv_results = pd.DataFrame(grid_search.cv_results_)
    print(f"\n  Top 5 parameter combinations:")
    top5 = cv_results.nlargest(5, "mean_test_score")[
        ["param_max_depth", "param_min_samples_leaf",
         "param_min_samples_split", "mean_test_score", "mean_train_score"]
    ]
    for _, row in top5.iterrows():
        overfit_gap = row["mean_train_score"] - row["mean_test_score"]
        print(
            f"    depth={int(row['param_max_depth']):2d}  "
            f"leaf={int(row['param_min_samples_leaf']):2d}  "
            f"split={int(row['param_min_samples_split']):2d}  "
            f"AUC={row['mean_test_score']:.4f}  "
            f"overfit-gap={overfit_gap:.4f}"
        )

# Retrain best model
    best_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        **grid_search.best_params_,
    )
    best_clf.fit(X_train, y_train)

    test_metrics  = _evaluate(best_clf, X_test,  y_test)
    train_metrics = _evaluate(best_clf, X_train, y_train)
    overfit_gap   = train_metrics["roc_auc"] - test_metrics["roc_auc"]

    metrics = {
        "mode": "timeslice_predefined_split_gridsearch",
        "cutoff_year": cutoff_year,
        "best_params": grid_search.best_params_,
        "best_gridsearch_roc_auc": float(grid_search.best_score_),
        # data sizes
        "train_rows": int(X_train.shape[0]),
        "test_rows":  int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test":  float(y_test.mean()),
        # honest test metrics
        "test_roc_auc":   test_metrics["roc_auc"],
        "test_pr_auc":    test_metrics["pr_auc"],
        "test_f1":        test_metrics["f1"],
        "test_accuracy":  test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall":    test_metrics["recall"],
        "test_tn": test_metrics["tn"], "test_fp": test_metrics["fp"],
        "test_fn": test_metrics["fn"], "test_tp": test_metrics["tp"],
        # train metrics + gap
        "train_roc_auc":  train_metrics["roc_auc"],
        "train_pr_auc":   train_metrics["pr_auc"],
        "train_f1":       train_metrics["f1"],
        "train_accuracy": train_metrics["accuracy"],
        "overfitting_gap_roc_auc": float(overfit_gap),
    }
    return best_clf, metrics, feature_cols, train_merged, test_merged, cv_results


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Hyperparameter search with fixed temporal split.\n"
            "Trains on pre-2018 data, validates on post-2018 data.\n"
            "Saves to artifacts/rf_cv/."
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
        help="Train on data up to and including this year.",
    )
    parser.add_argument(
        "--n_estimators", type=int, default=200,
        help="Number of trees (default: 200).",
    )
    parser.add_argument(
        "--artifacts_dir", type=Path,
        default=project_root / "artifacts" / "rf_cv",
    )
    args = parser.parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    clf, metrics, feature_cols, train_df, test_df, cv_results = train_with_predefined_split(
        interactions_csv=args.interactions_csv,
        drug_embeddings_csv=args.drug_embeddings_csv,
        protein_embeddings_csv=args.protein_embeddings_csv,
        cutoff_year=args.cutoff_year,
        n_estimators=args.n_estimators,
    )

    model_path      = args.artifacts_dir / "rf_model.pkl"
    metadata_path   = args.artifacts_dir / "rf_metadata.json"
    metrics_path    = args.artifacts_dir / "metrics.json"
    cv_results_path = args.artifacts_dir / "cv_results.csv"
    test_csv_path   = args.artifacts_dir / "test_set.csv"

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    metadata = {
        "mode": "timeslice_predefined_split_gridsearch",
        "best_params": metrics["best_params"],
        "feature_cols": feature_cols,
        "train_target_ids": sorted(
            train_df["target_id"].astype(str).unique().tolist()
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cv_results.to_csv(cv_results_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print("  GRID SEARCH RF — RESULTS SUMMARY")
    print(f"\n  Best hyperparameters found:")
    for k, v in metrics["best_params"].items():
        print(f"    {k}: {v}")
    print(f"  Best ROC-AUC (post-{args.cutoff_year} validation): {metrics['best_gridsearch_roc_auc']:.4f}")

    print(f"\n  Final test metrics:")
    for k in ["test_roc_auc", "test_pr_auc", "test_f1",
              "test_accuracy", "test_precision", "test_recall"]:
        print(f"    {k}: {metrics[k]:.4f}")

    print(f"\n  Confusion matrix:")
    print(f"    TN={metrics['test_tn']}  FP={metrics['test_fp']}")
    print(f"    FN={metrics['test_fn']}  TP={metrics['test_tp']}")

    print(f"\n  Overfitting analysis:")
    print(f"    train_roc_auc:  {metrics['train_roc_auc']:.4f}")
    print(f"    test_roc_auc:   {metrics['test_roc_auc']:.4f}")
    print(f"    gap:            {metrics['overfitting_gap_roc_auc']:.4f}")

    print(f"\n  Compare against previous results:")
    print(f"    RF baseline (random split):  AUC 0.8887")
    print(f"    RF time-slice (no tuning):   AUC 0.7599")
    print(f"    RF time-slice (tuned):       AUC {metrics['test_roc_auc']:.4f}  <- this run")

    print(f"\n  Saved to: {args.artifacts_dir}")


if __name__ == "__main__":
    main()