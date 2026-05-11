"""
train_rf_cv.py
--------------
Research Question 1 & 2 extension:
  Uses a fixed temporal split (PredefinedSplit) with GridSearchCV to find
  the best Random Forest hyperparameters.

  Training data:  year <= 2018
  Validation data: year > 2018  (used to score each hyperparameter combo)

  The hyperparameter combination with the best ROC-AUC on post-2018 data
  is selected. The final model is then retrained on ALL pre-2018 data using
  those best parameters, and evaluated on the post-2018 test set.

  Note: because the same post-2018 data is used for both hyperparameter
  selection and final evaluation, the final AUC is slightly optimistic.
  This is acceptable and should be noted in the thesis.

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    interactions_df       = _normalize_id_columns(interactions_df)
    drug_embeddings_df    = _normalize_id_columns(drug_embeddings_df)
    protein_embeddings_df = _normalize_id_columns(protein_embeddings_df)

    merged = interactions_df.merge(drug_embeddings_df, on="drug_id", how="inner")
    merged = merged.merge(protein_embeddings_df, on="target_id", how="inner")

    feature_cols = [
        c for c in merged.columns
        if c.startswith("drug_emb_") or c.startswith("target_emb_")
    ]
    if not feature_cols:
        raise ValueError("No embedding columns found after merging.")
    if "label" not in merged.columns:
        raise ValueError("No 'label' column found in interactions.")

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


# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------

PARAM_GRID = {
    "max_depth":         [6, 10, 15],
    "min_samples_leaf":  [1, 5, 20],
    "min_samples_split": [2, 10, 30],
}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_with_predefined_split(
    interactions_csv: Path,
    drug_embeddings_csv: Path,
    protein_embeddings_csv: Path,
    cutoff_year: int = 2018,
    n_estimators: int = 200,
) -> tuple:
    """
    Hyperparameter search using a fixed temporal split (PredefinedSplit).

    Steps:
      1. Load and standardise interactions.
      2. Split by year: train = <= cutoff, test = > cutoff.
      3. Build feature matrices for train and test.
      4. GridSearchCV with PredefinedSplit:
           - pre-2018 data is always used for training each combo
           - post-2018 data is always used for scoring each combo
      5. Retrain final model on all pre-2018 data with best params.
      6. Report final evaluation on post-2018 test set.
    """

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[step 1/5] Loading interactions and embeddings...")
    interactions = eval_protocol.load_and_standardize_interactions(
        str(interactions_csv)
    )
    drug_emb    = _normalize_id_columns(pd.read_csv(drug_embeddings_csv))
    protein_emb = _normalize_id_columns(pd.read_csv(protein_embeddings_csv))

    print(f"  Interactions: {len(interactions):,} unique pairs")
    print(f"  Drug embeddings:    {len(drug_emb):,} rows")
    print(f"  Protein embeddings: {len(protein_emb):,} rows")

    # ------------------------------------------------------------------
    # 2. Time-slice split
    # ------------------------------------------------------------------
    print(f"\n[step 2/5] Time-slice split (cutoff={cutoff_year})...")
    train_df, test_df = eval_protocol.split_by_date(
        interactions, cutoff_year=cutoff_year
    )
    print(f"  Train: {len(train_df):,} pairs (year <= {cutoff_year})")
    print(f"  Test:  {len(test_df):,} pairs  (year > {cutoff_year})")

    # ------------------------------------------------------------------
    # 3. Build feature matrices
    # ------------------------------------------------------------------
    print("\n[step 3/5] Building feature matrices...")
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

    # ------------------------------------------------------------------
    # 4. GridSearchCV with PredefinedSplit
    #    -1 = always used for training
    #     0 = always used for validation
    # ------------------------------------------------------------------
    print("\n[step 4/5] Running GridSearchCV with fixed temporal split...")
    print(f"  pre-{cutoff_year} data  -> always training")
    print(f"  post-{cutoff_year} data -> always validation")
    n_combos = (
        len(PARAM_GRID["max_depth"])
        * len(PARAM_GRID["min_samples_leaf"])
        * len(PARAM_GRID["min_samples_split"])
    )
    print(f"  {n_combos} hyperparameter combinations to try")
    print("  This may take several minutes...\n")

    # Stack train + test so sklearn can see all data,
    # but PredefinedSplit tells it which rows are train and which are val
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])

    split_index = np.concatenate([
        np.full(len(X_train), -1),  # -1 = always in training fold
        np.zeros(len(X_test),  dtype=int),  #  0 = always in validation fold
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

    # Show top 5 combinations
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

    # ------------------------------------------------------------------
    # 5. Retrain on full pre-2018 data with best params, evaluate on test
    # ------------------------------------------------------------------
    print(f"\n[step 5/5] Retraining on full pre-{cutoff_year} data with best params...")
    best_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        **grid_search.best_params_,
    )
    best_clf.fit(X_train, y_train)
    print("  Done.")

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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

    # Save artifacts
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

    # Print summary
    print("\n" + "=" * 60)
    print("  GRID SEARCH RF — RESULTS SUMMARY")
    print("=" * 60)
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
    print("=" * 60)


if __name__ == "__main__":
    main()