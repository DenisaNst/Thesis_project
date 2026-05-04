"""
train_rf_cv.py
--------------
Research Question 1 & 2 extension:
  Uses time-aware cross-validation (TimeSeriesSplit) combined with
  GridSearchCV to find the best regularized Random Forest hyperparameters.
  This addresses the severe overfitting gap (train AUC ~0.999 vs test ~0.760)
  observed in the baseline experiments.

Why TimeSeriesSplit instead of standard k-fold:
  Standard k-fold randomly mixes past and future data across folds,
  reintroducing the same data leakage we fixed with time-slice evaluation.
  TimeSeriesSplit always trains on past folds and validates on future folds,
  preserving temporal ordering throughout cross-validation.

What this script produces:
  - Best hyperparameters found by CV
  - Final model trained on all pre-2018 data with best hyperparameters
  - Honest evaluation on post-2018 test set
  - Comparison of train vs test AUC to quantify overfitting gap
  - Saves to artifacts/rf_cv/

Usage:
    python src/models/train_rf_cv.py
    python src/models/train_rf_cv.py --protein_embeddings_csv data/processed/drkg_target_embeddings.csv
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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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

# These ranges specifically target the overfitting problem:
# - max_depth: reduce from 20 to force more general trees
# - min_samples_leaf: require more samples per leaf to prevent memorisation
# - min_samples_split: require more samples to make a split
# Total combinations: 3 × 3 × 3 = 27 (fast enough with n_jobs=-1)

PARAM_GRID = {
    "max_depth":         [6, 10, 15],
    "min_samples_leaf":  [1, 5, 20],
    "min_samples_split": [2, 10, 30],
}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_with_cv(
    interactions_csv: Path,
    drug_embeddings_csv: Path,
    protein_embeddings_csv: Path,
    cutoff_year: int = 2018,
    n_cv_splits: int = 5,
    n_estimators: int = 200,
) -> tuple:
    """
    Time-aware cross-validated RF training.

    Steps:
      1. Load and standardise interactions.
      2. Split by year: train=≤cutoff, test=>cutoff.
      3. Sort training data by year for TimeSeriesSplit.
      4. GridSearchCV with TimeSeriesSplit to find best hyperparameters.
      5. Retrain final model on all training data with best params.
      6. Evaluate on held-out post-cutoff test set.
    """

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[step 1/6] Loading interactions and embeddings...")
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
    print(f"\n[step 2/6] Time-slice split (cutoff={cutoff_year})...")
    train_df, test_df = eval_protocol.split_by_date(
        interactions, cutoff_year=cutoff_year
    )
    print(f"  Train: {len(train_df):,} pairs (year ≤ {cutoff_year})")
    print(f"  Test:  {len(test_df):,} pairs  (year > {cutoff_year})")

    # ------------------------------------------------------------------
    # 3. Build feature matrices
    # ------------------------------------------------------------------
    print("\n[step 3/6] Building feature matrices...")
    train_merged, X_train_full, y_train_full, feature_cols = _prepare_matrix(
        train_df, drug_emb, protein_emb
    )
    test_merged,  X_test,       y_test,       _            = _prepare_matrix(
        test_df, drug_emb, protein_emb
    )

    print(f"  Train: {X_train_full.shape[0]:,} rows × {X_train_full.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} rows")
    print(f"  Train positives: {y_train_full.mean()*100:.1f}%")
    print(f"  Test positives:  {y_test.mean()*100:.1f}%")

    # ------------------------------------------------------------------
    # 4. Sort training data by year for TimeSeriesSplit
    #    TimeSeriesSplit requires data ordered chronologically.
    # ------------------------------------------------------------------
    print("\n[step 4/6] Preparing time-ordered CV folds...")

    if "year" in train_merged.columns:
        year_order = train_merged["year"].fillna(
            train_merged["year"].min()
        ).argsort().values
        X_train_sorted = X_train_full[year_order]
        y_train_sorted = y_train_full[year_order]
    else:
        print("  [warn] No year column — using data order for TimeSeriesSplit")
        X_train_sorted = X_train_full
        y_train_sorted = y_train_full

    tscv = TimeSeriesSplit(n_splits=n_cv_splits)

    # Quick check on fold sizes
    print(f"  TimeSeriesSplit with {n_cv_splits} folds:")
    for i, (tr_idx, val_idx) in enumerate(tscv.split(X_train_sorted)):
        print(f"    Fold {i+1}: train={len(tr_idx):,}  val={len(val_idx):,}")

    # ------------------------------------------------------------------
    # 5. Grid search
    # ------------------------------------------------------------------
    print(f"\n[step 5/6] Running GridSearchCV ({len(PARAM_GRID['max_depth']) * len(PARAM_GRID['min_samples_leaf']) * len(PARAM_GRID['min_samples_split'])} combinations × {n_cv_splits} folds)...")
    print("  Scoring metric: ROC-AUC")
    print("  This may take several minutes...\n")

    base_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid_search.fit(X_train_sorted, y_train_sorted)

    print(f"\n  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    # Show CV results summary
    cv_results = pd.DataFrame(grid_search.cv_results_)
    print(f"\n  Top 5 parameter combinations by CV AUC:")
    top5 = cv_results.nlargest(5, "mean_test_score")[
        ["param_max_depth", "param_min_samples_leaf",
         "param_min_samples_split", "mean_test_score",
         "mean_train_score", "std_test_score"]
    ]
    for _, row in top5.iterrows():
        overfit_gap = row["mean_train_score"] - row["mean_test_score"]
        print(
            f"    depth={int(row['param_max_depth']):2d}  "
            f"leaf={int(row['param_min_samples_leaf']):2d}  "
            f"split={int(row['param_min_samples_split']):2d}  "
            f"CV-AUC={row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}  "
            f"overfit-gap={overfit_gap:.4f}"
        )

    # ------------------------------------------------------------------
    # 6. Retrain on full training set with best params, evaluate on test
    # ------------------------------------------------------------------
    print(f"\n[step 6/6] Retraining on full training set with best params...")
    best_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        **grid_search.best_params_,
    )
    best_clf.fit(X_train_full, y_train_full)
    print("  Done.")

    test_metrics  = _evaluate(best_clf, X_test,       y_test)
    train_metrics = _evaluate(best_clf, X_train_full, y_train_full)
    overfit_gap   = train_metrics["roc_auc"] - test_metrics["roc_auc"]

    metrics = {
        "mode": "timeslice_cv_gridsearch",
        "cutoff_year": cutoff_year,
        "best_params": grid_search.best_params_,
        "best_cv_roc_auc": float(grid_search.best_score_),
        "n_cv_splits": n_cv_splits,
        # data sizes
        "train_rows": int(X_train_full.shape[0]),
        "test_rows":  int(X_test.shape[0]),
        "n_features": int(X_train_full.shape[1]),
        "positive_rate_train": float(y_train_full.mean()),
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
            "Time-aware CV GridSearch RF.\n"
            "Finds best regularization to reduce overfitting gap.\n"
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
        "--n_cv_splits", type=int, default=5,
        help="Number of TimeSeriesSplit folds (default: 5).",
    )
    parser.add_argument(
        "--n_estimators", type=int, default=200,
        help="Number of trees (default: 200 — reduced for CV speed).",
    )
    parser.add_argument(
        "--artifacts_dir", type=Path,
        default=project_root / "artifacts" / "rf_cv",
    )
    args = parser.parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    clf, metrics, feature_cols, train_df, test_df, cv_results = train_with_cv(
        interactions_csv=args.interactions_csv,
        drug_embeddings_csv=args.drug_embeddings_csv,
        protein_embeddings_csv=args.protein_embeddings_csv,
        cutoff_year=args.cutoff_year,
        n_cv_splits=args.n_cv_splits,
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
        "mode": "timeslice_cv_gridsearch",
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
    print("  CV GRID SEARCH RF — RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  Best hyperparameters found:")
    for k, v in metrics["best_params"].items():
        print(f"    {k}: {v}")
    print(f"  Best CV ROC-AUC: {metrics['best_cv_roc_auc']:.4f}")

    print(f"\n  Held-out test metrics (honest):")
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
    print(f"    (baseline gap was: ~0.24)")
    print(f"    (smaller gap = less overfitting)")

    print(f"\n  Compare against previous results:")
    print(f"    RF baseline (random):    AUC 0.8887")
    print(f"    RF time-slice (no CV):   AUC 0.7599")
    print(f"    RF time-slice (with CV): AUC {metrics['test_roc_auc']:.4f}  ← this run")

    print(f"\n  Saved to: {args.artifacts_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()