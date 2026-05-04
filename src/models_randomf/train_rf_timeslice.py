from __future__ import annotations

"""
train_rf_timeslice.py
---------------------
Research Question 2:
  How does a time-slice evaluation strategy combined with Double-Member
  Exclusion impact the predictive reliability of drug repositioning models_GNN?

What this script does differently from train_rf.py (baseline):
  - Splits data by year (cutoff=2018) instead of randomly.
    Train = interactions discovered up to and including 2018.
    Test  = interactions discovered after 2018 (2019-2024).
  - Optionally applies Double-Member Exclusion (DME) to the test set:
    removes any test pair where the drug OR the target already appeared
    in training. This forces the model to generalise to unseen entities.

Why keep this separate from train_rf.py:
  - You can rerun the baseline and this script independently.
  - Each produces its own random_forest so results never overwrite each other.
  - The comparison between the two is your answer to RQ2.
"""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ChEMBL-style ID columns to the standard pipeline names."""
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
    """
    Merge interactions with embeddings and return feature matrix + labels.
    Feature vector = [drug_emb_0..767, target_emb_0..479] = 1,248 dims.
    """
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
        raise ValueError(
            "No embedding feature columns found after merging. "
            "Expected columns starting with 'drug_emb_' and 'target_emb_'."
        )
    if "label" not in merged.columns:
        raise ValueError("No 'label' column found. Interactions CSV must have 0/1 labels.")

    X = merged[feature_cols].to_numpy(dtype=np.float32)
    y = merged["label"].to_numpy(dtype=int)
    return merged, X, y, feature_cols


def _evaluate(clf, X: np.ndarray, y: np.ndarray) -> dict:
    """Compute all metrics for a given split."""
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
# Main training function
# ---------------------------------------------------------------------------

def train_and_evaluate_timeslice(
    interactions_csv: Path,
    drug_embeddings_csv: Path,
    protein_embeddings_csv: Path,
    cutoff_year: int = 2018,
    apply_dme: bool = True,
) -> tuple:
    """
    Time-slice training and evaluation.

    Parameters
    ----------
    cutoff_year:
        Interactions from this year and earlier go into training.
        Interactions after this year go into the test set.
    apply_dme:
        If True, apply Double-Member Exclusion to the test set.
        This removes test pairs where the drug OR target was seen in training,
        forcing the model to generalise to completely new entities.
    """

    # ------------------------------------------------------------------
    # 1. Load and standardise interactions
    # ------------------------------------------------------------------
    print("[step 1/6] Loading and standardising interactions...")
    interactions = eval_protocol.load_and_standardize_interactions(str(interactions_csv))
    print(f"  Total unique drug-target pairs: {len(interactions)}")
    print(f"  Label counts:\n{interactions['label'].value_counts().to_string()}")

    year_coverage = interactions['year'].notna().sum()
    print(f"  Pairs with a valid year: {year_coverage} "
          f"({100*year_coverage/len(interactions):.1f}%)")

    # ------------------------------------------------------------------
    # 2. Time-slice split
    # ------------------------------------------------------------------
    print(f"\n[step 2/6] Splitting by year (cutoff = {cutoff_year})...")
    train_df, test_df = eval_protocol.split_by_date(
        interactions, cutoff_year=cutoff_year
    )
    print(f"  Train (year <= {cutoff_year}): {len(train_df)} pairs")
    print(f"  Test  (year >  {cutoff_year}): {len(test_df)} pairs")
    print(f"  Train label counts:\n{train_df['label'].value_counts().to_string()}")
    print(f"  Test  label counts:\n{test_df['label'].value_counts().to_string()}")

    # Sanity check: train set must have both classes
    if len(train_df) == 0:
        raise ValueError("Train set is empty after time-slice. Check your cutoff year.")
    if train_df['label'].nunique() < 2:
        raise ValueError("Train set has only one class after time-slice.")

    # ------------------------------------------------------------------
    # 3. Double-Member Exclusion (optional)
    # ------------------------------------------------------------------
    test_df_final = test_df.copy()
    dme_applied = False

    if apply_dme:
        print(f"\n[step 3/6] Applying Double-Member Exclusion...")
        print(f"  Unique drugs in train:   {train_df['drug_id'].nunique()}")
        print(f"  Unique targets in train: {train_df['target_id'].nunique()}")

        test_df_dme = eval_protocol.double_member_exclusion(train_df, test_df)

        print(f"  Test pairs before DME: {len(test_df)}")
        print(f"  Test pairs after DME:  {len(test_df_dme)} "
              f"({len(test_df) - len(test_df_dme)} removed)")

        if len(test_df_dme) == 0:
            # This is expected given only 63 targets — warn loudly but don't crash.
            print("\n  [WARNING] DME removed ALL test pairs.")
            print("  This happens because all 63 PD targets appear in the training set.")
            print("  DME requires both a new drug AND a new target — with only 63 targets")
            print("  this condition is almost never met.")
            print("  Falling back to time-slice only (no DME) for evaluation.")
            print("  This is an important finding to discuss in your thesis.\n")
            test_df_final = test_df.copy()
            dme_applied = False
        else:
            if test_df_dme['label'].nunique() < 2:
                print("  [WARNING] DME test set has only one class — cannot compute AUC.")
                print("  Falling back to time-slice only for evaluation.")
                test_df_final = test_df.copy()
                dme_applied = False
            else:
                test_df_final = test_df_dme
                dme_applied = True
                print(f"  Label counts after DME:\n"
                      f"{test_df_final['label'].value_counts().to_string()}")
    else:
        print(f"\n[step 3/6] Skipping Double-Member Exclusion (--no_dme flag set).")

    # ------------------------------------------------------------------
    # 4. Load embeddings
    # ------------------------------------------------------------------
    print(f"\n[step 4/6] Loading embeddings...")
    drug_emb    = _normalize_id_columns(pd.read_csv(drug_embeddings_csv))
    protein_emb = _normalize_id_columns(pd.read_csv(protein_embeddings_csv))
    print(f"  Drug embeddings:    {len(drug_emb)} rows")
    print(f"  Protein embeddings: {len(protein_emb)} rows")

    # ------------------------------------------------------------------
    # 5. Build feature matrices for train and test separately
    # ------------------------------------------------------------------
    print(f"\n[step 5/6] Building feature matrices...")
    train_merged, X_train, y_train, feature_cols = _prepare_matrix(
        train_df, drug_emb, protein_emb
    )
    test_merged, X_test, y_test, _ = _prepare_matrix(
        test_df_final, drug_emb, protein_emb
    )

    print(f"  Train: {len(X_train)} rows | "
          f"{X_train.shape[1]} features | "
          f"positives: {y_train.sum()} ({100*y_train.mean():.1f}%)")
    print(f"  Test:  {len(X_test)} rows  | "
          f"positives: {y_test.sum()} ({100*y_test.mean():.1f}%)")

    if len(X_train) == 0:
        raise ValueError("Train feature matrix is empty after merge with embeddings.")
    if len(X_test) == 0:
        raise ValueError(
            "Test feature matrix is empty after merge with embeddings. "
            "Check that test drugs/targets have embeddings."
        )

    # ------------------------------------------------------------------
    # 6. Train Random Forest
    # ------------------------------------------------------------------
    print(f"\n[step 6/6] Training Random Forest...")
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
    print("  Done.")

    # Evaluate on both sets
    test_metrics  = _evaluate(clf, X_test,  y_test)
    train_metrics = _evaluate(clf, X_train, y_train)

    metrics = {
        # --- configuration ---
        "mode": "timeslice_with_dme" if dme_applied else "timeslice_only",
        "cutoff_year": cutoff_year,
        "dme_applied": dme_applied,
        # --- data sizes ---
        "train_rows":          int(len(X_train)),
        "test_rows_before_dme": int(len(test_df)),
        "test_rows_after_dme":  int(len(test_df_final)),
        "test_rows_after_merge": int(len(X_test)),
        "n_features":          int(X_train.shape[1]),
        "positive_rate_train": float(np.mean(y_train)),
        "positive_rate_test":  float(np.mean(y_test)),
        # --- honest test metrics (use these in your thesis) ---
        "test_roc_auc":   test_metrics["roc_auc"],
        "test_pr_auc":    test_metrics["pr_auc"],
        "test_f1":        test_metrics["f1"],
        "test_accuracy":  test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall":    test_metrics["recall"],
        "test_tn": test_metrics["tn"], "test_fp": test_metrics["fp"],
        "test_fn": test_metrics["fn"], "test_tp": test_metrics["tp"],
        # --- train metrics (for overfitting comparison) ---
        "train_roc_auc":  train_metrics["roc_auc"],
        "train_pr_auc":   train_metrics["pr_auc"],
        "train_f1":       train_metrics["f1"],
        "train_accuracy": train_metrics["accuracy"],
    }

    return clf, metrics, feature_cols, train_merged, test_merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Time-slice + DME RF training.\n"
            "Answers RQ2: how does time-slice + DME affect predictive reliability?\n"
            "Compare metrics.json here vs random_forest/rf_baseline/metrics.json."
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
        "--no_dme", action="store_true",
        help="Skip Double-Member Exclusion (run time-slice only).",
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
        apply_dme=not args.no_dme,
    )

    # Save random_forest
    model_path    = args.artifacts_dir / "rf_model.pkl"
    metadata_path = args.artifacts_dir / "rf_metadata.json"
    metrics_path  = args.artifacts_dir / "metrics.json"
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

    # Print summary
    print("\n" + "=" * 60)
    print("  TIME-SLICE + DME RF — RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Mode:        {metrics['mode']}")
    print(f"  Cutoff year: {metrics['cutoff_year']}")
    print(f"  Train rows:  {metrics['train_rows']}")
    print(f"  Test rows (after DME + merge): {metrics['test_rows_after_merge']}")

    print(f"\n  Held-out test metrics (use these for your thesis):")
    for k in ["test_roc_auc", "test_pr_auc", "test_f1",
              "test_accuracy", "test_precision", "test_recall"]:
        print(f"    {k}: {metrics[k]:.4f}")

    print(f"\n  Confusion matrix (test set):")
    print(f"    TN={metrics['test_tn']}  FP={metrics['test_fp']}")
    print(f"    FN={metrics['test_fn']}  TP={metrics['test_tp']}")

    print(f"\n  Train metrics (overfitting check):")
    for k in ["train_roc_auc", "train_pr_auc", "train_f1", "train_accuracy"]:
        print(f"    {k}: {metrics[k]:.4f}")

    print(f"\n  Baseline to compare against: random_forest/rf_baseline/metrics.json")
    print(f"  Saved to: {args.artifacts_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()