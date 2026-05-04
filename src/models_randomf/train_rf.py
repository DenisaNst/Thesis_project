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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """ROC-AUC is undefined when only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _prepare_matrix(
    interactions_df: pd.DataFrame,
    drug_embeddings_df: pd.DataFrame,
    protein_embeddings_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Merge interactions with drug and protein embeddings, then extract the
    feature matrix X and label vector y.

    Feature vector per row = [drug_emb_0..767, target_emb_0..479]
    That is ~1,248 numbers describing one (drug, target) pair.
    """
    # Normalise column names in all three dataframes.
    interactions_df       = _normalize_id_columns(interactions_df)
    drug_embeddings_df    = _normalize_id_columns(drug_embeddings_df)
    protein_embeddings_df = _normalize_id_columns(protein_embeddings_df)

    # Inner join: keeps only rows where BOTH drug AND target have embeddings.
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
        raise ValueError(
            "No 'label' column found after merging. "
            "Your interactions CSV must have a label column with 0s and 1s."
        )

    X = merged[feature_cols].to_numpy(dtype=np.float32)
    y = merged["label"].to_numpy(dtype=int)
    return merged, X, y, feature_cols


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_and_evaluate(
    interactions_csv: Path,
    drug_embeddings_csv: Path,
    protein_embeddings_csv: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, dict, list[str], pd.DataFrame, pd.DataFrame]:
    """
    BASELINE mode: random 80/20 train/test split, no time-slice, no DME.

    Steps:
      1. Load & standardise interactions (dedup to one row per drug-target pair).
      2. Load drug and protein embeddings, normalise column names.
      3. Merge everything into a single feature matrix.
      4. Random stratified 80/20 split.
      5. Fit Random Forest on 80%.
      6. Evaluate on held-out 20% — these are the honest metrics.
    """

    # ------------------------------------------------------------------
    # 1. Load interactions
    # ------------------------------------------------------------------
    print("[step 1/5] Loading and standardising interactions...")
    interactions = eval_protocol.load_and_standardize_interactions(str(interactions_csv))
    print(f"  Unique drug-target pairs after deduplication: {len(interactions)}")
    print(f"  Label counts:\n{interactions['label'].value_counts().to_string()}")

    # ------------------------------------------------------------------
    # 2. Load embeddings
    # ------------------------------------------------------------------
    print("\n[step 2/5] Loading embeddings...")
    drug_emb    = _normalize_id_columns(pd.read_csv(drug_embeddings_csv))
    protein_emb = _normalize_id_columns(pd.read_csv(protein_embeddings_csv))

    print(f"  Drug embeddings:    {len(drug_emb)} rows  "
          f"| id col: {'drug_id' if 'drug_id' in drug_emb.columns else '*** MISSING drug_id ***'}")
    print(f"  Protein embeddings: {len(protein_emb)} rows  "
          f"| id col: {'target_id' if 'target_id' in protein_emb.columns else '*** MISSING target_id ***'}")

    # ------------------------------------------------------------------
    # 3. Build feature matrix
    # ------------------------------------------------------------------
    print("\n[step 3/5] Merging interactions + embeddings...")
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

    # ------------------------------------------------------------------
    # 4. Random stratified 80/20 split
    #    Stratified = keeps the same positive/negative ratio in both sets.
    # ------------------------------------------------------------------
    print("\n[step 4/5] Splitting into train (80%) / test (20%)...")
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

    print(f"  Train: {len(X_train)} rows | "
          f"positives: {y_train.sum()} ({100*y_train.mean():.1f}%)")
    print(f"  Test:  {len(X_test)} rows  | "
          f"positives: {y_test.sum()} ({100*y_test.mean():.1f}%)")

    # ------------------------------------------------------------------
    # 5. Train Random Forest
    # ------------------------------------------------------------------
    print("\n[step 5/5] Training Random Forest...")
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
    print("  Done.")

    # ------------------------------------------------------------------
    # Evaluate on held-out test set  ← these are your honest metrics
    # ------------------------------------------------------------------
    class_to_idx = {cls: i for i, cls in enumerate(clf.classes_)}
    pos_idx = class_to_idx.get(1, None)

    y_pred_test = clf.predict(X_test)
    y_prob_test = (
        clf.predict_proba(X_test)[:, pos_idx]
        if pos_idx is not None else np.zeros(len(y_test))
    )

    # Train metrics too so you can see the overfitting gap clearly.
    y_pred_train = clf.predict(X_train)
    y_prob_train = (
        clf.predict_proba(X_train)[:, pos_idx]
        if pos_idx is not None else np.zeros(len(y_train))
    )

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()

    metrics = {
        # --- data sizes ---
        "total_pairs_after_merge": int(len(merged)),
        "train_rows":              int(len(X_train)),
        "test_rows":               int(len(X_test)),
        "n_features":              int(X_train.shape[1]),
        "positive_rate_train":     float(np.mean(y_train)),
        "positive_rate_test":      float(np.mean(y_test)),
        # --- HONEST held-out metrics (use these in your thesis) ---
        "test_roc_auc":   _safe_auc(y_test, y_prob_test),
        "test_pr_auc":    float(average_precision_score(y_test, y_prob_test)),
        "test_f1":        float(f1_score(y_test, y_pred_test, zero_division=0)),
        "test_accuracy":  float(accuracy_score(y_test, y_pred_test)),
        "test_precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "test_recall":    float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "test_tn": int(tn), "test_fp": int(fp),
        "test_fn": int(fn), "test_tp": int(tp),
        # --- train metrics (will be higher; the gap shows overfitting) ---
        "train_roc_auc":  _safe_auc(y_train, y_prob_train),
        "train_pr_auc":   float(average_precision_score(y_train, y_prob_train)),
        "train_f1":       float(f1_score(y_train, y_pred_train, zero_division=0)),
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
    }

    return clf, metrics, feature_cols, train_df, test_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Baseline RF: random 80/20 split, no time-slice, no DME.\n"
            "Establishes the naive baseline for Research Question 1."
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

    # Save random_forest
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

    # Print summary
    print("\n" + "=" * 55)
    print("  BASELINE RF — RESULTS SUMMARY")
    print("=" * 55)
    print("\n  Held-out test set (use these for your thesis):")
    for k in ["test_roc_auc", "test_pr_auc", "test_f1",
              "test_accuracy", "test_precision", "test_recall"]:
        print(f"    {k}: {metrics[k]:.4f}")
    print(f"\n  Confusion matrix (test set):")
    print(f"    TN={metrics['test_tn']}  FP={metrics['test_fp']}")
    print(f"    FN={metrics['test_fn']}  TP={metrics['test_tp']}")
    print(f"\n  Train set (compare to test to check overfitting gap):")
    for k in ["train_roc_auc", "train_pr_auc", "train_f1", "train_accuracy"]:
        print(f"    {k}: {metrics[k]:.4f}")
    print(f"\n  Saved to: {args.artifacts_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()