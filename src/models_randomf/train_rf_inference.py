"""
Trains the final inference model for drug repositioning predictions.

Unlike train_rf_cv.py (which uses a time-slice split for honest evaluation),
this script trains on ALL available ChEMBL interactions, no data is held back.

Hyperparameters are found via RandomizedSearchCV with 5-fold stratified CV.
RandomizedSearchCV is preferred over GridSearchCV because it:
  - Explores a much wider and continuous hyperparameter space
  - Finds equally good or better params in the same number of fits
  - Is considered more state-of-the-art than exhaustive grid search

Pipeline:
  1. Load ALL ChEMBL interactions (no time-slice)
  2. Merge with drug + protein embeddings
  3. RandomizedSearchCV (50 iterations, 5-fold stratified CV)
  4. Retrain on ALL data with best params
  5. Save model + metadata → ready for random_forest.py inference

Usage:
    # With ESM2 protein embeddings (default)
    python src/models_randomf/train_rf_inference.py

    # With DRKG protein embeddings
    python src/models_randomf/train_rf_inference.py \
        --protein_embeddings_csv data/processed/drkg_target_embeddings.csv \
        --artifacts_dir artifacts/rf_inference_drkg
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from typing import List, Tuple

try:
    from src.evaluation import evaluation_protocol as eval_protocol
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluation import evaluation_protocol as eval_protocol  # type: ignore

PARAM_DISTRIBUTIONS = {
    "max_depth":         randint(5, 25),
    "min_samples_leaf":  randint(1, 30),
    "min_samples_split": randint(2, 40),
    "n_estimators":      randint(100, 400),
}


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


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Train final inference RF on ALL ChEMBL data with randomized CV search."
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
        "--n_iter", type=int, default=50,
        help="Number of random hyperparameter combinations to try (default: 50)."
    )
    parser.add_argument(
        "--cv_folds", type=int, default=5,
        help="Number of CV folds for hyperparameter search (default: 5)."
    )
    parser.add_argument(
        "--artifacts_dir", type=Path,
        default=project_root / "artifacts" / "rf_inference",
    )
    args = parser.parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    interactions = eval_protocol.load_and_standardize_interactions(
        str(args.interactions_csv)
    )
    drug_emb    = _normalize_id_columns(pd.read_csv(args.drug_embeddings_csv))
    protein_emb = _normalize_id_columns(pd.read_csv(args.protein_embeddings_csv))

    print(f"  Interactions: {len(interactions):,} unique pairs")
    print(f"  Drug embeddings:    {len(drug_emb):,} rows")
    print(f"  Protein embeddings: {len(protein_emb):,} rows")

    merged, X, y, feature_cols = _prepare_matrix(
        interactions, drug_emb, protein_emb
    )

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    base_clf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    random_search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=args.n_iter,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True,
    )
    random_search.fit(X, y)

    best_params = random_search.best_params_
    best_cv_auc = random_search.best_score_

    print(f"\n  Best parameters found: {best_params}")
    print(f"  Best CV AUC:           {best_cv_auc:.4f}")

    cv_results = pd.DataFrame(random_search.cv_results_)
    print(f"\n  Top 5 combinations found:")
    top5 = cv_results.nlargest(5, "mean_test_score")[
        ["param_max_depth", "param_min_samples_leaf",
         "param_min_samples_split", "param_n_estimators",
         "mean_test_score", "mean_train_score"]
    ]
    for _, row in top5.iterrows():
        gap = row["mean_train_score"] - row["mean_test_score"]
        print(
            f"    depth={int(row['param_max_depth']):2d}  "
            f"leaf={int(row['param_min_samples_leaf']):2d}  "
            f"split={int(row['param_min_samples_split']):2d}  "
            f"trees={int(row['param_n_estimators']):3d}  "
            f"CV-AUC={row['mean_test_score']:.4f}  "
            f"gap={gap:.4f}"
        )

# Retrain the model
    final_clf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        **best_params,
    )
    final_clf.fit(X, y)
    print("  Done.")

    # Sanity check
    class_to_idx = {cls: i for i, cls in enumerate(final_clf.classes_)}
    pos_idx = class_to_idx.get(1, 0)
    y_prob_train = final_clf.predict_proba(X)[:, pos_idx]
    train_auc = roc_auc_score(y, y_prob_train)
    train_pr  = average_precision_score(y, y_prob_train)


    metadata = {
        "mode":        "full_data_inference",
        "description": (
            "Trained on ALL ChEMBL interactions with RandomizedSearchCV-optimised "
            "hyperparameters. Use for FDA drug repositioning predictions only. "
            "For performance metrics use rf_cv (AUC 0.7518)."
        ),
        "best_params":          best_params,
        "best_cv_auc":          float(best_cv_auc),
        "cv_folds":             args.cv_folds,
        "n_iter":               args.n_iter,
        "cv_strategy":          "RandomizedSearchCV + StratifiedKFold",
        "feature_cols":         feature_cols,
        "train_rows":           int(X.shape[0]),
        "n_features":           int(X.shape[1]),
        "train_roc_auc_sanity": float(train_auc),
        "train_target_ids": sorted(
            merged["target_id"].astype(str).unique().tolist()
        ),
    }

    model_path      = args.artifacts_dir / "rf_model.pkl"
    metadata_path   = args.artifacts_dir / "rf_metadata.json"
    metrics_path    = args.artifacts_dir / "metrics.json"
    cv_results_path = args.artifacts_dir / "cv_results.csv"

    with open(model_path, "wb") as f:
        pickle.dump(final_clf, f)

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps({
        "mode":                 "full_data_inference",
        "train_rows":           int(X.shape[0]),
        "n_features":           int(X.shape[1]),
        "best_params":          best_params,
        "best_cv_auc":          float(best_cv_auc),
        "train_roc_auc_sanity": float(train_auc),
    }, indent=2), encoding="utf-8")
    cv_results.to_csv(cv_results_path, index=False)

    print(f"  [saved] {model_path}")
    print(f"  [saved] {metadata_path}")
    print(f"  [saved] {metrics_path}")
    print(f"  [saved] {cv_results_path}")

    print("  INFERENCE MODEL READY")
    print(f"  Best params: {best_params}")
    print(f"  Best CV AUC: {best_cv_auc:.4f}")
    print(f"  Saved to:    {args.artifacts_dir}")


if __name__ == "__main__":
    main()