"""
How this script works mechanically:
1. Model Loading: It loads the pre-trained Random Forest model and its metadata
   (to know exactly which 1,168 or 1,248 embedding features it expects).
2. Data Loading: It loads the novel DrugBank/FDA drug embeddings and the 63
   Parkinson's target embeddings.
3. Cartesian Product (Combinations): To test every drug against every target, it
   creates a massive combination matrix.
4. Scoring: It feeds these combined embedding vectors into the trained Random Forest,
   extracts the predicted probability (predict_proba) that the interaction is active,
   and sorts them from highest to lowest.
5. Export: Saves the scored combinations to `fda_target_scores.csv`. This output
   file is what your prediction analysis script reads to filter out known drugs
   and find the Top 25 novel candidates!

Usage:
    python src/models_randomf/predict_rf.py
    python src/models_randomf/predict_rf.py --protein_embeddings_csv data/processed/drkg_target_embeddings.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import pickle

import numpy as np
import pandas as pd


def _normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _predict_chunk(clf, pairs: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = pairs[feature_cols].to_numpy(dtype=np.float32)
    class_to_idx = {cls: i for i, cls in enumerate(clf.classes_)}
    if 1 in class_to_idx:
        prob = clf.predict_proba(X)[:, class_to_idx[1]]
    else:
        prob = np.zeros(len(pairs), dtype=np.float32)

    cols = [c for c in ["drug_id", "drug_name", "target_id"] if c in pairs.columns]
    out = pairs[cols].copy()
    out["score"] = prob
    return out


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Score DrugBank/FDA candidates with trained RF.")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=project_root / "random_forest" / "rf_cv" / "rf_model.pkl",
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        default=project_root / "random_forest" / "rf_cv" / "rf_metadata.json",
    )
    parser.add_argument(
        "--drug_embeddings_csv",
        type=Path,
        default=project_root / "data" / "processed" / "drug_embeddings.csv",
    )
    parser.add_argument(
        "--protein_embeddings_csv",
        type=Path,
        default=project_root / "data" / "processed" / "protein_embeddings.csv",
    )
    parser.add_argument(
        "--target_ids_csv",
        type=Path,
        default=None,
        help="Optional CSV with a target_id column to restrict scoring.",
    )
    parser.add_argument(
        "--use_train_targets_only",
        action="store_true",
        help="Restrict targets to train_target_ids from metadata.",
    )
    parser.add_argument("--batch_size_drugs", type=int, default=256)
    parser.add_argument("--top_k", type=int)
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=project_root / "random_forest" / "rf_cv" / "fda_target_scores.csv",
    )
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        clf = pickle.load(f)
    metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    feature_cols = metadata["feature_cols"]

    drugs = _normalize_id_columns(pd.read_csv(args.drug_embeddings_csv))
    targets = _normalize_id_columns(pd.read_csv(args.protein_embeddings_csv))

    scored_parts = []
    targets_keyed = targets.assign(_k=1)

    for start in range(0, len(drugs), args.batch_size_drugs):
        dchunk = drugs.iloc[start:start + args.batch_size_drugs].copy()
        pairs = dchunk.assign(_k=1).merge(targets_keyed, on="_k", how="inner").drop(columns=["_k"])

        part = _predict_chunk(clf, pairs, feature_cols)
        scored_parts.append(part)

        print(f"[info] Scored drugs {start}..{start + len(dchunk) - 1} -> {len(part)} pairs")

    scores = pd.concat(scored_parts, ignore_index=True)
    scores = scores.sort_values("score", ascending=False).reset_index(drop=True)
    if args.top_k > 0:
        scores = scores.head(args.top_k)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_csv, index=False)

    print("\n Inference complete ")
    print(f"[saved] {args.output_csv}")

if __name__ == "__main__":
    main()