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
    if "molecule_chembl_id" in out.columns and "drug_id" not in out.columns:
        rename_map["molecule_chembl_id"] = "drug_id"
    if "target_chembl_id" in out.columns and "target_id" not in out.columns:
        rename_map["target_chembl_id"] = "target_id"
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

    if any(c.startswith("pheno_emb_") for c in feature_cols):
        raise ValueError(
            "Model expects phenotype features (pheno_emb_*). "
            "This inference script builds drug-target pairs only."
        )

    drugs = _normalize_id_columns(pd.read_csv(args.drug_embeddings_csv))
    targets = _normalize_id_columns(pd.read_csv(args.protein_embeddings_csv))

    if args.target_ids_csv is not None:
        target_filter_df = pd.read_csv(args.target_ids_csv)
        if "target_id" not in target_filter_df.columns:
            raise ValueError("target_ids_csv must contain a 'target_id' column.")
        targets = targets[targets["target_id"].isin(target_filter_df["target_id"])].copy()

    if args.use_train_targets_only:
        train_targets = set(metadata.get("train_target_ids", []))
        targets = targets[targets["target_id"].isin(train_targets)].copy()

    if drugs.empty or targets.empty:
        raise ValueError("No drugs or targets available after filtering.")

    # Validate features before batch scoring.
    probe = (
        drugs.head(1).assign(_k=1)
        .merge(targets.head(1).assign(_k=1), on="_k", how="inner")
        .drop(columns=["_k"])
    )
    missing = [c for c in feature_cols if c not in probe.columns]
    if missing:
        raise ValueError(f"Missing features required by model: first 10 -> {missing[:10]}")

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

    print("\n--- Inference complete ---")
    print(f"[saved] {args.output_csv}")
    print(f"[info] rows saved: {len(scores)}")


if __name__ == "__main__":
    main()