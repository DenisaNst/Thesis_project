from __future__ import annotations

from pathlib import Path
import argparse
import json
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

try:
    from src.evaluation import evaluation_protocol as eval_protocol
except ImportError:
    from evaluation import evaluation_protocol as eval_protocol  # type: ignore


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
    phenotype_embeddings_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    interactions_df = _normalize_id_columns(interactions_df)
    drug_embeddings_df = _normalize_id_columns(drug_embeddings_df)
    protein_embeddings_df = _normalize_id_columns(protein_embeddings_df)

    merged = interactions_df.merge(drug_embeddings_df, on="drug_id", how="inner")
    merged = merged.merge(protein_embeddings_df, on="target_id", how="inner")

    if phenotype_embeddings_df is not None:
        if "phenotype_id" in merged.columns and "phenotype_id" in phenotype_embeddings_df.columns:
            merged = merged.merge(phenotype_embeddings_df, on="phenotype_id", how="inner")
            print("[info] Triad mode: Drug + Target + Phenotype")
        else:
            print("[warn] phenotype_id missing; running Drug + Target baseline.")

    feature_cols = [
        c for c in merged.columns
        if c.startswith("drug_emb_") or c.startswith("target_emb_") or c.startswith("pheno_emb_")
    ]
    if not feature_cols:
        raise ValueError("No embedding features found after merges.")

    if "label" not in merged.columns:
        raise ValueError("Missing 'label' in merged training set.")

    X = merged[feature_cols].to_numpy(dtype=np.float32)
    y = merged["label"].to_numpy(dtype=int)
    return merged, X, y, feature_cols


def train_and_evaluate(
    interactions_csv: Path,
    drug_embeddings_csv: Path,
    protein_embeddings_csv: Path,
    phenotype_embeddings_csv: Path | None,
    cutoff_year: int,  # can keep for compatibility, unused in simple mode
    apply_double_member_exclusion: bool = True,  # unused in simple mode
) -> tuple[RandomForestClassifier, dict, list[str], pd.DataFrame]:
    interactions = eval_protocol.load_and_standardize_interactions(str(interactions_csv))

    # Simple mode: no time split, no DME
    train_df = interactions.copy()

    drug_emb = pd.read_csv(drug_embeddings_csv)
    protein_emb = pd.read_csv(protein_embeddings_csv)
    pheno_emb = None
    if phenotype_embeddings_csv is not None and phenotype_embeddings_csv.exists():
        pheno_emb = pd.read_csv(phenotype_embeddings_csv)

    train_m, X_train, y_train, feature_cols = _prepare_matrix(train_df, drug_emb, protein_emb, pheno_emb)

    if X_train.shape[0] == 0:
        raise ValueError("No overlap between interactions and embedding IDs.")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training set has one class after filtering.")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Training-set metrics (optimistic, but useful for sanity check)
    y_pred = clf.predict(X_train)
    class_to_idx = {cls: i for i, cls in enumerate(clf.classes_)}
    y_prob = clf.predict_proba(X_train)[:, class_to_idx[1]] if 1 in class_to_idx else np.zeros(len(y_train))

    metrics = {
        "train_rows": int(len(train_m)),
        "n_features": int(X_train.shape[1]),
        "positive_rate_train": float(np.mean(y_train)),
        "train_accuracy": float(accuracy_score(y_train, y_pred)),
        "train_f1": float(f1_score(y_train, y_pred, zero_division=0)),
        "train_roc_auc": _safe_auc(y_train, y_prob),
        "train_pr_auc": float(average_precision_score(y_train, y_prob)),
    }
    return clf, metrics, feature_cols, train_m


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Train RF in ChEMBL interaction space.")
    parser.add_argument(
        "--interactions_csv",
        type=Path,
        default=project_root / "data" / "raw" / "chembl_pd_interactions.csv",
    )
    parser.add_argument(
        "--drug_embeddings_csv",
        type=Path,
        default=project_root / "data" / "processed" / "chembl_drug_embeddings.csv",
    )
    parser.add_argument(
        "--protein_embeddings_csv",
        type=Path,
        default=project_root / "data" / "processed" / "protein_embeddings.csv",
    )
    parser.add_argument(
        "--phenotype_embeddings_csv",
        type=Path,
        default=project_root / "data" / "processed" / "phenotype_embeddings.csv",
    )
    parser.add_argument("--cutoff_year", type=int, default=2019)
    parser.add_argument("--no_dme", action="store_true", help="Disable double-member exclusion.")
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        default=project_root / "artifacts" / "rf",
    )
    args = parser.parse_args()

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    clf, metrics, feature_cols, train_m = train_and_evaluate(
        interactions_csv=args.interactions_csv,
        drug_embeddings_csv=args.drug_embeddings_csv,
        protein_embeddings_csv=args.protein_embeddings_csv,
        phenotype_embeddings_csv=args.phenotype_embeddings_csv,
        cutoff_year=args.cutoff_year,
        apply_double_member_exclusion=not args.no_dme,
    )

    model_path = args.artifacts_dir / "rf_model.pkl"
    metadata_path = args.artifacts_dir / "rf_metadata.json"
    metrics_path = args.artifacts_dir / "metrics.json"

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    metadata = {
        "feature_cols": feature_cols,
        "train_target_ids": sorted(train_m["target_id"].astype(str).unique().tolist()),
        "uses_phenotype_features": any(c.startswith("pheno_emb_") for c in feature_cols),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n--- RF training complete ---")
    print(f"[saved] model: {model_path}")
    print(f"[saved] metadata: {metadata_path}")
    print(f"[saved] metrics: {metrics_path}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"- {k}: {v:.4f}")
        else:
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
