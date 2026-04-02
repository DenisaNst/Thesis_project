from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    from src.evaluation import evaluation_protocol as eval_protocol
except ImportError:
    from evaluation import evaluation_protocol as eval_protocol  # type: ignore

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError:
    torch = None
    nn = object
    F = None
    GCNConv = None
    global_mean_pool = None


class ConcatenatedClassifier:
    def __init__(self, n_estimators=300, random_state=42):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

    def train(self, features, labels):
        self.rf.fit(features, labels)

    def predict(self, features):
        return self.rf.predict(features)

    def predict_proba(self, features):
        return self.rf.predict_proba(features)[:, 1]


def _prepare_matrix(interactions_df, drug_embeddings_df, protein_embeddings_df, phenotype_embeddings_df=None):
    merged = interactions_df.merge(drug_embeddings_df, on="drug_id", how="inner")
    merged = merged.merge(protein_embeddings_df, on="target_id", how="inner")

    if phenotype_embeddings_df is not None and not phenotype_embeddings_df.empty:
        pheno_cols = [c for c in phenotype_embeddings_df.columns if c.startswith("pheno_emb_")]
        if pheno_cols:
            pheno_mean = phenotype_embeddings_df[pheno_cols].mean(axis=0)
            for col in pheno_cols:
                merged[col] = float(pheno_mean[col])

    feature_cols = [c for c in merged.columns if c.startswith("drug_emb_") or c.startswith("target_emb_") or c.startswith("pheno_emb_")]
    X = merged[feature_cols].to_numpy(dtype=np.float32)
    y = merged["label"].to_numpy(dtype=int)
    return merged, X, y, feature_cols


def _safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


def train_baseline_model(interactions_csv, drug_embeddings_csv, protein_embeddings_csv, phenotype_embeddings_csv=None, cutoff_year=2019):
    interactions = eval_protocol.load_and_standardize_interactions(interactions_csv)
    train_df, test_df = eval_protocol.split_by_date(interactions, cutoff_year=cutoff_year)
    test_df = eval_protocol.double_member_exclusion(train_df, test_df)

    if test_df.empty:
        print("[warn] Test set is empty after double-member exclusion; falling back to pre-exclusion split.")
        train_df, test_df = eval_protocol.split_by_date(interactions, cutoff_year=cutoff_year)

    drug_emb = pd.read_csv(drug_embeddings_csv)
    protein_emb = pd.read_csv(protein_embeddings_csv)
    pheno_emb = pd.read_csv(phenotype_embeddings_csv) if phenotype_embeddings_csv and Path(phenotype_embeddings_csv).exists() else None

    train_m, X_train, y_train, feature_cols = _prepare_matrix(train_df, drug_emb, protein_emb, pheno_emb)
    test_m, X_test, y_test, _ = _prepare_matrix(test_df, drug_emb, protein_emb, pheno_emb)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("No overlapping IDs between interactions and embedding tables.")

    clf = ConcatenatedClassifier()
    clf.train(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    metrics = {
        "train_rows": int(len(train_m)),
        "test_rows": int(len(test_m)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(_safe_auc(y_test, y_prob)),
    }
    return clf, metrics, feature_cols


class ParkinsonGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        if GCNConv is None:
            raise ImportError("torch-geometric is required for ParkinsonGNN.")
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train baseline RF model on concatenated embeddings.")
    parser.add_argument("--interactions_csv", type=Path, default=project_root / "data" / "raw" / "chembl_pd_interactions_auto.csv")
    parser.add_argument("--drug_embeddings_csv", type=Path, default=project_root / "data" / "processed" / "drug_embeddings.csv")
    parser.add_argument("--protein_embeddings_csv", type=Path, default=project_root / "data" / "processed" / "protein_embeddings.csv")
    parser.add_argument("--phenotype_embeddings_csv", type=Path, default=project_root / "data" / "processed" / "phenotype_embeddings.csv")
    parser.add_argument("--cutoff_year", type=int, default=2019)
    args = parser.parse_args()

    clf, metrics, _ = train_baseline_model(
        interactions_csv=args.interactions_csv,
        drug_embeddings_csv=args.drug_embeddings_csv,
        protein_embeddings_csv=args.protein_embeddings_csv,
        phenotype_embeddings_csv=args.phenotype_embeddings_csv,
        cutoff_year=args.cutoff_year,
    )
    print("Baseline RF metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
