import pickle
import json
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from similarity_based_DRKG import load_data, compute_drug_similarity, compute_protein_similarity, prepare_feature_matrix


def main():
    OUT_DIR = Path("artifacts/rf_drkg_similarity_1_0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data")
    train_int, test_int, drug_emb, prot_emb = load_data()

    train_drugs = train_int["drug_id"].unique().tolist()
    test_drugs = test_int["drug_id"].unique().tolist()
    train_targets = train_int["target_id"].unique().tolist()
    test_targets = test_int["target_id"].unique().tolist()

    drug_max_sim = compute_drug_similarity(train_drugs, test_drugs, drug_emb)
    prot_max_sim = compute_protein_similarity(train_targets, test_targets, prot_emb)

    d_thresh = 1.0
    p_thresh = 1.0

    filtered_train = train_int[
        (train_int["drug_id"].map(drug_max_sim).fillna(0) < d_thresh) &
        (train_int["target_id"].map(prot_max_sim).fillna(0) < p_thresh)
        ].copy()

    filtered_test = test_int[
        (test_int["drug_id"].map(drug_max_sim).fillna(0) < d_thresh) &
        (test_int["target_id"].map(prot_max_sim).fillna(0) < p_thresh)
        ].copy()

    _, X_train, y_train, feature_cols = prepare_feature_matrix(filtered_train, drug_emb, prot_emb)

    # Train the Random Forest
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        min_samples_split=30, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Save Model
    with open(OUT_DIR / "rf_model.pkl", "wb") as f:
        pickle.dump(clf, f)

    # Save Test Set
    filtered_test.to_csv(OUT_DIR / "test_set.csv", index=False)

    # Save Metadata
    metadata = {"feature_cols": feature_cols}
    (OUT_DIR / "rf_metadata.json").write_text(json.dumps(metadata))


if __name__ == "__main__":
    main()