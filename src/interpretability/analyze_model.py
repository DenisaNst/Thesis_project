"""
How this script works:
1. Data Loading: It reads the ground-truth interaction pairs and two sets of
   high-dimensional embeddings (Drugs and Targets).
2. Matrix Construction: It loops through the pairs, finds the matching drug and
   target vectors, and concatenates them horizontally into a single feature row.
3. Training: It trains a Random Forest Classifier on this combined feature matrix.
4. Analysis: It applies three interpretability techniques to crack open the
   Random Forest and measure how much it relied on specific embedding dimensions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from rf_interpretability import (
    feature_importance_from_rf,
    grouped_feature_importance,
    permutation_importance_rf,
    explain_single_prediction_rf
)


def load_real_biological_dataset(target_type="esm2"):
    pairs_df = pd.read_csv("data/raw/chembl_pd_interactions.csv")
    drug_emb_df = pd.read_csv("data/processed/chembl_drug_embeddings.csv")

    drug_emb_df.set_index("molecule_chembl_id", inplace=True)
    drug_features = [col for col in drug_emb_df.columns if col.startswith("drug_emb_")]
    drug_emb_matrix = drug_emb_df[drug_features]

    if target_type.lower() == "esm2":
        target_emb_df = pd.read_csv("data/processed/protein_embeddings.csv")
    elif target_type.lower() == "drkg":
        target_emb_df = pd.read_csv("data/processed/drkg_target_embeddings.csv")

    target_emb_df.set_index("target_id", inplace=True)
    target_features = [col for col in target_emb_df.columns if col.startswith("target_emb_")]
    target_emb_matrix = target_emb_df[target_features]

    X_list = []
    y_list = []

    missing_count = 0
    for _, row in pairs_df.iterrows():
        d_id = row["molecule_chembl_id"]
        t_id = row["target_chembl_id"]
        label = row["label"]

        if d_id in drug_emb_matrix.index and t_id in target_emb_matrix.index:
            drug_vec = drug_emb_matrix.loc[d_id].values
            target_vec = target_emb_matrix.loc[t_id].values

            combined_vec = np.concatenate([drug_vec, target_vec])

            X_list.append(combined_vec)
            y_list.append(label)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"  Note: Skipped {missing_count} pairs.")

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)

    feature_names = drug_features + target_features

    print(f"Successfully loaded {X.shape[0]} interactions with {X.shape[1]} dimensional features.")
    return X, y, feature_names


def main():
    EXPERIMENT = "drkg"

    X, y, feature_names = load_real_biological_dataset(target_type=EXPERIMENT)

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y)

    print("\n Global Feature Importance (Top 10 Embedding Dimensions)")
    global_imp = feature_importance_from_rf(rf_model, feature_names=feature_names, top_k=10)
    for item in global_imp:
        print(f"  Rank {item['rank']}: {item['feature']} ({item['importance']:.5f})")

    print("\nGrouped Modality Importance")
    grouped_imp = grouped_feature_importance(rf_model, feature_names=feature_names)
    for modality, score in grouped_imp.items():
        if score > 0:
            print(f"  Modality '{modality.upper()}': {score * 100:.2f}% contribution")

    print("\nLocal Explanation (Ablation on Sample #1)")
    sample_pair = X[0]
    local_exp = explain_single_prediction_rf(rf_model, sample_pair, feature_names=feature_names, top_k=5)

    print(f"  Predicted Probability of Interaction: {local_exp['predicted_probability']:.4f}")
    print("  Top 5 driving embedding dimensions for this specific pair:")
    for item in local_exp['top_feature_contributions']:
        shift_dir = "INCREASED" if item['delta_proba'] > 0 else "DECREASED"
        print(f"    - {item['feature']} {shift_dir} probability by {abs(item['delta_proba']):.4f}")


if __name__ == "__main__":
    main()