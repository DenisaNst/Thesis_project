"""
Standalone script to load a pre-trained GNN and score Saliency Candidates
without needing to retrain the model.
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_final.GNN_pd import PDHeteroGNN
from build_drkg import build_pd_drkg_graph, PRED_SRC, PRED_DST

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "gnn_random_initialize"
MODEL_PATH = ARTIFACT_DIR / "gnn_model.pt"
METADATA_PATH = ARTIFACT_DIR / "gnn_metadata.json"
SALIENCY_CANDIDATES = PROJECT_ROOT / "artifacts" / "gnn_v2" / "saliency_candidates_all.csv"


def main():
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # 1. Load the Graph
    # (Note: we use a fixed seed here so the random node initialization
    # for Diseases/Pathways matches the training run as closely as possible)
    np.random.seed(42)
    torch.manual_seed(42)
    data, node_to_idx, idx_to_node, _, _ = build_pd_drkg_graph()
    data = data.to(device)

    # 2. Load the Model Metadata (to get the exact hidden_channels used)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    params = metadata["best_params"]

    # 3. Initialize the Empty Model Architecture
    print("\nLoading pre-trained model weights...")
    model = PDHeteroGNN(
        metadata=data.metadata(),
        hidden_channels=params["hidden_channels"],
        out_channels=params["out_channels"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    # 4. Inject the saved weights into the model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("Model successfully loaded!")

    # 5. Load your Saliency Candidates
    saliency_rows = []
    sal_df = pd.read_csv(SALIENCY_CANDIDATES)
    for _, row in sal_df.iterrows():
        saliency_rows.append([int(row["drug_node_idx"]), int(row["target_node_idx"]), 1])
    saliency_edges = np.array(saliency_rows, dtype=np.int64)
    print(f"Loaded {len(saliency_edges)} candidates to score.")

    # 6. Score the Candidates
    print("\nGenerating prediction scores...")
    with torch.no_grad():
        # Encode the whole graph once
        z = model.encode(data.x_dict, data.edge_index_dict)

        # Calculate dot products for the candidate pairs
        scores = model.score_pairs(
            z,
            torch.tensor(saliency_edges[:, :2].T, dtype=torch.long, device=device),
            PRED_SRC, PRED_DST
        ).cpu().numpy()

    # Convert raw logits to probabilities (0.0 to 1.0)
    probs = torch.sigmoid(torch.tensor(scores)).numpy()

    # 7. Print Results and Save
    print("\n=== FINAL CANDIDATE SCORES ===")
    print(f"Total Candidates Evaluated: {len(probs)}")
    print(f"Mean Predicted Probability: {float(np.mean(probs)):.4f}")
    print(f"Min Probability:            {float(np.min(probs)):.4f}")
    print(f"Max Probability:            {float(np.max(probs)):.4f}")

    # (Optional) Save the exact scores back to the CSV so you can see which drug got which score!
    sal_df["gnn_probability"] = probs
    out_csv = ARTIFACT_DIR / "saliency_candidates_scored.csv"
    sal_df.to_csv(out_csv, index=False)
    print(f"\nSaved detailed scores to: {out_csv}")


if __name__ == "__main__":
    main()