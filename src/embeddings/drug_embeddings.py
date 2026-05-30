"""
This module generates chemical embeddings for FDA-approved drugs using ChemBERTa.
It processes drug data from DrugBank, canonicalizes SMILES strings for consistency, and
produces dense vector embeddings for computational drug analysis tasks.

Key functionality:
  - Load FDA-approved drug data with DrugBank IDs and SMILES strings
  - Standardize column names and filter for required fields
  - Canonicalize SMILES to ensure valid chemistry and remove duplicates
  - Generate embeddings using the pre-trained ChemBERTa model
  - Apply mean pooling with attention mask for optimal representation
  - Normalize embeddings to unit length for similarity comparisons

Dependencies:
  - rdkit: SMILES canonicalization and validation
  - transformers: ChemBERTa model and tokenizer
  - torch: GPU acceleration support
  - pandas, numpy: Data handling

Input:
  FDA-approved drug data with columns: drugbank_id, name, smiles

Output:
  Saves embeddings to CSV with drug IDs, names, SMILES, and 512-dimensional
  embedding vectors (drug_emb_0 through drug_emb_511).
"""
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

from rdkit import Chem

def canonicalize_smiles(smiles):
    smiles = smiles.strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            pass
    return None


def load_data(path):
    df = pd.read_csv(path)

    df = df.rename(columns={
        "drugbank_id": "drug_id",
        "name": "drug_name"
    })

    df = df[["drug_id", "drug_name", "smiles"]].copy()
    df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)

    valid_df = df[df["canonical_smiles"].notna()].copy()

    valid_df["smiles"] = valid_df["canonical_smiles"]
    valid_df = valid_df.drop(columns=["canonical_smiles"])
    valid_df = valid_df.drop_duplicates(subset=["drug_id"]).reset_index(drop=True)

    return valid_df


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    return tokenizer, model, device


def embed_smiles(smiles_list, tokenizer, model, device, batch_size=32):
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]

            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            outputs = model(**tokens)
            hidden = outputs.last_hidden_state

            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

            pooled = F.normalize(pooled, p=2, dim=1)

            all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)


def main():
    project_root = Path(__file__).resolve().parents[2]  # Thesis_project
    input_path = project_root / "data" / "processed" / "fda_approved_drugs.csv"
    output_path = project_root / "data" / "processed" / "drug_embeddings.csv"

    df = load_data(input_path)
    print(f"[info] {len(df)} drugs loaded")

    tokenizer, model, device = load_model()

    print("[info] Generating embeddings")
    embeddings = embed_smiles(df["smiles"].tolist(), tokenizer, model, device)

    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"drug_emb_{i}" for i in range(embeddings.shape[1])]
    )

    result = pd.concat([df, emb_df], axis=1)
    result.to_csv(output_path, index=False)

    print(f"[done] Saved to {output_path}")

if __name__ == "__main__":
    main()