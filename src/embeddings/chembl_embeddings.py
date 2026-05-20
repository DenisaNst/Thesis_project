from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

def canonicalize_smiles(smiles):
    """Standardize SMILES to canonical form."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            pass
    return None


def load_chembl_data(path):
    """Load SMILES and canonicalize."""
    df = pd.read_csv(path)
    df = df[["molecule_chembl_id", "smiles"]].copy()

    df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
    valid_df = df[df["canonical_smiles"].notna()].copy()
    valid_df["smiles"] = valid_df["canonical_smiles"]
    valid_df = valid_df.drop(columns=["canonical_smiles"])
    valid_df = valid_df.drop_duplicates(subset=["molecule_chembl_id"]).reset_index(drop=True)

    print(f"{len(valid_df)} / {len(df)} molecules valid")
    return valid_df


def load_model():
    """Load ChemBERTa tokenizer and model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model, device


def embed_smiles(smiles_list, tokenizer, model, device, batch_size=32):
    """Generate ChemBERTa embeddings for SMILES."""
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i: i + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**tokens)
            hidden = outputs.last_hidden_state

            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            pooled = F.normalize(pooled, p=2, dim=1)

            all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "raw" / "pd_molecule_smiles.csv"
    output_path = project_root / "data" / "processed" / "chembl_drug_embeddings.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_chembl_data(input_path)
    tokenizer, model, device = load_model()
    embeddings = embed_smiles(df["smiles"].tolist(), tokenizer, model, device)

    emb_df = pd.DataFrame(embeddings, columns=[f"drug_emb_{i}" for i in range(embeddings.shape[1])])
    result = pd.concat([df[["molecule_chembl_id", "smiles"]], emb_df], axis=1)
    result.to_csv(output_path, index=False)

    print(f"Saved: {output_path} ({len(result)} molecules)")


if __name__ == "__main__":
    main()