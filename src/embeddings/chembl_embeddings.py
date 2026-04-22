from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Using the exact same model you used for FDA drugs so the math aligns!
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


def canonicalize_smiles(smiles):
    if not isinstance(smiles, str):
        return None
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


def load_chembl_data(path):
    df = pd.read_csv(path)

    # We only need the ID and the SMILES
    df = df[["molecule_chembl_id", "smiles"]].copy()

    original_n = len(df)

    # Standardize the chemistry
    df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)

    # Drop invalid SMILES
    valid_df = df[df["canonical_smiles"].notna()].copy()
    valid_df["smiles"] = valid_df["canonical_smiles"]
    valid_df = valid_df.drop(columns=["canonical_smiles"])

    # Ensure no duplicates
    valid_df = valid_df.drop_duplicates(subset=["molecule_chembl_id"]).reset_index(drop=True)

    print(f"[info] Original ChEMBL molecules: {original_n}")
    print(f"[info] Valid molecules remaining: {len(valid_df)}")

    return valid_df


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Loading ChemBERTa on {device}")

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

            # Mean pooling exactly like your previous script
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

            # L2 Normalization
            pooled = F.normalize(pooled, p=2, dim=1)

            all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)


def main():
    project_root = Path(__file__).resolve().parents[2]

    # 1. Point to the ChEMBL SMILES you extracted in Phase 1
    input_path = project_root / "data" / "raw" / "pd_molecule_smiles.csv"
    output_path = project_root / "data" / "processed" / "chembl_drug_embeddings.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[error] Cannot find {input_path}. Run the ChEMBL SMILES script first.")
        return

    print("[info] Loading ChEMBL molecules...")
    df = load_chembl_data(input_path)

    tokenizer, model, device = load_model()

    print("[info] Generating ChemBERTa embeddings...")
    embeddings = embed_smiles(df["smiles"].tolist(), tokenizer, model, device)

    # 2. Format the output vector columns
    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"drug_emb_{i}" for i in range(embeddings.shape[1])]
    )

    # 3. Stitch the IDs back to the vectors
    result = pd.concat([df, emb_df], axis=1)
    result.to_csv(output_path, index=False)

    print(f"[done] Saved ChEMBL embeddings to {output_path}")


if __name__ == "__main__":
    main()