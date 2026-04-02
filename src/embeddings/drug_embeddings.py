from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator


def canonicalize_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def fetch_smiles_from_chembl(molecule_ids):
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError as exc:
        raise ImportError(
            "Install chembl_webresource_client to fetch SMILES from ChEMBL IDs."
        ) from exc

    rows = []
    for mid in sorted(set(molecule_ids)):
        try:
            rec = new_client.molecule.get(mid)
            smiles = (rec.get("molecule_structures") or {}).get("canonical_smiles")
            rows.append({"drug_id": mid, "smiles": smiles})
        except Exception as exc:
            print(f"[warn] Could not fetch SMILES for {mid}: {exc}")
            rows.append({"drug_id": mid, "smiles": None})
    return pd.DataFrame(rows)


def morgan_embeddings(smiles_list, n_bits=2048, radius=2):
    # Prefer the newer generator API to avoid deprecation warnings.
    generator = None
    try:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    except Exception:
        generator = None

    vectors = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            vectors.append(np.zeros(n_bits, dtype=np.float32))
            continue
        if generator is not None:
            fp = generator.GetFingerprint(mol)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        vectors.append(arr)
    return np.vstack(vectors)


def molformer_embeddings(smiles_list, model_name="ibm/MoLFormer-XL-both-10pct", batch_size=32, max_length=256):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("MolFormer backend requires torch and transformers.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    batches = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]
            toks = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks)
            mask = toks["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            batches.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(batches)


def prepare_smiles_table(input_csv, id_col=None, smiles_col="smiles"):
    df = pd.read_csv(input_csv)

    if id_col is None:
        if "molecule_chembl_id" in df.columns:
            id_col = "molecule_chembl_id"
        elif "drugbank_id" in df.columns:
            id_col = "drugbank_id"
        elif "drug_id" in df.columns:
            id_col = "drug_id"
        else:
            raise ValueError("Could not infer an ID column. Set --id_col.")

    if smiles_col in df.columns:
        work = df[[id_col, smiles_col]].rename(columns={id_col: "drug_id", smiles_col: "smiles"})
    else:
        work = fetch_smiles_from_chembl(df[id_col].dropna().tolist())

    work["smiles"] = work["smiles"].apply(canonicalize_smiles)
    work = work.dropna(subset=["drug_id", "smiles"]).drop_duplicates(subset=["drug_id"]).reset_index(drop=True)
    return work


def generate_drug_embeddings(input_csv, output_csv, id_col=None, smiles_col="smiles", backend="molformer"):
    smiles_df = prepare_smiles_table(input_csv=input_csv, id_col=id_col, smiles_col=smiles_col)
    smiles = smiles_df["smiles"].tolist()

    used_backend = backend
    if backend == "molformer":
        try:
            vectors = molformer_embeddings(smiles)
        except Exception as exc:
            print(f"[warn] MolFormer failed ({exc}); falling back to Morgan fingerprints.")
            vectors = morgan_embeddings(smiles)
            used_backend = "morgan"
    else:
        vectors = morgan_embeddings(smiles)

    emb_df = pd.DataFrame(vectors)
    emb_df.columns = [f"drug_emb_{i}" for i in range(emb_df.shape[1])]
    out_df = pd.concat([smiles_df[["drug_id", "smiles"]], emb_df], axis=1)
    out_df["embedding_backend"] = used_backend

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} drug embeddings to {out_path}")
    return out_df


def main():
    project_root = Path(__file__).resolve().parents[2]
    default_input = project_root / "data" / "raw" / "chembl_pd_interactions_auto.csv"
    if not default_input.exists():
        default_input = project_root / "data" / "processed" / "fda_approved_drugs.csv"

    parser = argparse.ArgumentParser(description="Generate drug embeddings from SMILES.")
    parser.add_argument("--input_csv", type=Path, default=default_input)
    parser.add_argument("--output_csv", type=Path, default=project_root / "data" / "processed" / "drug_embeddings.csv")
    parser.add_argument("--id_col", type=str, default=None)
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--backend", type=str, choices=["morgan", "molformer"], default="molformer")
    args = parser.parse_args()

    generate_drug_embeddings(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        id_col=args.id_col,
        smiles_col=args.smiles_col,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
