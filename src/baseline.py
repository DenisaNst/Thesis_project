import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
from pathlib import Path


def generate_morgan_fingerprint(smiles, radius=2, n_bits=768):
    """
    Converts a SMILES string into a traditional 2D Morgan Fingerprint.
    We set n_bits=768 to perfectly match the size of your ChemBERTa vectors,
    ensuring a 100% fair mathematical comparison for the Random Forest!
    """
    try:
        # Convert string to RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)  # Return empty vector if SMILES is invalid

        # Generate the fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

        # Convert to a numpy array of 1s and 0s
        arr = np.zeros((1,), dtype=int)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    except Exception as e:
        print(f"[warn] Failed to parse SMILES: {smiles} - {e}")
        return np.zeros(n_bits)


def main():
    parser = argparse.ArgumentParser(description="Generate Morgan Fingerprint Baseline.")
    parser.add_argument("--smiles_csv", type=Path, required=True,
                        help="Path to CSV containing 'molecule_chembl_id' and 'smiles'")
    parser.add_argument("--output_csv", type=Path, required=True,
                        help="Where to save the baseline embeddings")
    args = parser.parse_args()

    print(f"[info] Loading SMILES from {args.smiles_csv}...")
    df = pd.read_csv(args.smiles_csv)

    if 'smiles' not in df.columns or 'molecule_chembl_id' not in df.columns:
        raise ValueError("Input CSV must contain 'molecule_chembl_id' and 'smiles' columns.")

    print(f"[info] Generating Morgan Fingerprints (Radius 2, 768 bits)...")

    # Apply the function to every SMILES string
    fingerprints = df['smiles'].apply(generate_morgan_fingerprint)

    # Convert the list of arrays into a DataFrame of individual columns
    fp_df = pd.DataFrame(fingerprints.tolist(), index=df.index)

    # Rename columns to match your existing prepare_matrix logic (drug_emb_0, drug_emb_1, etc.)
    fp_df.columns = [f"drug_emb_{i}" for i in range(fp_df.shape[1])]

    # Add the ID column back so it can merge with ChEMBL later
    fp_df['molecule_chembl_id'] = df['molecule_chembl_id']

    # Reorder columns to put ID first
    cols = ['molecule_chembl_id'] + [c for c in fp_df.columns if c != 'molecule_chembl_id']
    fp_df = fp_df[cols]

    print(f"[info] Saving Baseline Embeddings to {args.output_csv}...")
    fp_df.to_csv(args.output_csv, index=False)
    print("[info] Done! You can now feed this CSV into your Random Forest script.")


if __name__ == "__main__":
    main()