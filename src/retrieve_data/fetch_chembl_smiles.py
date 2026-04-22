import pandas as pd
from pathlib import Path
import time
from chembl_webresource_client.new_client import new_client


def fetch_smiles_in_batches(molecule_ids, batch_size=50):
    smiles_dict = {}
    total_batches = (len(molecule_ids) // batch_size) + 1

    print(f"\n[info] Fetching SMILES for {len(molecule_ids)} unique molecules in {total_batches} batches...")

    for i in range(0, len(molecule_ids), batch_size):
        batch_ids = molecule_ids[i: i + batch_size]
        print(f"  -> Processing batch {(i // batch_size) + 1}/{total_batches}")

        try:
            mols = new_client.molecule.filter(molecule_chembl_id__in=batch_ids).only(
                ['molecule_chembl_id', 'molecule_structures']
            )
            for m in mols:
                m_id = m.get('molecule_chembl_id')
                structs = m.get('molecule_structures')
                if structs and isinstance(structs, dict):
                    smiles = structs.get('canonical_smiles')
                    if smiles:
                        smiles_dict[m_id] = smiles
        except Exception as e:
            print(f"[error] Failed on batch {(i // batch_size) + 1}: {e}")
            time.sleep(2)  # Brief pause if API hiccups

    return pd.DataFrame(list(smiles_dict.items()), columns=['molecule_chembl_id', 'smiles'])


def main():
    project_root = Path(__file__).resolve().parents[2]
    interactions_path = project_root / "data" / "raw" / "chembl_pd_interactions.csv"
    output_path = project_root / "data" / "raw" / "pd_molecule_smiles.csv"

    # 1. Check if the interactions file exists first!
    if not interactions_path.exists():
        print(f"[error] Cannot find {interactions_path}. Please run the interactions script first.")
        return

    # 2. Load the unique ChEMBL molecule IDs
    df = pd.read_csv(interactions_path)
    unique_mols = df['molecule_chembl_id'].dropna().unique().tolist()

    print(f"[info] Found {len(unique_mols)} unique molecules to fetch.")

    # 3. Fetch the SMILES
    smiles_df = fetch_smiles_in_batches(unique_mols)

    # 4. Save everything
    smiles_df.to_csv(output_path, index=False)

    print(f"[done] Saved SMILES dataset: {output_path}")
    print(f"[done] Total SMILES retrieved: {len(smiles_df)}")


if __name__ == "__main__":
    main()