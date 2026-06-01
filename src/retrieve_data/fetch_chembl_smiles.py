"""
How this script works:
1. Data Loading: Reads the master interactions CSV to extract a unique list of all
   molecule IDs (ChEMBL IDs) involved in the dataset.
2. Batching: Divides the list of IDs into smaller batches to respect the ChEMBL
   REST API rate limits and prevent connection timeouts.
3. API Querying: Requests the structural properties for each batch of molecules
   from the ChEMBL database.
4. SMILES Extraction: Parses the JSON response to extract the 1D 'canonical_smiles'
   string representation for each molecule.
5. Export: Saves the mapped ChEMBL IDs and their corresponding SMILES strings to
   a new CSV file for downstream embedding generation.
"""

import pandas as pd
from pathlib import Path
import time
from chembl_webresource_client.new_client import new_client


def fetch_smiles_in_batches(molecule_ids, batch_size=50):
    smiles_dict = {}

    for i in range(0, len(molecule_ids), batch_size):
        batch_ids = molecule_ids[i: i + batch_size]

        try:
            mols = new_client.molecule.filter(molecule_chembl_id__in=batch_ids).only(
                ['molecule_chembl_id', 'molecule_structures']
            )
            for m in mols:
                m_id = m.get('molecule_chembl_id')
                structs = m.get('molecule_structures') or {}
                smiles = structs.get('canonical_smiles')

                if smiles:
                    smiles_dict[m_id] = smiles

        except Exception:
            # Pause briefly if the ChEMBL API throttles or drops the connection
            time.sleep(2)

    return pd.DataFrame(
        list(smiles_dict.items()),
        columns=['molecule_chembl_id', 'smiles']
    )


def main():
    project_root = Path(__file__).resolve().parents[2]
    interactions_path = project_root / "data" / "raw" / "chembl_pd_interactions.csv"
    output_path = project_root / "data" / "raw" / "pd_molecule_smiles.csv"

    df = pd.read_csv(interactions_path)
    unique_mols = df['molecule_chembl_id'].dropna().unique().tolist()

    smiles_df = fetch_smiles_in_batches(unique_mols)
    smiles_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path} ({len(smiles_df)} SMILES)")


if __name__ == "__main__":
    main()