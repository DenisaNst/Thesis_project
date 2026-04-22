from pathlib import Path
import pandas as pd
import requests
import time
from chembl_webresource_client.new_client import new_client


def get_target_ids(targets_csv):
    """
    Reads the target metadata CSV and filters for single proteins.
    Protein complexes are excluded because ESM embeddings require
    a single, continuous amino acid chain.
    """
    df = pd.read_csv(targets_csv)

    # Safeguard: Only keep SINGLE PROTEIN targets
    if "target_type" in df.columns:
        df = df[df["target_type"] == "SINGLE PROTEIN"]

    return sorted(df["target_chembl_id"].dropna().unique())


def get_uniprot_accessions(target_chembl_id):
    t = new_client.target.get(target_chembl_id)
    accessions = []
    for comp in t.get("target_components", []):
        acc = comp.get("accession")
        if acc:
            accessions.append(acc)
    return sorted(set(accessions))


def fetch_uniprot_fasta(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def main():
    project_root = Path(__file__).resolve().parents[2]

    targets_csv = project_root / "data" / "raw" / "pd_targets_metadata.csv"
    out_dir = project_root / "data" / "raw" / "protein_sequences"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check if the metadata file exists first
    if not targets_csv.exists():
        print(f"[error] Cannot find {targets_csv}. Please run the interactions script first.")
        return

    target_ids = get_target_ids(targets_csv)
    print(f"[info] Found {len(target_ids)} valid SINGLE PROTEIN targets in metadata.")

    for tid in target_ids:
        accessions = get_uniprot_accessions(tid)
        if not accessions:
            print(f"[skip] {tid}: no UniProt accession in ChEMBL target_components")
            continue

        # Usually 1 accession for SINGLE PROTEIN; keep first deterministically
        acc = accessions[0]
        out_file = out_dir / f"{tid}_{acc}.fasta"

        # 2. Skip if we already downloaded this sequence (Resume feature)
        if out_file.exists():
            print(f"[skip] {out_file.name} already exists.")
            continue

        try:
            fasta = fetch_uniprot_fasta(acc)
            out_file.write_text(fasta, encoding="utf-8")
            print(f"[ok] {tid} -> {acc} -> {out_file.name}")

            # 3. Be polite to the UniProt servers to avoid IP bans
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"[err] Network error for {tid} ({acc}): {e}")
        except Exception as e:
            print(f"[err] Unexpected error for {tid} ({acc}): {e}")


if __name__ == "__main__":
    main()