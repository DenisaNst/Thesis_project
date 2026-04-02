from pathlib import Path
import pandas as pd
import requests
from chembl_webresource_client.new_client import new_client

def get_target_ids(interactions_csv):
    df = pd.read_csv(interactions_csv)
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
    interactions_csv = project_root / "data" / "raw" / "chembl_pd_interactions_auto.csv"
    out_dir = project_root / "data" / "raw" / "protein_sequences"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_ids = get_target_ids(interactions_csv)

    for tid in target_ids:
        accessions = get_uniprot_accessions(tid)
        if not accessions:
            print(f"[skip] {tid}: no UniProt accession in ChEMBL target_components")
            continue

        # usually 1 accession for SINGLE PROTEIN; keep first deterministically
        acc = accessions[0]
        try:
            fasta = fetch_uniprot_fasta(acc)
            out_file = out_dir / f"{tid}_{acc}.fasta"
            out_file.write_text(fasta, encoding="utf-8")
            print(f"[ok] {tid} -> {acc} -> {out_file.name}")
        except Exception as e:
            print(f"[err] {tid} ({acc}): {e}")

if __name__ == "__main__":
    main()
