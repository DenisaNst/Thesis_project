from pathlib import Path
import pandas as pd
import requests
import time
from chembl_webresource_client.new_client import new_client


def get_target_ids(targets_csv):
    df = pd.read_csv(targets_csv)
    if "target_type" in df.columns:
        df = df[df["target_type"] == "SINGLE PROTEIN"]
    return sorted(df["target_chembl_id"].dropna().unique())


def get_uniprot_accessions(target_chembl_id):
    t = new_client.target.get(target_chembl_id)
    accessions = [
        comp.get("accession")
        for comp in t.get("target_components", [])
        if comp.get("accession")
    ]
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

    target_ids = get_target_ids(targets_csv)
    print(f"Processing {len(target_ids)} targets")

    for tid in target_ids:
        accessions = get_uniprot_accessions(tid)
        if not accessions:
            continue
        acc = accessions[0]
        out_file = out_dir / f"{tid}_{acc}.fasta"

        if out_file.exists():
            continue
        try:
            fasta = fetch_uniprot_fasta(acc)
            out_file.write_text(fasta, encoding="utf-8")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error for {tid}: {e}")
    print(f"Sequences saved to {out_dir}")


if __name__ == "__main__":
    main()