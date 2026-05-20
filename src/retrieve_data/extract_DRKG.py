"""
Maps the 63 ChEMBL PD targets to DRKG TransE embeddings.

Pipeline:
  1. Read FASTA filenames → extract ChEMBL ID + UniProt accession
  2. Convert UniProt → Entrez Gene ID via UniProt API
  3. Look up each Entrez ID in DRKG entities.tsv
  4. Extract the 400-dim TransE embedding from DRKG_TransE_l2_entity.npy
  5. Save as drkg_target_embeddings.csv — drop-in replacement
     for protein_embeddings.csv in your RF pipeline

Usage:
    python src/data/extract_drkg_target_embeddings.py

Requirements:
    - DRKG downloaded and unpacked at data/raw/drkg/
      (wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz)
    - Your FASTA files at data/raw/protein_sequences/
"""

from pathlib import Path
import time
import re
import requests
import numpy as np
import pandas as pd


PROJECT_ROOT  = Path(__file__).resolve().parents[2]
FASTA_DIR     = PROJECT_ROOT / "data" / "raw" / "protein_sequences"
DRKG_DIR      = PROJECT_ROOT / "data" / "raw" / "drkg"
OUTPUT_CSV    = PROJECT_ROOT / "data" / "processed" / "drkg_target_embeddings.csv"

DRKG_ENTITIES = DRKG_DIR / "embed" / "entities.tsv"
DRKG_EMB_NPY  = DRKG_DIR / "embed" / "DRKG_TransE_l2_entity.npy"

def extract_ids_from_fasta_dir(fasta_dir: Path) -> pd.DataFrame:
    rows = []
    fasta_files = sorted(
        list(fasta_dir.glob("*.fasta")) + list(fasta_dir.glob("*.fa"))
    )

    if not fasta_files:
        raise FileNotFoundError(
            f"No FASTA files found in {fasta_dir}. "
            "Make sure you ran fetch_protein_sequences.py first."
        )

    for f in fasta_files:
        stem = f.stem
        parts = stem.split("_")

        chembl_id = parts[0]

        # UniProt accession is the second part if it exists
        # UniProt accessions match pattern like P41543, Q9Y4I1, O60260
        uniprot_id = None
        if len(parts) > 1:
            candidate = parts[1]
            # Basic UniProt accession pattern check
            if re.match(r"^[A-Z][0-9][A-Z0-9]{3}[0-9]$", candidate) or \
               re.match(r"^[A-Z][0-9][A-Z0-9]{3}[0-9]-\d+$", candidate):
                uniprot_id = candidate.split("-")[0]  # strip isoform suffix
            else:
                # Try all parts for a UniProt-like accession
                for p in parts[1:]:
                    p_clean = p.split("-")[0]
                    if re.match(r"^[A-Z][0-9][A-Z0-9]{3}[0-9]$", p_clean):
                        uniprot_id = p_clean
                        break

        rows.append({
            "target_id":  chembl_id,
            "uniprot_id": uniprot_id,
            "fasta_file": f.name,
        })

    df = pd.DataFrame(rows)
    print(f"  Found {len(df)} FASTA files")
    print(f"  With UniProt ID: {df['uniprot_id'].notna().sum()}")
    print(f"  Missing UniProt ID: {df['uniprot_id'].isna().sum()}")

    if df['uniprot_id'].isna().any():
        missing = df[df['uniprot_id'].isna()]['fasta_file'].tolist()
        print(f"  [warn] Could not parse UniProt ID from: {missing[:5]}")

    return df

def uniprot_to_entrez(uniprot_ids: list, batch_size: int = 50) -> dict:
    mapping = {}
    ids_to_query = [uid for uid in uniprot_ids if uid is not None]

    print(f"  Querying UniProt ID mapping for {len(ids_to_query)} accessions")

    for i in range(0, len(ids_to_query), batch_size):
        batch = ids_to_query[i:i + batch_size]
        batch_str = ",".join(batch)

        # UniProt REST API for ID mapping
        url = "https://rest.uniprot.org/idmapping/run"
        payload = {
            "from": "UniProtKB_AC-ID",
            "to":   "GeneID",
            "ids":  batch_str,
        }

        try:
            r = requests.post(url, data=payload, timeout=30)
            r.raise_for_status()
            job_id = r.json()["jobId"]

            result_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
            for attempt in range(20):
                time.sleep(2)
                result = requests.get(result_url, timeout=30)
                if result.status_code == 200:
                    data = result.json()
                    if "results" in data:
                        for entry in data["results"]:
                            uniprot = entry["from"]
                            entrez  = entry["to"]
                            if uniprot not in mapping:
                                mapping[uniprot] = entrez
                        break
                elif result.status_code == 303:
                    time.sleep(2)
                    continue
            else:
                print(f"  [warn] Timeout waiting for UniProt batch {i//batch_size + 1}")

        except Exception as e:
            print(f"  [warn] UniProt API error for batch {i//batch_size + 1}: {e}")
        time.sleep(1)

    print(f"  Successfully mapped: {len(mapping)} / {len(ids_to_query)} UniProt IDs")
    return mapping

def load_drkg_entities(entities_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        entities_path,
        sep="\t",
        header=None,
        names=["entity_name", "entity_idx"],
    )
    print(f"  Total DRKG entities: {len(df):,}")
    gene_mask = df["entity_name"].str.startswith("Gene::")
    print(f"  Gene entities: {gene_mask.sum():,}")
    return df

def build_entrez_to_drkg_idx(entities_df: pd.DataFrame) -> dict:
    gene_rows = entities_df[
        entities_df["entity_name"].str.startswith("Gene::")
    ].copy()

    gene_rows["entrez_id"] = (
        gene_rows["entity_name"]
        .str.replace("Gene::", "", regex=False)
        .str.strip()
    )
    return dict(zip(gene_rows["entrez_id"], gene_rows["entity_idx"]))

def extract_transe_embeddings(
    target_df: pd.DataFrame,
    entrez_lookup: dict,
    entity_embeddings: np.ndarray,
) -> pd.DataFrame:
    n_dims = entity_embeddings.shape[1]
    emb_cols = [f"target_emb_{i}" for i in range(n_dims)]

    rows = []
    not_found = []

    for _, row in target_df.iterrows():
        chembl_id  = row["target_id"]
        entrez_id  = row.get("entrez_id")

        if entrez_id is None or str(entrez_id) == "nan":
            not_found.append(chembl_id)
            continue

        entrez_str = str(entrez_id).split(".")[0]

        drkg_idx = entrez_lookup.get(entrez_str)
        if drkg_idx is None:
            not_found.append(chembl_id)
            continue

        emb_vector = entity_embeddings[int(drkg_idx)]

        record = {
            "target_id":   chembl_id,
            "uniprot_id":  row.get("uniprot_id"),
            "entrez_id":   entrez_str,
            "drkg_entity": f"Gene::{entrez_str}",
            "drkg_idx":    int(drkg_idx),
        }
        for col, val in zip(emb_cols, emb_vector):
            record[col] = float(val)

        rows.append(record)

    result_df = pd.DataFrame(rows)

    print(f"\n  Successfully embedded: {len(result_df)} / {len(target_df)} targets")
    if not_found:
        print(f"  Not found in DRKG ({len(not_found)}): {not_found}")

    return result_df

def main():
    print("  Extracting DRKG TransE embeddings for PD targets")

    if not DRKG_EMB_NPY.exists():
        print(f"\n[error] DRKG embeddings not found at: {DRKG_EMB_NPY}")
        return

    print("\n Parsing FASTA filenames")
    target_df = extract_ids_from_fasta_dir(FASTA_DIR)

    print("\n Converting UniProt → Entrez Gene IDs")
    uniprot_ids = target_df["uniprot_id"].dropna().unique().tolist()
    uniprot_to_entrez_map = uniprot_to_entrez(uniprot_ids)

    target_df["entrez_id"] = target_df["uniprot_id"].map(uniprot_to_entrez_map)

    n_mapped = target_df["entrez_id"].notna().sum()
    print(f"  Targets with Entrez ID: {n_mapped} / {len(target_df)}")

    print("\n Loading DRKG entity index")
    entities_df   = load_drkg_entities(DRKG_ENTITIES)
    entrez_lookup = build_entrez_to_drkg_idx(entities_df)

    print(f"  Loading TransE embeddings from {DRKG_EMB_NPY.name}")
    entity_embeddings = np.load(str(DRKG_EMB_NPY))
    print(f"  Embedding matrix shape: {entity_embeddings.shape}")
    print(f"  ({entity_embeddings.shape[0]:,} entities × {entity_embeddings.shape[1]} dims)")

    print("\nExtracting TransE embeddings for your targets")
    result_df = extract_transe_embeddings(target_df, entrez_lookup, entity_embeddings)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)

    print(f"  Saved {len(result_df)} target embeddings to:")
    print(f"  {OUTPUT_CSV}")
    print(f"\n  Embedding dimensions: {entity_embeddings.shape[1]}")
    print(f"  (vs ESM2: 480 dims)")
    print(f"\n  To use in RF training, run:")
    print(f"  python src/models/train_rf.py \\")
    print(f"    --protein_embeddings_csv data/processed/drkg_target_embeddings.csv")

    print("\n  Target coverage summary:")
    merged = target_df.merge(
        result_df[["target_id"]].assign(found=True),
        on="target_id", how="left"
    )
    merged["found"] = merged["found"].fillna(False)
    for _, row in merged.iterrows():
        status = "✓" if row["found"] else "✗ NOT FOUND"
        print(f"    {row['target_id']}  UniProt:{row.get('uniprot_id', 'N/A')}  "
              f"Entrez:{row.get('entrez_id', 'N/A')}  {status}")


if __name__ == "__main__":
    main()