from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def infer_target_id_from_filename(path_obj):
    stem = path_obj.stem
    return stem.split("_")[0] if "_" in stem else stem


def sequence_composition_embedding(sequence):
    sequence = str(sequence).upper()
    length = max(len(sequence), 1)
    counts = {aa: 0 for aa in AA_ORDER}
    for aa in sequence:
        if aa in counts:
            counts[aa] += 1

    comp = [counts[aa] / length for aa in AA_ORDER]
    # Add sequence length as a simple global descriptor.
    return np.array(comp + [float(length)], dtype=np.float32)


def generate_protein_embeddings(fasta_dir, output_csv):
    fasta_dir = Path(fasta_dir)
    fasta_files = sorted(list(fasta_dir.glob("*.fasta")) + list(fasta_dir.glob("*.fa")))

    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found in {fasta_dir}")

    rows = []
    for fasta_file in fasta_files:
        target_id = infer_target_id_from_filename(fasta_file)
        try:
            record = next(SeqIO.parse(str(fasta_file), "fasta"))
        except StopIteration:
            print(f"[warn] Empty FASTA file skipped: {fasta_file}")
            continue

        vec = sequence_composition_embedding(record.seq)
        row = {"target_id": target_id, "sequence_id": record.id}
        for i, val in enumerate(vec):
            row[f"target_emb_{i}"] = float(val)
        rows.append(row)

    out_df = pd.DataFrame(rows).drop_duplicates(subset=["target_id"]).reset_index(drop=True)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} protein embeddings to {out_path}")
    return out_df


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate protein embeddings from FASTA files.")
    parser.add_argument(
        "--fasta_dir",
        type=Path,
        default=project_root / "data" / "raw" / "protein_sequences",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=project_root / "data" / "processed" / "protein_embeddings.csv",
    )
    args = parser.parse_args()
    generate_protein_embeddings(args.fasta_dir, args.output_csv)


if __name__ == "__main__":
    main()
