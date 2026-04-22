from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO


def infer_target_id_from_filename(path_obj: Path) -> str:
    stem = path_obj.stem
    return stem.split("_")[0] if "_" in stem else stem


def load_fasta_sequences(fasta_dir: Path) -> pd.DataFrame:
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

        sequence = str(record.seq).strip().upper().replace(" ", "")
        if not sequence:
            print(f"[warn] Empty sequence skipped: {fasta_file}")
            continue

        rows.append(
            {
                "target_id": target_id,
                "sequence_id": record.id,
                "sequence": sequence,
                "source_file": fasta_file.name,
            }
        )

    if not rows:
        raise ValueError(f"No valid protein sequences could be read from {fasta_dir}")

    return pd.DataFrame(rows).drop_duplicates(subset=["target_id"]).reset_index(drop=True)


def esm_mean_pooled_embeddings(
    sequences,
    model_name: str = "facebook/esm2_t12_35M_UR50D",
    batch_size: int = 4,
    max_length: int = 1022,
    normalize: bool = True,
    hf_token: str | None = None,
):
    """
    Generate one embedding per protein sequence using an ESM model.
    Pooling strategy: mean pooling over residue token embeddings, excluding
    special tokens and padding.
    """
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "ESM embeddings require torch and transformers. "
            "Install them with: pip install torch transformers"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Loading ESM model '{model_name}' on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModel.from_pretrained(model_name, token=hf_token).to(device)
    model.eval()

    all_embeddings = []

    # ESM-2 tokenizer expects continuous strings, no spaces needed!
    seq_texts = sequences

    with torch.no_grad():
        for i in range(0, len(seq_texts), batch_size):
            batch_texts = seq_texts[i: i + batch_size]

            toks = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            toks = {k: v.to(device) for k, v in toks.items()}

            outputs = model(**toks)
            hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

            attention_mask = toks["attention_mask"].clone()

            # Exclude BOS/EOS when present by masking first and last valid positions.
            for row_idx in range(attention_mask.shape[0]):
                valid_positions = torch.where(attention_mask[row_idx] == 1)[0]
                if len(valid_positions) >= 2:
                    attention_mask[row_idx, valid_positions[0]] = 0
                    attention_mask[row_idx, valid_positions[-1]] = 0

            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)

            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

    return np.vstack(all_embeddings)


def generate_protein_embeddings(
    fasta_dir,
    output_csv,
    model_name: str = "facebook/esm2_t12_35M_UR50D",
    batch_size: int = 4,
    max_length: int = 1022,
    normalize: bool = True,
    hf_token: str | None = None,
):
    fasta_dir = Path(fasta_dir)
    seq_df = load_fasta_sequences(fasta_dir)

    vectors = esm_mean_pooled_embeddings(
        sequences=seq_df["sequence"].tolist(),
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        normalize=normalize,
        hf_token=hf_token,
    )

    emb_df = pd.DataFrame(vectors, columns=[f"target_emb_{i}" for i in range(vectors.shape[1])])

    out_df = pd.concat(
        [
            seq_df[["target_id", "sequence_id", "sequence", "source_file"]].reset_index(drop=True),
            emb_df.reset_index(drop=True),
        ],
        axis=1,
    )
    out_df["embedding_model"] = model_name
    out_df["embedding_pooling"] = "mean_over_residue_tokens_excluding_special_tokens"
    out_df["embedding_normalized"] = bool(normalize)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} protein embeddings to {out_path}")
    return out_df


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings from FASTA files using ESM mean pooling."
    )
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
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm2_t12_35M_UR50D",
        help="Hugging Face ESM model name, e.g. facebook/esm2_t12_35M_UR50D",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1022)
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable L2 normalization after mean pooling.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face token. You can also set HF_TOKEN in your environment.",
    )
    args = parser.parse_args()

    token = args.hf_token
    if token is None:
        import os
        token = os.environ.get("HF_TOKEN")

    generate_protein_embeddings(
        fasta_dir=args.fasta_dir,
        output_csv=args.output_csv,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        normalize=not args.no_normalize,
        hf_token=token,
    )


if __name__ == "__main__":
    main()
