from pathlib import Path
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_MDS_UPDRS = [
    "Tremor at rest, slight and infrequently present.",
    "Rigidity, moderate resistance through full range of passive movement.",
    "Bradykinesia with reduced amplitude and speed of repetitive movements.",
    "Postural instability affecting gait and balance in daily activities.",
]


def _load_descriptions(input_path=None, text_col="description"):
    if input_path is None:
        return pd.DataFrame({"phenotype_id": [f"P{i+1}" for i in range(len(DEFAULT_MDS_UPDRS))], "description": DEFAULT_MDS_UPDRS})

    path = Path(input_path)
    if path.suffix.lower() == ".txt":
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame({"phenotype_id": [f"P{i+1}" for i in range(len(lines))], "description": lines})

    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {path}")
    if "phenotype_id" not in df.columns:
        df = df.copy()
        df["phenotype_id"] = [f"P{i+1}" for i in range(len(df))]
    return df[["phenotype_id", text_col]].rename(columns={text_col: "description"})


def generate_phenotypic_embeddings(output_csv, input_path=None, text_col="description", max_features=512):
    df = _load_descriptions(input_path=input_path, text_col=text_col)
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df["description"].astype(str).tolist())

    emb_df = pd.DataFrame(matrix.toarray())
    emb_df.columns = [f"pheno_emb_{i}" for i in range(emb_df.shape[1])]
    out_df = pd.concat([df[["phenotype_id", "description"]].reset_index(drop=True), emb_df], axis=1)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} phenotype embeddings to {out_path}")
    return out_df


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate phenotype embeddings from MDS-UPDRS text.")
    parser.add_argument("--input_path", type=Path, default=None)
    parser.add_argument("--text_col", type=str, default="description")
    parser.add_argument("--max_features", type=int, default=512)
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=project_root / "data" / "processed" / "phenotype_embeddings.csv",
    )
    args = parser.parse_args()

    generate_phenotypic_embeddings(
        output_csv=args.output_csv,
        input_path=args.input_path,
        text_col=args.text_col,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
