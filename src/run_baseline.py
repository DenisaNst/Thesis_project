from pathlib import Path
import argparse

from embeddings.drug_embeddings import generate_drug_embeddings
from embeddings.protein_embeddings import generate_protein_embeddings
from embeddings.phenotypic_embeddings import generate_phenotypic_embeddings
from models.predictive_models import train_baseline_model
from interpretability.interpretability_tools import feature_importance_from_rf


def main():
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Run end-to-end baseline pipeline.")
    parser.add_argument(
        "--interactions_csv",
        type=Path,
        default=project_root / "data" / "raw" / "chembl_pd_interactions_auto.csv",
    )
    parser.add_argument(
        "--protein_fasta_dir",
        type=Path,
        default=project_root / "data" / "raw" / "protein_sequences",
    )
    parser.add_argument(
        "--drug_backend",
        type=str,
        choices=["morgan", "molformer"],
        default="molformer",
    )
    parser.add_argument("--cutoff_year", type=int, default=2019)
    args = parser.parse_args()

    if not args.interactions_csv.exists():
        raise FileNotFoundError(
            f"Missing interactions file: {args.interactions_csv}. Run src/retrieve_data/fetch_chembl_interactions.py first."
        )

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    drug_csv = processed_dir / "drug_embeddings.csv"
    protein_csv = processed_dir / "protein_embeddings.csv"
    pheno_csv = processed_dir / "phenotype_embeddings.csv"

    generate_drug_embeddings(
        input_csv=args.interactions_csv,
        output_csv=drug_csv,
        id_col="molecule_chembl_id",
        smiles_col="smiles",
        backend=args.drug_backend,
    )
    generate_protein_embeddings(args.protein_fasta_dir, protein_csv)
    generate_phenotypic_embeddings(output_csv=pheno_csv)

    clf, metrics, feature_names = train_baseline_model(
        interactions_csv=args.interactions_csv,
        drug_embeddings_csv=drug_csv,
        protein_embeddings_csv=protein_csv,
        phenotype_embeddings_csv=pheno_csv,
        cutoff_year=args.cutoff_year,
    )

    print("\nBaseline model metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")

    top_features = feature_importance_from_rf(clf.rf, feature_names, top_k=15)
    print("\nTop 15 feature importances:")
    for item in top_features:
        print(f"- {item['feature']}: {item['importance']:.6f}")


if __name__ == "__main__":
    main()

