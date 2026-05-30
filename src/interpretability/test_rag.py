import pandas as pd
from rag import ScientificRAG
from pathlib import Path


def simplify_target_name(full_name: str, max_words: int = 3) -> str:
    """
    Extract the first few words from a target name.
    E.g., "Mitogen-activated protein kinase kinase kinase 12" → "protein kinase"
    This helps PubMed find more relevant results.
    """
    if not full_name or pd.isna(full_name):
        return ""

    # Remove common suffixes that don't help search
    name = full_name.strip()
    for suffix in [" (human)", " isoform", " subunit", " complex"]:
        if name.endswith(suffix):
            name = name.replace(suffix, "")

    # Take first few meaningful words
    words = name.split()
    return " ".join(words[:max_words])


def main():
    project_root = Path(__file__).resolve().parents[2]

    # Load predictions - already has drug_name and target_name columns
    preds_file = project_root / "artifacts/gnn_v2/saliency_candidates_both.csv"
    try:
        top_preds = pd.read_csv(preds_file)
    except FileNotFoundError:
        print(f"File not found: {preds_file}")
        return

    # Output file for results
    output_file = project_root / "artifacts/gnn_v2/rag_evidence.csv"
    results = []

    rag = ScientificRAG(email="denisa.elena.nastasa@gmail.com")

    for idx, row in top_preds.head(20).iterrows():
        drug_id = row["drug_id"]
        drug_name = row.get("drug_name", drug_id)
        target_id = row["target_id"]
        target_name = row.get("target_name", target_id)
        score = row["score"]

        # Simplify long target names for better PubMed search
        search_target_name = simplify_target_name(target_name, max_words=4)

        print(f"\n{'=' * 70}")
        print(f"#{idx + 1}: {drug_name} ({drug_id}) → {target_name} ({target_id})")
        print(f"Score: {score:.3f}")
        print(f"Searching PubMed with simplified name: '{search_target_name}'")
        print(f"{'=' * 70}")

        try:
            articles = rag.query_pubmed(drug_name, search_target_name, top_k=5)

            if articles:
                justification = rag.generate_justification(
                    drug_name, search_target_name, articles, max_items=3
                )
                print(justification)
                evidence_found = "Yes"
                num_articles = len(articles)
            else:
                print(f"No PubMed evidence found for '{drug_name}' + '{search_target_name}'")
                evidence_found = "No"
                num_articles = 0

            # Store result
            results.append({
                "rank": idx + 1,
                "drug_id": drug_id,
                "drug_name": drug_name,
                "target_id": target_id,
                "target_name": target_name,
                "gnn_score": score,
                "evidence_found": evidence_found,
                "num_articles": num_articles,
            })

        except Exception as e:
            print(f"[Error retrieving PubMed data]: {e}")
            results.append({
                "rank": idx + 1,
                "drug_id": drug_id,
                "drug_name": drug_name,
                "target_id": target_id,
                "target_name": target_name,
                "gnn_score": score,
                "evidence_found": "Error",
                "num_articles": 0,
            })

    # Save summary to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"\n[saved] Evidence summary → {output_file}")


if __name__ == "__main__":
    main()