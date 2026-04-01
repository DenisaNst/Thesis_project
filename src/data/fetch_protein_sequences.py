from Bio import Entrez, SeqIO
import os

# Set your email for NCBI API
Entrez.email = os.getenv("NCBI_EMAIL", "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi")

def fetch_protein_sequence(target_name, accession_id):
    print(f"Fetching sequence for {target_name} ({accession_id})...")
    try:
        handle = Entrez.efetch(db="protein", id=accession_id, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        handle.close()
        return record
    except Exception as e:
        print(f"Error fetching {target_name}: {e}")
        return None

def main():
    targets = {
        "LRRK2": "NP_940980.3",
        "SNCA": "NP_000336.1",
        "GBA": "NP_000148.2"
    }
    
    output_dir = os.path.join("data", "raw", "protein_sequences")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    for name, acc in targets.items():
        record = fetch_protein_sequence(name, acc)
        if record:
            output_file = os.path.join(output_dir, f"{name}.fasta")
            SeqIO.write(record, output_file, "fasta")
            print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
