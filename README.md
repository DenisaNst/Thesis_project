# Parkinson's Drug Discovery Framework

This repository implements a multi-phase research project on drug discovery for Parkinson's disease using Transformers, Graph Neural Networks, and Scientific RAG.

## Project Structure

- `src/data/`: Scripts for data acquisition from DrugBank, ChEMBL, and NCBI.
- `src/embeddings/`: Methods for generating drug (Molecular Transformer), protein (ESM), and phenotypic (LLM) embeddings.
- `src/models/`: Implementation of Random Forest and Graph Neural Network classifiers.
- `src/evaluation/`: Evaluation protocol based on time-split and similarity-based partitioning.
- `src/interpretability/`: Tools for GNN saliency mapping and Scientific RAG via PubMed.

## Phases

### Phase 1: Data Acquisition
- **DrugBank**: FDA-approved drug SMILES strings.
- **NCBI**: Amino acid sequences for PD targets (LRRK2, SNCA, GBA).
- **ChEMBL**: Drug-target interactions for training and evaluation.

### Phase 2: Embedding Generation
- **Molecular Transformer (MT)**: Captures SMILES-based molecular features.
- **ESM (Evolutionary Scale Modeling)**: Captures protein evolutionary and structural features via mean pooling.
- **LLM (Llama-3.2/MedLlama)**: Captures clinical context of MDS-UPDRS symptom descriptions.

### Phase 3: Predictive Modeling
- **Random Forest Classifier**: Handles high-dimensional concatenated embeddings.
- **Graph Neural Network (GNN)**: Captures systems-level topology using DRKG relationships.

### Phase 4: Evaluation Protocol
- **Time-based Splitting**: Data split by discovery/approval date to prevent over-optimism.
- **Similarity-based Partitioning**: Ensures test sets differ structurally/sequence-wise from training sets.
- **Double-Member Exclusion**: Neither drug nor target in test set appears in training set.

### Phase 5: Interpretability
- **Saliency Maps**: Identifies explanatory subgraphs from GNN.
- **Scientific RAG**: Justifies predictions via automated PubMed literature retrieval.

## Usage
1. Parse DrugBank: `python src\retrieve_data\extract_drugbank_smiles.py`
2. Fetch interactions: `python src\retrieve_data\fetch_chembl_interactions.py`
3. Fetch protein sequences: `python src\retrieve_data\fetch_protein_sequences.py`
4. Run baseline pipeline (embeddings + RF model + feature importance): `python src\run_baseline.py`
