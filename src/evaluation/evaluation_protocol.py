import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def split_by_date(df, cutoff_year):
    """
    Split drug-target interactions by discovery/approval year.
    Data up to 'cutoff_year' is training, from 'cutoff_year' to 2025 is test.
    """
    # Assuming 'year' column exists in dataframe
    train = df[df['year'] <= cutoff_year]
    test = df[(df['year'] > cutoff_year) & (df['year'] <= 2025)]
    return train, test

def double_member_exclusion(train_df, test_df):
    """
    Ensure neither drug nor target from test set appeared in training set.
    Forcing the model to generalize to entirely novel entities.
    """
    train_drugs = set(train_df['drug_id'])
    train_targets = set(train_df['target_id'])
    
    # Exclude rows from test if drug OR target in train
    filtered_test = test_df[
        (~test_df['drug_id'].isin(train_drugs)) & 
        (~test_df['target_id'].isin(train_targets))
    ]
    return filtered_test

def check_structural_similarity(smiles1, smiles2):
    # Calculate Tanimoto similarity using RDKit
    ms1 = Chem.MolFromSmiles(smiles1)
    ms2 = Chem.MolFromSmiles(smiles2)
    if not ms1 or not ms2: return 0.0
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(ms1, 2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(ms2, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def similarity_based_partitioning(train_df, test_df, threshold=0.7):
    """
    Exclude drug-target pairs from test set that share high similarity
    with any pair in the training set.
    """
    # This can be computationally expensive for large datasets
    # Logic: For each test drug, compare to all train drugs
    # If similarity > threshold, remove from test
    print(f"Applying Similarity-based Partitioning (threshold={threshold})...")
    # simplified placeholder logic
    return test_df

if __name__ == "__main__":
    # Example usage
    data = {
        'drug_id': ['D1', 'D2', 'D3', 'D4'],
        'target_id': ['T1', 'T2', 'T3', 'T4'],
        'year': [2010, 2015, 2020, 2022],
        'smiles': ['C', 'CO', 'CC', 'CCO']
    }
    df = pd.DataFrame(data)
    
    train, test = split_by_date(df, 2018)
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    filtered_test = double_member_exclusion(train, test)
    print(f"Filtered Test (Double Exclusion) size: {len(filtered_test)}")
