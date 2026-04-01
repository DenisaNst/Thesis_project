from chembl_webresource_client.new_client import new_client
import pandas as pd
import os

def fetch_drug_target_interactions(target_chembl_ids):
    interaction_list = []
    
    for target_id in target_chembl_ids:
        print(f"Fetching interactions for target: {target_id}...")
        activities = new_client.activity.filter(target_chembl_id=target_id, standard_type="IC50")
        
        for activity in activities:
            interaction_list.append({
                'molecule_chembl_id': activity['molecule_chembl_id'],
                'target_chembl_id': activity['target_chembl_id'],
                'standard_value': activity['standard_value'],
                'standard_units': activity['standard_units'],
                'standard_type': activity['standard_type']
            })
            
    return pd.DataFrame(interaction_list)

def main():
    # ChEMBL IDs for PD targets (example)
    # LRRK2: CHEMBL5141
    # SNCA: CHEMBL4633 (Alpha-synuclein)
    # GBA: CHEMBL4303 (Glucosylceramidase)
    pd_targets = ["CHEMBL5141", "CHEMBL4633", "CHEMBL4303"]
    
    df = fetch_drug_target_interactions(pd_targets)
    
    # Ensure directory exists
    output_dir = os.path.join("data", "raw")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "chembl_pd_interactions.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} interactions to {output_path}")

if __name__ == "__main__":
    main()
