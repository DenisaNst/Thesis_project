import xml.etree.ElementTree as ET
import pandas as pd
import os
from tqdm import tqdm

def parse_drugbank(xml_path, output_path):
    print(f"Parsing {xml_path}...")
    
    # DrugBank uses a namespace
    ns = {'db': 'http://www.drugbank.ca'}
    
    drugs = []
    
    # We use iterparse to handle the large XML file
    context = ET.iterparse(xml_path, events=('end',))
    
    for event, elem in tqdm(context, desc="Processing drugs"):
        if elem.tag == '{http://www.drugbank.ca}drug':
            # Check if FDA approved
            groups = [g.text for g in elem.findall('db:groups/db:group', ns)]
            
            if 'approved' in groups:
                drug_dbid = elem.findtext('db:drugbank-id[@primary="true"]', namespaces=ns)
                name = elem.findtext('db:name', namespaces=ns)
                
                # Extract SMILES
                smiles = None
                properties = elem.findall('db:calculated-properties/db:property', ns)
                for prop in properties:
                    kind = prop.findtext('db:kind', namespaces=ns)
                    if kind == 'SMILES':
                        smiles = prop.findtext('db:value', namespaces=ns)
                        break
                
                if smiles:
                    drugs.append({
                        'drugbank_id': drug_dbid,
                        'name': name,
                        'smiles': smiles
                    })
            
            # Clear element to save memory
            elem.clear()
            
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.DataFrame(drugs)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(drugs)} approved drugs with SMILES to {output_path}")

if __name__ == "__main__":
    # Correct relative path for DrugBank XML
    xml_file = os.path.join("drugbank_all_full_database.xml", "full database.xml")
    output_file = os.path.join("data", "processed", "fda_approved_drugs.csv")
    
    if os.path.exists(xml_file):
        parse_drugbank(xml_file, output_file)
    else:
        # Check if it's in the project root directly (different structure)
        if os.path.exists("full database.xml"):
             parse_drugbank("full database.xml", output_file)
        else:
            print(f"File not found: {xml_file}")
