import xml.etree.ElementTree as ET
import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path

def parse_drugbank(xml_path, output_path):
    ns = {'db': 'http://www.drugbank.ca'}
    drugs = []

    context = ET.iterparse(xml_path, events=('end',))
    
    for event, elem in tqdm(context, desc="Processing drugs"):
        if elem.tag == '{http://www.drugbank.ca}drug':
            # Check if FDA approved
            groups = [g.text for g in elem.findall('db:groups/db:group', ns)]
            
            if 'approved' in groups:
                drug_dbid = elem.findtext('db:drugbank-id[@primary="true"]', namespaces=ns)
                name = elem.findtext('db:name', namespaces=ns)
                
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
            elem.clear()
        
    df = pd.DataFrame(drugs)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(drugs)} approved drugs with SMILES to {output_path}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    xml_file = project_root / "drugbank_all_full_database.xml" / "full database.xml"
    output_file = project_root / "retrieve_data" / "processed" / "fda_approved_drugs.csv"

    if xml_file.exists():
        parse_drugbank(str(xml_file), str(output_file))
    elif (project_root / "full database.xml").exists():
        parse_drugbank(str(project_root / "full database.xml"), str(output_file))
    else:
        print(f"File not found: {xml_file}")
