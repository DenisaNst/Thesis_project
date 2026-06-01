"""
Steps:
1. Indication Mapping: Queries the ChEMBL API for any molecules explicitly indicated
   for Parkinson's Disease (using EFO terms).
2. Target Extraction: Finds the Mechanism of Action (MoA) for those PD molecules
   to identify the specific protein targets they interact with.
3. Target Filtering: Restricts the targets strictly to 'Homo sapiens' (human) and
   'SINGLE PROTEIN' to ensure structural consistency for the ESM2 embeddings later.
4. Bioactivity Fetching: Pulls all raw binding affinity data (IC50, Ki, Kd) for
   these specific targets across the entire ChEMBL database.
5. Temporal Tagging: Fetches the exact publication year of the document that reported
   each interaction. (This is critically important—it makes the Time-Slice evaluation
   for RQ2 possible).
6. Binarization: Converts continuous pChEMBL values into binary classes. A pChEMBL
   score >= 6.0 (equivalent to <= 1 µM) is labeled as an Active interaction (1), and
   anything lower is Inactive (0).
"""

from chembl_webresource_client.new_client import new_client
import pandas as pd
from pathlib import Path
import time

def get_pd_molecules():
    rows = list(new_client.drug_indication.filter(efo_term__icontains="parkinson"))
    mol_ids = sorted({r.get("molecule_chembl_id") for r in rows if r.get("molecule_chembl_id")})
    return mol_ids, pd.DataFrame(rows)


def get_targets_from_molecules(molecule_ids):
    rows = []
    for mid in molecule_ids:
        mechs = new_client.mechanism.filter(molecule_chembl_id=mid)
        for m in mechs:
            tid = m.get("target_chembl_id")
            if tid:
                rows.append({
                    "molecule_chembl_id": mid,
                    "target_chembl_id": tid,
                    "mechanism_of_action": m.get("mechanism_of_action"),
                    "action_type": m.get("action_type"),
                })
    mech_df = pd.DataFrame(rows)
    target_ids = sorted(set(mech_df["target_chembl_id"])) if not mech_df.empty else []
    return target_ids, mech_df


def get_target_metadata(target_ids):
    rows = []
    for tid in target_ids:
        t = new_client.target.get(tid)
        rows.append({
            "target_chembl_id": tid,
            "pref_name": t.get("pref_name"),
            "target_type": t.get("target_type"),
            "organism": t.get("organism"),
        })
    return pd.DataFrame(rows)


def filter_targets(target_df):
    if target_df.empty:
        return []
    keep = target_df[
        (target_df["organism"] == "Homo sapiens") &
        (target_df["target_type"] == "SINGLE PROTEIN")
        ]
    return sorted(set(keep["target_chembl_id"]))


def get_document_year(document_ids):
    doc_map = {}
    for doc_id in document_ids:
        try:
            doc = new_client.document.get(doc_id)
            doc_map[doc_id] = doc.get("year")
        except Exception:
            doc_map[doc_id] = None
    return doc_map


def fetch_target_activities(target_ids, standard_types=("IC50", "Ki", "Kd"), max_rows_per_target=5000, max_retries=3):
    rows = []
    columns = ["target_chembl_id", "molecule_chembl_id", "standard_type", "standard_value",
               "standard_units", "standard_relation", "pchembl_value", "assay_chembl_id", "document_chembl_id"]

    for idx, tid in enumerate(target_ids, start=1):
        for attempt in range(1, max_retries + 1):
            try:
                acts = (
                    new_client.activity.filter(
                        target_chembl_id=tid,
                        standard_type__in=list(standard_types),
                        assay_type="B",
                        standard_relation="=",
                        pchembl_value__isnull=False,
                    ).only(columns)
                )[:max_rows_per_target]

                for a in acts:
                    rows.append({
                        "target_chembl_id": a.get("target_chembl_id"),
                        "molecule_chembl_id": a.get("molecule_chembl_id"),
                        "standard_type": a.get("standard_type"),
                        "standard_value": a.get("standard_value"),
                        "standard_units": a.get("standard_units"),
                        "standard_relation": a.get("standard_relation"),
                        "pchembl_value": a.get("pchembl_value"),
                        "assay_chembl_id": a.get("assay_chembl_id"),
                        "document_chembl_id": a.get("document_chembl_id"),
                    })
                break
            except Exception as exc:
                wait_s = 2 ** (attempt - 1)
                print(f"  [retry {attempt}/{max_retries}] {tid} failed: {exc}")
                if attempt < max_retries:
                    time.sleep(wait_s)
    return pd.DataFrame(rows)

def add_interaction_label(df, positive_pchembl_threshold=6.0):
    df = df.copy()
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df["label"] = (df["pchembl_value"] >= positive_pchembl_threshold).astype(int)
    return df

def save_intermediates(out_dir, indication_df, mech_df, target_df, act_df):
    indication_df.to_csv(out_dir / "pd_indications.csv", index=False)
    mech_df.to_csv(out_dir / "pd_mechanisms.csv", index=False)
    target_df.to_csv(out_dir / "pd_targets_metadata.csv", index=False)
    act_df.to_csv(out_dir / "pd_target_activities_raw.csv", index=False)


def main():
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_out_file = out_dir / "chembl_pd_interactions.csv"

    molecule_ids, indication_df = get_pd_molecules()

    target_ids, mech_df = get_targets_from_molecules(molecule_ids)
    target_df = get_target_metadata(target_ids)
    filtered_target_ids = filter_targets(target_df)

    act_df = fetch_target_activities(filtered_target_ids)

    if not act_df.empty:
        doc_ids = act_df["document_chembl_id"].dropna().unique()
        doc_map = get_document_year(doc_ids)
        act_df["year"] = act_df["document_chembl_id"].map(doc_map)
        act_df = add_interaction_label(act_df, positive_pchembl_threshold=6.0)

        combined = act_df.merge(target_df, on="target_chembl_id", how="left")
        combined = combined.drop_duplicates().reset_index(drop=True)
    else:
        combined = pd.DataFrame()

    save_intermediates(out_dir, indication_df, mech_df, target_df, act_df)
    combined.to_csv(final_out_file, index=False)

    print(f"[done] Saved final interactions dataset: {final_out_file}")


if __name__ == "__main__":
    main()