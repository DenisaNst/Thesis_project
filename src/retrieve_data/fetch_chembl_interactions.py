from chembl_webresource_client.new_client import new_client
import pandas as pd
from pathlib import Path
import time


def get_pd_molecules():
    """
    Retrieve molecules associated with Parkinson's disease indications.
    Uses efo_term text match; depending on ChEMBL version you may also try mesh_heading.
    """
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
                rows.append(
                    {
                        "molecule_chembl_id": mid,
                        "target_chembl_id": tid,
                        "mechanism_of_action": m.get("mechanism_of_action"),
                        "action_type": m.get("action_type"),
                    }
                )
    mech_df = pd.DataFrame(rows)
    target_ids = sorted(set(mech_df["target_chembl_id"])) if not mech_df.empty else []
    return target_ids, mech_df


def get_target_metadata(target_ids):
    rows = []
    for tid in target_ids:
        t = new_client.target.get(tid)
        rows.append(
            {
                "target_chembl_id": tid,
                "pref_name": t.get("pref_name"),
                "target_type": t.get("target_type"),
                "organism": t.get("organism"),
            }
        )
    return pd.DataFrame(rows)


def filter_targets(target_df):
    if target_df.empty:
        return []
    keep = target_df[
        (target_df["organism"] == "Homo sapiens")
        & (target_df["target_type"] == "SINGLE PROTEIN")
    ]
    return sorted(set(keep["target_chembl_id"]))


def _load_completed_targets(completed_targets_file):
    if not completed_targets_file.exists():
        return set()
    return set(
        x.strip()
        for x in completed_targets_file.read_text(encoding="utf-8").splitlines()
        if x.strip()
    )


def _mark_target_completed(completed_targets_file, target_id):
    with completed_targets_file.open("a", encoding="utf-8") as f:
        f.write(f"{target_id}\n")


def fetch_target_activities(
    target_ids,
    pd_mol_set,
    activities_checkpoint_csv,
    completed_targets_file,
    standard_types=("IC50", "Ki", "Kd"),
    max_rows_per_target=2000,
    max_retries=3,
):
    """
    Fetch and checkpoint activities per target.
    - Resumes from completed_targets_file
    - Writes per-target batches to activities_checkpoint_csv
    """
    completed = _load_completed_targets(completed_targets_file)
    all_rows = []

    columns = [
        "target_chembl_id",
        "molecule_chembl_id",
        "standard_type",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "assay_chembl_id",
        "document_chembl_id",
    ]

    for idx, tid in enumerate(target_ids, start=1):
        if tid in completed:
            print(f"[{idx}/{len(target_ids)}] Skipping completed target: {tid}")
            continue

        print(f"[{idx}/{len(target_ids)}] Fetching activities for target: {tid}")
        target_rows = []
        success = False

        for attempt in range(1, max_retries + 1):
            try:
                acts = (
                    new_client.activity.filter(
                        target_chembl_id=tid,
                        standard_type__in=list(standard_types),
                        assay_type="B",               # binding assays
                        standard_relation="=",        # exact values
                        pchembl_value__isnull=False,  # normalized potency available
                    )
                    .only(columns)
                )[:max_rows_per_target]

                for a in acts:
                    mol_id = a.get("molecule_chembl_id")
                    if mol_id not in pd_mol_set:
                        continue
                    target_rows.append(
                        {
                            "target_chembl_id": a.get("target_chembl_id"),
                            "molecule_chembl_id": mol_id,
                            "standard_type": a.get("standard_type"),
                            "standard_value": a.get("standard_value"),
                            "standard_units": a.get("standard_units"),
                            "pchembl_value": a.get("pchembl_value"),
                            "assay_chembl_id": a.get("assay_chembl_id"),
                            "document_chembl_id": a.get("document_chembl_id"),
                        }
                    )

                success = True
                break

            except Exception as exc:
                wait_s = 2 ** (attempt - 1)
                print(f"  [retry {attempt}/{max_retries}] {tid} failed: {exc}")
                if attempt < max_retries:
                    print(f"  waiting {wait_s}s before retry...")
                    time.sleep(wait_s)

        if not success:
            print(f"  [skip] Could not fetch {tid} after {max_retries} retries.")
            continue

        # checkpoint write for this target
        if target_rows:
            target_df = pd.DataFrame(target_rows)
            write_header = not activities_checkpoint_csv.exists()
            target_df.to_csv(
                activities_checkpoint_csv,
                mode="a",
                index=False,
                header=write_header,
            )
            all_rows.extend(target_rows)

        _mark_target_completed(completed_targets_file, tid)
        print(f"  saved {len(target_rows)} rows for {tid}")

    # If current run fetched nothing but checkpoint exists, load from disk.
    if not all_rows and activities_checkpoint_csv.exists():
        return pd.read_csv(activities_checkpoint_csv)

    return pd.DataFrame(all_rows)


def main():
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    activities_checkpoint_csv = out_dir / "chembl_pd_interactions_auto_activities_checkpoint.csv"
    completed_targets_file = out_dir / "chembl_completed_targets.txt"
    final_out_file = out_dir / "chembl_pd_interactions_auto.csv"

    # 1) PD molecules from indication
    molecule_ids, indication_df = get_pd_molecules()
    print(f"PD-indicated molecules found: {len(molecule_ids)}")

    # 2) Targets from those molecules' mechanisms
    target_ids, mech_df = get_targets_from_molecules(molecule_ids)
    print(f"Targets from mechanisms (before filtering): {len(target_ids)}")

    # 3) Target metadata + filtering
    target_df = get_target_metadata(target_ids)

    filtered_target_ids = filter_targets(target_df)
    max_targets = 5
    final_target_ids = filtered_target_ids[:max_targets]

    print(f"Targets after filters (human + single protein): {len(final_target_ids)}")

    # 4) Activities with resume/checkpoint
    pd_mol_set = set(molecule_ids)
    act_df = fetch_target_activities(
        target_ids=final_target_ids,
        pd_mol_set=pd_mol_set,
        activities_checkpoint_csv=activities_checkpoint_csv,
        completed_targets_file=completed_targets_file,
        standard_types=("IC50", "Ki", "Kd"),
        max_rows_per_target=2000,
        max_retries=3,
    )

    if act_df.empty and activities_checkpoint_csv.exists():
        act_df = pd.read_csv(activities_checkpoint_csv)

    print(f"Activity rows fetched/loaded: {len(act_df)}")

    # 5) Final merge + save
    if not act_df.empty:
        combined = act_df.merge(target_df, on="target_chembl_id", how="left")
        combined = combined.drop_duplicates().reset_index(drop=True)
    else:
        combined = pd.DataFrame()

    combined.to_csv(final_out_file, index=False)
    print(f"Saved final file: {final_out_file}")


if __name__ == "__main__":
    main()
