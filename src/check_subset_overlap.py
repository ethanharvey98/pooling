"""
Check if the torchmil/RSNA_ICH_MIL HuggingFace subset has patient-level
data leakage in its pre-defined train/test splits.

Usage:
    python check_subset_overlap.py \
        --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv'
"""
import argparse
from collections import Counter
import pandas as pd
from huggingface_hub import hf_hub_download


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_csv', required=True, help='Path to RSNA labels.csv (has Patient ID)')
    args = parser.parse_args()

    # 1. Download splits.csv from HuggingFace
    print("Downloading splits.csv from torchmil/RSNA_ICH_MIL...")
    hf_path = hf_hub_download(
        repo_id="torchmil/RSNA_ICH_MIL",
        filename="dataset/splits.csv",
        repo_type="dataset",
    )
    splits_df = pd.read_csv(hf_path)

    # splits.csv has repeated rows per slice; get unique scan-level splits
    scan_splits = splits_df.groupby('bag_name')['split'].first().reset_index()
    scan_splits.columns = ['Study ID', 'split']
    print(f"HuggingFace subset: {len(scan_splits)} scans")
    print(f"  train: {(scan_splits['split'] == 'train').sum()}")
    print(f"  test:  {(scan_splits['split'] == 'test').sum()}")

    # 2. Load labels.csv to get Patient ID -> Study ID mapping
    labels_df = pd.read_csv(args.labels_csv)
    patient_map = dict(zip(labels_df['Study ID'], labels_df['Patient ID']))
    print(f"\nFull RSNA dataset: {len(labels_df)} scans, {labels_df['Patient ID'].nunique()} patients")

    # 3. Map subset scans to patients
    scan_splits['Patient ID'] = scan_splits['Study ID'].map(patient_map)
    missing = scan_splits['Patient ID'].isna().sum()
    if missing:
        print(f"WARNING: {missing} scans not found in labels.csv")
        scan_splits = scan_splits.dropna(subset=['Patient ID'])

    print(f"Subset patients: {scan_splits['Patient ID'].nunique()}")

    # 4. Check for patient overlap
    train_scans = set(scan_splits[scan_splits['split'] == 'train']['Study ID'])
    test_scans = set(scan_splits[scan_splits['split'] == 'test']['Study ID'])

    train_patients = set(scan_splits[scan_splits['split'] == 'train']['Patient ID'])
    test_patients = set(scan_splits[scan_splits['split'] == 'test']['Patient ID'])

    overlap = train_patients & test_patients

    print(f"\nPatients in train: {len(train_patients)}")
    print(f"Patients in test:  {len(test_patients)}")
    print(f"Patient overlap:   {len(overlap)}")

    if overlap:
        affected = scan_splits[scan_splits['Patient ID'].isin(overlap)]
        total = len(scan_splits)
        print(f"\n!!! DATA LEAKAGE: {len(overlap)} patients appear in both train and test !!!")
        print(f"Affected scans: {len(affected)} / {total} ({100*len(affected)/total:.1f}%)")

        print(f"\nOverlapping patients:")
        for pid in sorted(overlap)[:20]:
            patient_rows = scan_splits[scan_splits['Patient ID'] == pid]
            in_train = (patient_rows['split'] == 'train').sum()
            in_test = (patient_rows['split'] == 'test').sum()
            print(f"  {pid}: train={in_train}, test={in_test} (total {len(patient_rows)} scans)")
        if len(overlap) > 20:
            print(f"  ... and {len(overlap) - 20} more")
    else:
        print("\nNo patient-level data leakage in HuggingFace splits.")

    # 5. How many patients in the subset have multiple scans?
    patient_counts = Counter(scan_splits['Patient ID'])
    multi = {p: c for p, c in patient_counts.items() if c > 1}
    print(f"\nSubset patients with 1 scan: {sum(1 for c in patient_counts.values() if c == 1)}, "
          f"with 2+ scans: {len(multi)}")
    if multi:
        for pid, count in sorted(multi.items(), key=lambda x: -x[1])[:10]:
            print(f"  {pid}: {count} scans")
