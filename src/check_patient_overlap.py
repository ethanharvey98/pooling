"""
Check if the RSNA ICH train/val/test splits have patient-level data leakage.

labels.csv already has Patient ID, so no DICOM reading needed.

Usage (full dataset):
    python check_patient_overlap.py \
        --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' \
        --seed=1001

Usage (subset - filters labels.csv to the 1149 HuggingFace scans):
    python check_patient_overlap.py \
        --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' \
        --subset_csv='/home/denny-loevlie/RSNA_Investigation/rsna_ich_subset_1149_complete.csv' \
        --seed=1001

Run all 3 seeds at once:
    for seed in 1001 2001 3001; do
        echo "===== SEED=$seed ====="
        python check_patient_overlap.py \
            --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' \
            --seed=$seed
        echo
        python check_patient_overlap.py \
            --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' \
            --subset_csv='/home/denny-loevlie/RSNA_Investigation/rsna_ich_subset_1149_complete.csv' \
            --seed=$seed
        echo
    done
"""
import argparse
import ast
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split


def check_overlap(train_ids, val_ids, test_ids, scan_to_patient):
    """Check for patient-level overlap across splits."""
    train_patients = set(scan_to_patient[sid] for sid in train_ids if sid in scan_to_patient)
    val_patients = set(scan_to_patient[sid] for sid in val_ids if sid in scan_to_patient)
    test_patients = set(scan_to_patient[sid] for sid in test_ids if sid in scan_to_patient)

    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients

    print(f"Unique patients — train: {len(train_patients)}, val: {len(val_patients)}, test: {len(test_patients)}, "
          f"total: {len(train_patients | val_patients | test_patients)}")

    print(f"Patient overlap — train∩val: {len(train_val_overlap)}, "
          f"train∩test: {len(train_test_overlap)}, val∩test: {len(val_test_overlap)}")

    all_overlap_patients = train_val_overlap | train_test_overlap | val_test_overlap
    if all_overlap_patients:
        print(f"\n!!! DATA LEAKAGE: {len(all_overlap_patients)} patients appear in multiple splits !!!")

        # Count affected scans
        affected_scans = [sid for sid, pid in scan_to_patient.items() if pid in all_overlap_patients]
        total_scans = len(train_ids) + len(val_ids) + len(test_ids)
        print(f"Affected scans: {len(affected_scans)} / {total_scans} ({100*len(affected_scans)/total_scans:.1f}%)")

        for name, overlap in [("train∩val", train_val_overlap),
                              ("train∩test", train_test_overlap),
                              ("val∩test", val_test_overlap)]:
            if overlap:
                print(f"\n  {name} ({len(overlap)} patients):")
                for pid in sorted(overlap)[:15]:
                    scans = [sid for sid, p in scan_to_patient.items() if p == pid]
                    in_splits = []
                    if any(s in train_ids for s in scans): in_splits.append(f"train={sum(s in train_ids for s in scans)}")
                    if any(s in val_ids for s in scans): in_splits.append(f"val={sum(s in val_ids for s in scans)}")
                    if any(s in test_ids for s in scans): in_splits.append(f"test={sum(s in test_ids for s in scans)}")
                    print(f"    {pid}: {', '.join(in_splits)} (total {len(scans)} scans)")
                if len(overlap) > 15:
                    print(f"    ... and {len(overlap) - 15} more")
    else:
        print("\nNo patient-level data leakage detected.")

    # Patients with multiple scans
    patient_counts = Counter(scan_to_patient.values())
    multi = {p: c for p, c in patient_counts.items() if c > 1}
    print(f"\nPatients with 1 scan: {sum(1 for c in patient_counts.values() if c == 1)}, "
          f"with 2+ scans: {len(multi)}")
    if multi:
        for pid, count in sorted(multi.items(), key=lambda x: -x[1])[:10]:
            print(f"  {pid}: {count} scans")
        if len(multi) > 10:
            print(f"  ... and {len(multi) - 10} more")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_csv', required=True, help='Path to RSNA labels.csv (has Patient ID)')
    parser.add_argument('--subset_csv', default=None, help='Path to subset CSV to filter scans (optional)')
    parser.add_argument('--seed', default=1001, type=int)
    args = parser.parse_args()

    # Load full labels.csv (has Patient ID, Study ID, Any, etc.)
    labels_df = pd.read_csv(args.labels_csv)
    print(f"Full dataset: {len(labels_df)} scans, {labels_df['Patient ID'].nunique()} patients")

    # Build scan -> patient mapping
    scan_to_patient = dict(zip(labels_df['Study ID'], labels_df['Patient ID']))

    if args.subset_csv:
        # Filter to subset scans
        subset_df = pd.read_csv(args.subset_csv)
        subset_scan_ids = set(subset_df['scan_id'].unique())
        labels_df = labels_df[labels_df['Study ID'].isin(subset_scan_ids)]
        print(f"Subset: {len(labels_df)} scans, {labels_df['Patient ID'].nunique()} patients")
        scan_col = 'Study ID'
    else:
        scan_col = 'Study ID'

    # Compute scan-level label (same as encode scripts)
    labels_df['scan_label'] = labels_df['Any'].apply(lambda x: 1 if any(ast.literal_eval(x)) else 0)

    # Reproduce the split from encode_rsna.py / encode_rsna_full.py
    ids = labels_df[scan_col]
    id_labels = labels_df['scan_label']
    train_and_val_ids, test_ids, train_and_val_labels, _ = train_test_split(
        ids, id_labels, test_size=1/6, random_state=args.seed, stratify=id_labels
    )
    train_ids, val_ids = train_test_split(
        train_and_val_ids, test_size=1/5, random_state=args.seed, stratify=train_and_val_labels
    )

    train_ids = set(train_ids)
    val_ids = set(val_ids)
    test_ids = set(test_ids)

    print(f"Split (seed={args.seed}): train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print()

    check_overlap(train_ids, val_ids, test_ids, scan_to_patient)
