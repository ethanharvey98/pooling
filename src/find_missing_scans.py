"""
Find missing scans that weren't preprocessed and reprocess them.

Usage:
    python find_missing_scans.py \
        --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' \
        --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy'
"""
import argparse
import ast
import os

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

import ct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and reprocess missing RSNA scans")
    parser.add_argument("--labels_csv", help="Path to labels.csv", type=str, required=True)
    parser.add_argument("--numpy_dir", help="Directory with numpy dataset", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true", help="Only print missing scans, don't reprocess")
    args = parser.parse_args()

    print("Loading labels.csv...")
    labels_df = pd.read_csv(args.labels_csv)
    print(f"Total scans in labels.csv: {len(labels_df)}")

    # Find existing npz files
    existing = set(f.replace('.npz', '') for f in os.listdir(args.numpy_dir) if f.endswith('.npz'))
    print(f"Existing npz files: {len(existing)}")

    # Find missing scans
    missing_df = labels_df[~labels_df['Study ID'].isin(existing)]
    print(f"Missing scans: {len(missing_df)}")

    if args.dry_run:
        print("\nMissing scan IDs:")
        for scan_id in missing_df['Study ID']:
            print(f"  {scan_id}")
        exit(0)

    # Reprocess missing scans
    print(f"\nReprocessing {len(missing_df)} missing scans...")
    failed_scans = []
    for _, row in tqdm(missing_df.iterrows(), total=len(missing_df), desc="Processing"):
        scan_id = row["Study ID"]
        paths = ast.literal_eval(row["paths"])

        try:
            images = []
            for dcm_path in paths:
                dcm = pydicom.dcmread(dcm_path)
                image = dcm.pixel_array.astype(np.float32)
                image = image * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
                image = ct.strip_skull(image)
                images.append(image)

            volume = np.array(images)  # (S, H, W)
            volume = volume.transpose(1, 2, 0)  # (H, W, S)
            volume = volume[np.newaxis, ...]  # (1, H, W, S)
            np.savez(f"{args.numpy_dir}/{scan_id}.npz", volume)
        except Exception as e:
            failed_scans.append((scan_id, str(e)))

    print(f"\nDone! Successfully processed: {len(missing_df) - len(failed_scans)}")
    if failed_scans:
        print(f"Failed scans: {len(failed_scans)}")
        for scan_id, error in failed_scans:
            print(f"  {scan_id}: {error}")
