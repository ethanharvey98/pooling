"""
Preprocess the full RSNA ICH dataset using labels.csv.
Slices are already ordered from BOTTOM of head to TOP (ascending z_position) in labels.csv.

Usage:
    python preprocess_rsna_full.py \
        --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' \
        --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' \
        --start=0 --stop=2175
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
    parser = argparse.ArgumentParser(description="Preprocess full RSNA ICH dataset")
    parser.add_argument("--labels_csv", help="Path to labels.csv", type=str, required=True)
    parser.add_argument("--numpy_dir", help="Directory to save numpy dataset", type=str, required=True)
    parser.add_argument("--start", default=0, help="Start index (default: 0)", type=int)
    parser.add_argument("--stop", default=None, help="End index (default: None, process all)", type=int)
    args = parser.parse_args()

    os.makedirs(args.numpy_dir, exist_ok=True)

    # Load labels.csv (slices already sorted by ascending z within each scan)
    print("Loading labels.csv...")
    labels_df = pd.read_csv(args.labels_csv)
    print(f"Found {len(labels_df)} scans")

    # Apply start/stop
    stop = args.stop if args.stop is not None else len(labels_df)
    labels_df = labels_df.iloc[args.start:stop]
    print(f"Processing scans {args.start} to {stop} ({len(labels_df)} scans)")

    # Process each scan
    errors = []
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing scans"):
        scan_id = row["Study ID"]
        paths = ast.literal_eval(row["paths"])

        images = []
        for dcm_path in paths:
            try:
                dcm = pydicom.dcmread(dcm_path)
                image = dcm.pixel_array.astype(np.float32)
                image = image * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
                image = ct.strip_skull(image)
                images.append(image)
            except Exception as e:
                errors.append(f"{scan_id}\t{dcm_path}\t{str(e)}")

        if images:
            # Stack to (S, H, W) then reshape to (C, H, W, S) where C=1
            volume = np.array(images)  # (S, H, W)
            volume = volume.transpose(1, 2, 0)  # (H, W, S)
            volume = volume[np.newaxis, ...]  # (1, H, W, S)
            np.savez(f"{args.numpy_dir}/{scan_id}.npz", volume)
        else:
            errors.append(f"{scan_id}\t-\tall slices failed")

    # Save errors to file
    if errors:
        with open("./errors_with_preprocessing_rsna.txt", "a") as f:
            for error in errors:
                f.write(error + "\n")
        print(f"Logged {len(errors)} errors to ./errors_with_preprocessing_rsna.txt")

    print("Done!")
