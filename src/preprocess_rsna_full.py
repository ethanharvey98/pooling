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
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing scans"):
        scan_id = row["Study ID"]
        paths = ast.literal_eval(row["paths"])

        images = []
        for dcm_path in paths:
            dcm = pydicom.dcmread(dcm_path)
            image = dcm.pixel_array.astype(np.float32)
            image = image * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
            image = ct.strip_skull(image)
            images.append(image)

        np.savez(f"{args.numpy_dir}/{scan_id}.npz", np.array(images))

    print("Done!")
