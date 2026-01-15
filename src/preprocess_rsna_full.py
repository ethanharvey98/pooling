"""
Preprocess the full RSNA ICH dataset (not just the subset).
Slices are ordered from BOTTOM of head to TOP (ascending z_position).

Usage:
    python preprocess_rsna_full.py \
        --dicom_dir='/media/M2SSD/gen_models_data/stage_2_train' \
        --numpy_dir='/media/M2SSD/gen_models_data/RSNA_ICH_numpy_full' \
        --labels_csv='/media/M2SSD/gen_models_data/stage_2_train.csv'
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

import ct


def get_z_position(dcm):
    """Extract z-position from ImagePositionPatient."""
    if hasattr(dcm, 'ImagePositionPatient'):
        return float(dcm.ImagePositionPatient[2])
    return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess full RSNA ICH dataset")
    parser.add_argument("--dicom_dir", help="Directory containing DICOM files", type=str, required=True)
    parser.add_argument("--numpy_dir", help="Directory to save numpy dataset", type=str, required=True)
    parser.add_argument("--labels_csv", help="Path to RSNA labels CSV (stage_2_train.csv)", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.numpy_dir, exist_ok=True)

    # Load labels
    print("Loading labels...")
    labels_df = pd.read_csv(args.labels_csv)
    # Parse slice_id and label_type from ID column (format: ID_xxx_labeltype)
    labels_df['slice_id'] = labels_df['ID'].apply(lambda x: '_'.join(x.split('_')[:2]))
    labels_df['label_type'] = labels_df['ID'].apply(lambda x: x.split('_')[2])
    # Pivot to get one row per slice with all label types as columns
    labels_pivot = labels_df.pivot(index='slice_id', columns='label_type', values='Label')
    labels_pivot = labels_pivot.reset_index()

    # Get all DICOM files
    print("Scanning DICOM directory...")
    dcm_files = list(filter(lambda f: f.endswith('.dcm'), os.listdir(args.dicom_dir)))
    print(f"Found {len(dcm_files)} DICOM files")

    # Group slices by StudyInstanceUID (scan_id)
    print("Grouping slices by scan (StudyInstanceUID)...")
    scan_slices = defaultdict(list)

    for dcm_file in tqdm(dcm_files, desc="Reading DICOM headers"):
        dcm_path = os.path.join(args.dicom_dir, dcm_file)
        try:
            dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            scan_id = str(dcm.StudyInstanceUID)
            slice_id = dcm_file.replace('.dcm', '')
            z_pos = get_z_position(dcm)
            scan_slices[scan_id].append((slice_id, z_pos, dcm_path))
        except Exception as e:
            print(f"Error reading {dcm_file}: {e}")

    print(f"Found {len(scan_slices)} unique scans")

    # Process each scan
    print("Processing scans...")
    for scan_id, slices in tqdm(scan_slices.items(), desc="Processing scans"):
        # Sort by z_position ASCENDING (bottom to top)
        # Lower z = skull base (bottom), Higher z = vertex (top)
        slices_sorted = sorted(slices, key=lambda x: x[1])  # ascending z_position

        images = []
        for slice_id, z_pos, dcm_path in slices_sorted:
            try:
                dcm = pydicom.dcmread(dcm_path)
                image = dcm.pixel_array.astype(np.float32)
                image = image * float(getattr(dcm, 'RescaleSlope', 1)) + float(getattr(dcm, 'RescaleIntercept', 0))
                image = ct.strip_skull(image)
                images.append(image)
            except Exception as e:
                print(f"Error processing {dcm_path}: {e}")

        if images:
            np.savez(f"{args.numpy_dir}/{scan_id}.npz", np.array(images))

    print("Done!")
