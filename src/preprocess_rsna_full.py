"""
Preprocess the full RSNA ICH dataset (not just the subset).
Slices are ordered from BOTTOM of head to TOP (ascending z_position).

Usage:
    python preprocess_rsna_full.py \
        --dicom_dir='/media/M2SSD/gen_models_data/stage_2_train' \
        --numpy_dir='/media/M2SSD/gen_models_data/RSNA_ICH_numpy_full' \
        --start=0 --stop=2175
"""
import argparse
import os
from collections import defaultdict

import numpy as np
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
    parser.add_argument("--start", default=0, help="Start index (default: 0)", type=int)
    parser.add_argument("--stop", default=None, help="End index (default: None, process all)", type=int)
    args = parser.parse_args()

    os.makedirs(args.numpy_dir, exist_ok=True)

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

    # Convert to list for indexing
    scan_ids = list(scan_slices.keys())
    stop = args.stop if args.stop is not None else len(scan_ids)
    scan_ids_subset = scan_ids[args.start:stop]
    print(f"Processing scans {args.start} to {stop} ({len(scan_ids_subset)} scans)")

    # Process each scan
    for scan_id in tqdm(scan_ids_subset, desc="Processing scans"):
        slices = scan_slices[scan_id]
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
