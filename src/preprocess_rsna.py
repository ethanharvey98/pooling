import argparse
import os
import numpy as np
import pandas as pd
import pydicom
# Importing our custom module(s)
import ct

# python ../src/preprocess_rsna.py --csv_path='/home/denny-loevlie/RSNA_Investigation/rsna_ich_subset_1149_complete.csv' --dicom_dir='/media/M2SSD/gen_models_data/stage_2_train' --numpy_dir='/media/M2SSD/gen_models_data/RSNA_ICH_numpy'
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess_rsna.py")
    parser.add_argument("--csv_path", help="Path to CSV with scan/slice IDs", type=str)
    parser.add_argument("--dicom_dir", help="Directory containing DICOM files", type=str)
    parser.add_argument("--numpy_dir", help="Directory to save numpy dataset", type=str)
    args = parser.parse_args()

    os.makedirs(args.numpy_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    scan_ids = df['scan_id'].unique()

    for scan_id in scan_ids:

        slice_info = df[df['scan_id'] == scan_id].sort_values('slice_order')
        images = []

        for _, row in slice_info.iterrows():

            dcm_path = f"{args.dicom_dir}/{row['slice_id']}.dcm"
            dcm = pydicom.dcmread(dcm_path)
            image = dcm.pixel_array.astype(np.float32)
            image = image * float(getattr(dcm, 'RescaleSlope', 1)) + float(getattr(dcm, 'RescaleIntercept', 0))
            image = ct.strip_skull(image)
            images.append(image)

        print(f"{args.numpy_dir}/{scan_id}.npz")
        np.savez(f"{args.numpy_dir}/{scan_id}.npz", np.array(images))
