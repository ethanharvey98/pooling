#!/usr/bin/env python3
"""
Find best hyperparameters based on validation AUROC and report test AUROC
for KPSC site-based splits.

Usage:
    python find_best_results_kpsc.py /path/to/experiments/folder

Example:
    python find_best_results_kpsc.py /cluster/tufts/hugheslab/dloevl01/DINO3_Experiments/pooling/experiments/KPSC_MRI_800_CBI_3DINO_embedding_level=True
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np


SPLIT_SPECS = [
    "test_site_ids=9_train_site_ids=1_2_3_4_6_7_8_10_11_val_site_ids=5",
    "test_site_ids=1_4_7_10_11_train_site_ids=2_3_5_6_8_val_site_ids=9",
    "test_site_ids=2_6_train_site_ids=3_5_8_9_val_site_ids=1_4_7_10_11",
    "test_site_ids=3_8_train_site_ids=1_4_5_7_9_10_11_val_site_ids=2_6",
    "test_site_ids=5_train_site_ids=1_2_4_6_7_9_10_11_val_site_ids=3_8",
]


def find_best_for_split(split_path):
    """Find the best model (by val_auroc) for a given split and return its test_auroc."""
    csv_files = glob.glob(os.path.join(split_path, "*.csv"))

    if not csv_files:
        return None, None, None, None

    best_val_auroc = -1
    best_test_auroc = None
    best_file = None
    best_epoch = None

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'val_auroc' not in df.columns or 'test_auroc' not in df.columns:
                continue

            if 'train_auroc' in df.columns:
                valid_df = df[df['val_auroc'] <= df['train_auroc']]
                if valid_df.empty:
                    continue
            else:
                valid_df = df

            idx = valid_df['val_auroc'].idxmax()
            val_auroc = df.loc[idx, 'val_auroc']
            test_auroc = df.loc[idx, 'test_auroc']
            epoch = df.loc[idx, 'epoch'] if 'epoch' in df.columns else idx

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_test_auroc = test_auroc
                best_file = os.path.basename(csv_file)
                best_epoch = epoch

        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")
            continue

    return best_test_auroc, best_val_auroc, best_file, best_epoch


def main():
    parser = argparse.ArgumentParser(description='Find best results by validation AUROC (KPSC site splits)')
    parser.add_argument('folder', type=str, help='Path to folder containing split subdirectories')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory")
        return

    results = []

    print(f"\nSearching in: {args.folder}\n")
    print("=" * 80)

    for split_spec in SPLIT_SPECS:
        split_path = os.path.join(args.folder, split_spec)
        test_sites = split_spec.split("_train")[0].replace("test_site_ids=", "test=[") + "]"

        test_auroc, val_auroc, best_file, best_epoch = find_best_for_split(split_path)

        if test_auroc is not None:
            results.append(test_auroc)
            print(f"{test_sites}:")
            print(f"  Best file:    {best_file}")
            print(f"  Best epoch:   {best_epoch}")
            print(f"  Val AUROC:    {val_auroc:.4f}")
            print(f"  Test AUROC:   {test_auroc:.4f}")
            print()
        else:
            print(f"{test_sites}: No valid results found\n")

    print("=" * 80)

    if results:
        mean_auroc = np.mean(results)
        std_auroc = np.std(results)
        print(f"\nSUMMARY ({len(results)} splits):")
        print(f"  Test AUROCs:  {[f'{r:.4f}' for r in results]}")
        print(f"  Mean:         {mean_auroc:.4f}")
        print(f"  Std:          {std_auroc:.4f}")
        print(f"  Mean ± Std:   {mean_auroc:.4f} ± {std_auroc:.4f}")
    else:
        print("\nNo valid results found.")


if __name__ == "__main__":
    main()
