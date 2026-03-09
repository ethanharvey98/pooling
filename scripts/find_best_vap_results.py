#!/usr/bin/env python3
"""
Find best VAP hyperparameters by model type, based on validation AUROC.

Usage:
    python find_best_vap_results.py /cluster/tufts/hugheslab/dloevl01/Feb_2026/March_2026/VAP_Tests/pooling
"""

import argparse
import os
import glob
import re
import pandas as pd
import numpy as np


def parse_model_name(filename):
    """Extract model_type, beta, lr, seed from filename."""
    base = os.path.splitext(filename)[0]
    info = {}
    for part in base.split('_'):
        if '=' in part:
            key, val = part.split('=', 1)
            info[key] = val
    return info


def find_best_for_seed(csv_files):
    """Find the best model (by val_auroc) among a list of CSVs and return metrics."""
    best_val_auroc = -1
    best_info = None

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

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_info = {
                    'file': os.path.basename(csv_file),
                    'epoch': df.loc[idx, 'epoch'] if 'epoch' in df.columns else idx,
                    'val_auroc': val_auroc,
                    'test_auroc': df.loc[idx, 'test_auroc'],
                    'train_auroc': df.loc[idx, 'train_auroc'] if 'train_auroc' in df.columns else None,
                    'test_kl': df.loc[idx, 'test_kl'] if 'test_kl' in df.columns else None,
                }
        except Exception as e:
            continue

    return best_info


def main():
    parser = argparse.ArgumentParser(description='Find best VAP results by model type')
    parser.add_argument('folder', type=str, help='Path to folder containing CSV result files')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory")
        return

    csv_files = glob.glob(os.path.join(args.folder, '*.csv'))
    if not csv_files:
        print("No CSV files found yet.")
        return

    # Group files by model type and seed
    model_types = ['VAPGaussian', 'VAPBernoulli', 'VAPGaussianSparse']
    seeds = [1001, 2001, 3001]

    print(f"\nSearching in: {args.folder}")
    print(f"Found {len(csv_files)} CSV files\n")

    for model_type in model_types:
        print("=" * 80)
        print(f"  {model_type}")
        print("=" * 80)

        test_aurocs = []

        for seed in seeds:
            seed_files = [f for f in csv_files
                          if f"model={model_type}_" in os.path.basename(f)
                          and f"seed={seed}" in os.path.basename(f)]

            if not seed_files:
                print(f"  Seed {seed}: no results yet")
                continue

            best = find_best_for_seed(seed_files)
            if best is None:
                print(f"  Seed {seed}: no valid results yet")
                continue

            test_aurocs.append(best['test_auroc'])
            kl_str = f"  KL: {best['test_kl']:.4f}" if best['test_kl'] is not None else ""
            print(f"  Seed {seed}: test_auroc={best['test_auroc']:.4f}  val_auroc={best['val_auroc']:.4f}  epoch={best['epoch']}{kl_str}")
            print(f"           {best['file']}")

        if test_aurocs:
            mean = np.mean(test_aurocs)
            std = np.std(test_aurocs)
            print(f"\n  >> {model_type}: {mean:.4f} +/- {std:.4f}  (n={len(test_aurocs)})")
        print()


if __name__ == "__main__":
    main()
