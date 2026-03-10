#!/usr/bin/env python3
"""
Find best VAP hyperparameters by model type, based on validation AUROC.
Optionally use MC uncertainty to filter out most uncertain scans and re-evaluate bag-level AUROC.

Usage:
    python find_best_vap_results.py /path/to/results
    python find_best_vap_results.py /path/to/results --uncertain --mc_samples 20 --filter_pct 0.2
"""

import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import models


DATASET_DIR = '/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16'
MODEL_TYPES = ['VAPGaussian', 'VAPBernoulli', 'VAPGaussianSparse']
SEEDS = [1001, 2001, 3001]


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
                    'pt_file': csv_file.replace('.csv', '.pt'),
                    'epoch': df.loc[idx, 'epoch'] if 'epoch' in df.columns else idx,
                    'val_auroc': val_auroc,
                    'test_auroc': df.loc[idx, 'test_auroc'],
                    'train_auroc': df.loc[idx, 'train_auroc'] if 'train_auroc' in df.columns else None,
                    'test_kl': df.loc[idx, 'test_kl'] if 'test_kl' in df.columns else None,
                }
        except Exception:
            continue

    return best_info


def load_vap_model(model_path, in_features, model_type):
    """Load a VAP model from checkpoint."""
    if model_type == 'VAPGaussian':
        model = models.VAPGaussianMIL(in_features=in_features, out_features=1)
    elif model_type == 'VAPBernoulli':
        model = models.VAPBernoulliMIL(in_features=in_features, out_features=1)
    elif model_type == 'VAPGaussianSparse':
        model = models.VAPGaussianSparseMIL(in_features=in_features, out_features=1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def load_test_data(dataset_dir, seed):
    test_data = torch.load(f'{dataset_dir}/seed={seed}/test.pth', map_location='cpu', weights_only=False)
    return test_data['X'], test_data['lengths'], test_data['y']


def evaluate_with_uncertainty(model, X, lengths, y, mc_samples, filter_pct):
    """Run MC inference, compute bag-level AUROC on all and filtered (certain) scans."""
    unc = model.predict_with_uncertainty(X, lengths, n_samples=mc_samples)
    logits_mean = unc['logits_mean'].squeeze().numpy()
    logits_var = unc['logits_var'].squeeze().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_mean))
    labels = y.squeeze().numpy()

    # All scans
    auroc_all = roc_auc_score(labels, probs)

    # Filter out top filter_pct most uncertain
    threshold = np.percentile(logits_var, (1 - filter_pct) * 100)
    keep = logits_var <= threshold

    if len(np.unique(labels[keep])) > 1:
        auroc_filtered = roc_auc_score(labels[keep], probs[keep])
    else:
        auroc_filtered = float('nan')

    return {
        'auroc_all': auroc_all,
        'auroc_filtered': auroc_filtered,
        'n_total': len(labels),
        'n_kept': int(keep.sum()),
        'var_mean': float(logits_var.mean()),
        'var_threshold': float(threshold),
    }


def main():
    parser = argparse.ArgumentParser(description='Find best VAP results by model type')
    parser.add_argument('folder', type=str, help='Path to folder containing CSV result files')
    parser.add_argument('--dataset_dir', default=DATASET_DIR, type=str)
    parser.add_argument('--uncertain', action='store_true', default=False,
                        help='Re-evaluate best models with MC uncertainty filtering')
    parser.add_argument('--mc_samples', default=10, type=int,
                        help='Number of MC forward passes (default: 10)')
    parser.add_argument('--filter_pct', default=0.2, type=float,
                        help='Fraction of most uncertain scans to filter out (default: 0.2)')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory")
        return

    csv_files = glob.glob(os.path.join(args.folder, '*.csv'))
    if not csv_files:
        print("No CSV files found yet.")
        return

    mode_str = "with uncertainty filtering" if args.uncertain else "deterministic"
    print(f"\nSearching in: {args.folder}")
    print(f"Found {len(csv_files)} CSV files — {mode_str}")
    if args.uncertain:
        print(f"  MC samples: {args.mc_samples}, filtering top {args.filter_pct*100:.0f}% most uncertain")
    print()

    for model_type in MODEL_TYPES:
        print("=" * 80)
        print(f"  {model_type}")
        print("=" * 80)

        test_aurocs = []
        filtered_aurocs = []

        for seed in SEEDS:
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

            if args.uncertain and os.path.exists(best['pt_file']):
                X, lengths, y = load_test_data(args.dataset_dir, seed)
                model = load_vap_model(best['pt_file'], X.shape[1], model_type)
                unc_result = evaluate_with_uncertainty(model, X, lengths, y,
                                                       args.mc_samples, args.filter_pct)
                filtered_aurocs.append(unc_result['auroc_filtered'])
                print(f"           MC AUROC (all):      {unc_result['auroc_all']:.4f}")
                print(f"           MC AUROC (filtered):  {unc_result['auroc_filtered']:.4f}  "
                      f"(kept {unc_result['n_kept']}/{unc_result['n_total']}, "
                      f"var_threshold={unc_result['var_threshold']:.4f})")

        if test_aurocs:
            mean = np.mean(test_aurocs)
            std = np.std(test_aurocs)
            print(f"\n  >> {model_type}: {mean:.4f} +/- {std:.4f}  (n={len(test_aurocs)})")

        if filtered_aurocs:
            fmean = np.mean(filtered_aurocs)
            fstd = np.std(filtered_aurocs)
            print(f"  >> {model_type} (filtered): {fmean:.4f} +/- {fstd:.4f}  (n={len(filtered_aurocs)})")

        print()


if __name__ == "__main__":
    main()
