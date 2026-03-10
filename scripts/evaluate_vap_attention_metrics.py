#!/usr/bin/env python3
"""
Evaluate instance-level attention metrics for VAP models on RSNA dataset.

Computes: attention sum on positive slices, instance AUROC, AUPRC, max-attention accuracy.

Usage:
    python scripts/evaluate_vap_attention_metrics.py
    python scripts/evaluate_vap_attention_metrics.py --uncertain --mc_samples 20 --filter_pct 0.2
"""

import argparse
import ast
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import models


# Defaults
EXPERIMENTS_DIR = '/cluster/tufts/hugheslab/dloevl01/Feb_2026/March_2026/VAP_Tests/pooling'
DATASET_DIR = '/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16'
LABELS_CSV = '/cluster/tufts/hugheslab/datasets/RSNA/labels.csv'
NUMPY_DIR = '/cluster/tufts/hugheslab/datasets/RSNA_numpy'
SEEDS = [1001, 2001, 3001]
MODEL_TYPES = ['VAPGaussian', 'VAPGaussianCenter', 'VAPBernoulli', 'VAPGaussianSparse']


def find_best_model(experiments_dir, model_type, seed):
    """Find best model .pt file by val_auroc for a given model_type and seed."""
    pattern = os.path.join(experiments_dir, f"model={model_type}_*seed={seed}*.csv")
    best_val_auroc, best_file = -1, None

    for csv_file in glob.glob(pattern):
        try:
            df = pd.read_csv(csv_file)
        except (pd.errors.EmptyDataError, Exception):
            continue
        if 'val_auroc' not in df.columns or 'train_auroc' not in df.columns:
            continue
        valid_df = df[df['val_auroc'] <= df['train_auroc']]
        if valid_df.empty:
            continue
        idx = valid_df['val_auroc'].idxmax()
        if df.loc[idx, 'val_auroc'] > best_val_auroc:
            best_val_auroc = df.loc[idx, 'val_auroc']
            best_file = csv_file.replace('.csv', '.pt')

    return best_file, best_val_auroc


def load_vap_model(model_path, in_features, model_type):
    """Load a VAP model from checkpoint."""
    if model_type in ('VAPGaussian', 'VAPGaussianCenter'):
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


def get_attention(model, X, lengths):
    """Get attention weights from a VAP model."""
    with torch.no_grad():
        _, attn = model(X, lengths)
        return attn.squeeze().numpy()


def get_uncertainty(model, X, lengths, mc_samples):
    """Run MC forward passes and return mean attention, mean logits, and per-scan logits variance."""
    unc = model.predict_with_uncertainty(X, lengths, n_samples=mc_samples)
    # Mean attention across samples: (mc_samples, N, 1) -> (N,)
    mean_attn = unc['attn_samples'].mean(0).squeeze().numpy()
    # Per-scan logits variance: (n_bags,)
    logits_var = unc['logits_var'].squeeze().numpy()
    return mean_attn, logits_var


def load_test_data(dataset_dir, seed):
    """Load test data."""
    test_data = torch.load(f'{dataset_dir}/seed={seed}/test.pth', map_location='cpu', weights_only=False)
    return test_data['X'], test_data['lengths'], test_data['y']


def get_test_slice_labels(labels_csv, numpy_dir, seed):
    """Get per-slice labels for test set (same split logic as encoding)."""
    df = pd.read_csv(labels_csv)
    df['scan_label'] = df['Any'].apply(lambda x: 1 if any(ast.literal_eval(x)) else 0)
    df['path'] = df['Study ID'].apply(lambda x: f'{numpy_dir}/{x}.npz')
    df = df[df['path'].apply(os.path.exists)]

    _, test_ids, _, _ = train_test_split(
        df['Study ID'], df['scan_label'],
        test_size=1/6, random_state=seed, stratify=df['scan_label']
    )
    test_df = df[df['Study ID'].isin(test_ids)]
    return test_df['Study ID'].values, test_df['Any'].apply(lambda x: np.array(ast.literal_eval(x))).values


def compute_instance_metrics(attention, lengths, slice_labels, keep_mask=None):
    """Compute instance-level metrics from attention weights and slice labels.

    Args:
        attention: flat array of attention weights
        lengths: tuple of bag lengths
        slice_labels: list of per-bag label arrays
        keep_mask: optional boolean array (per bag) of which bags to include
    """
    attn_sums, aurocs, auprcs, max_corrects = [], [], [], []
    start = 0
    for i, length in enumerate(lengths):
        attn = attention[start:start + length]
        labels = slice_labels[i]
        start += length

        if keep_mask is not None and not keep_mask[i]:
            continue

        if labels.sum() == 0:
            continue

        attn_sums.append(attn[labels == 1].sum())
        if len(np.unique(labels)) > 1:
            aurocs.append(roc_auc_score(labels, attn))
            auprcs.append(average_precision_score(labels, attn))
        max_corrects.append(labels[np.argmax(attn)] == 1)

    return {
        'attn_sum': np.mean(attn_sums) if attn_sums else float('nan'),
        'auroc': np.mean(aurocs) if aurocs else float('nan'),
        'auprc': np.mean(auprcs) if auprcs else float('nan'),
        'max_correct': np.mean(max_corrects) if max_corrects else float('nan'),
        'n_bags': len(attn_sums),
    }


def evaluate_seed(experiments_dir, dataset_dir, labels_csv, numpy_dir,
                  model_type, seed, uncertain=False, mc_samples=10, filter_pct=0.2):
    """Evaluate instance-level metrics for one model_type + seed."""
    model_path, val_auroc = find_best_model(experiments_dir, model_type, seed)
    if not model_path or not os.path.exists(model_path):
        return None

    X, lengths, y = load_test_data(dataset_dir, seed)
    model = load_vap_model(model_path, X.shape[1], model_type)
    _, slice_labels = get_test_slice_labels(labels_csv, numpy_dir, seed)

    if uncertain:
        attention, logits_var = get_uncertainty(model, X, lengths, mc_samples)
        # Build keep mask: drop the top filter_pct most uncertain scans
        n_bags = len(lengths)
        threshold = np.percentile(logits_var, (1 - filter_pct) * 100)
        keep_mask = logits_var <= threshold

        # Compute metrics on all samples (baseline)
        metrics_all = compute_instance_metrics(attention, lengths, slice_labels)
        # Compute metrics on filtered (certain) samples
        metrics_filtered = compute_instance_metrics(attention, lengths, slice_labels, keep_mask=keep_mask)

        return {
            'val_auroc': val_auroc,
            'all': metrics_all,
            'filtered': metrics_filtered,
            'n_total': n_bags,
            'n_kept': int(keep_mask.sum()),
            'var_mean': float(logits_var.mean()),
            'var_threshold': float(threshold),
        }
    else:
        attention = get_attention(model, X, lengths)
        metrics = compute_instance_metrics(attention, lengths, slice_labels)
        metrics['val_auroc'] = val_auroc
        return metrics


def print_metrics(metrics, indent="    "):
    """Print a metrics dict."""
    print(f"{indent}Attn sum on positive: {metrics['attn_sum']:.4f}")
    print(f"{indent}Instance AUROC:       {metrics['auroc']:.4f}")
    print(f"{indent}Instance AUPRC:       {metrics['auprc']:.4f}")
    print(f"{indent}Max attn correct:     {metrics['max_correct']:.4f}")
    if 'n_bags' in metrics:
        print(f"{indent}Bags evaluated:       {metrics['n_bags']}")


def print_summary(all_results, model_type, key=None):
    """Print summary table for a list of result dicts."""
    if key:
        metric_dicts = [r[key] for r in all_results]
    else:
        metric_dicts = all_results

    label = f"{model_type} ({key})" if key else model_type
    print(f"\n  {label} Summary ({len(metric_dicts)} seeds):")
    print(f"  {'Metric':<28} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*48}")
    for mkey, name in [
        ('attn_sum', 'Attn sum on positive'),
        ('auroc', 'Instance AUROC'),
        ('auprc', 'Instance AUPRC'),
        ('max_correct', 'Max attn correct'),
    ]:
        vals = [m[mkey] for m in metric_dicts]
        print(f"  {name:<28} {np.mean(vals):>10.4f} {np.std(vals):>10.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VAP instance-level attention metrics')
    parser.add_argument('--experiments_dir', default=EXPERIMENTS_DIR, type=str)
    parser.add_argument('--dataset_dir', default=DATASET_DIR, type=str)
    parser.add_argument('--labels_csv', default=LABELS_CSV, type=str)
    parser.add_argument('--numpy_dir', default=NUMPY_DIR, type=str)
    parser.add_argument('--uncertain', action='store_true', default=False,
                        help='Use MC uncertainty to filter out most uncertain samples')
    parser.add_argument('--mc_samples', default=10, type=int,
                        help='Number of MC forward passes for uncertainty (default: 10)')
    parser.add_argument('--filter_pct', default=0.2, type=float,
                        help='Fraction of most uncertain scans to filter out (default: 0.2)')
    args = parser.parse_args()

    mode_str = "with uncertainty filtering" if args.uncertain else "deterministic"
    print("=" * 80)
    print(f"VAP Instance-Level Attention Metrics (RSNA) — {mode_str}")
    if args.uncertain:
        print(f"  MC samples: {args.mc_samples}, filtering top {args.filter_pct*100:.0f}% most uncertain")
    print("=" * 80)

    for model_type in MODEL_TYPES:
        print(f"\n{'='*80}")
        print(f"  {model_type}")
        print('='*80)

        all_results = []
        for seed in SEEDS:
            result = evaluate_seed(
                args.experiments_dir, args.dataset_dir,
                args.labels_csv, args.numpy_dir,
                model_type, seed,
                uncertain=args.uncertain,
                mc_samples=args.mc_samples,
                filter_pct=args.filter_pct,
            )
            if result is None:
                print(f"  Seed {seed}: no model found")
                continue

            all_results.append(result)

            if args.uncertain:
                print(f"  Seed {seed}:  (val_auroc={result['val_auroc']:.4f}, "
                      f"kept {result['n_kept']}/{result['n_total']} scans, "
                      f"var_threshold={result['var_threshold']:.4f})")
                print(f"    --- All samples ---")
                print_metrics(result['all'])
                print(f"    --- After filtering top {args.filter_pct*100:.0f}% uncertain ---")
                print_metrics(result['filtered'])
            else:
                print(f"  Seed {seed}:")
                print(f"    Val AUROC:            {result['val_auroc']:.4f}")
                print_metrics(result)

        if all_results:
            if args.uncertain:
                print_summary(all_results, model_type, key='all')
                print_summary(all_results, model_type, key='filtered')
            else:
                print_summary(all_results, model_type)

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
