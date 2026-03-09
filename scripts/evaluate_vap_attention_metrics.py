#!/usr/bin/env python3
"""
Evaluate instance-level attention metrics for VAP models on RSNA dataset.

Computes: attention sum on positive slices, instance AUROC, AUPRC, max-attention accuracy.

Usage:
    python scripts/evaluate_vap_attention_metrics.py
    python scripts/evaluate_vap_attention_metrics.py --experiments_dir /path/to/results --dataset_dir /path/to/data
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
MODEL_TYPES = ['VAPGaussian', 'VAPBernoulli', 'VAPGaussianSparse']


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


def get_attention(model, X, lengths):
    """Get attention weights from a VAP model."""
    with torch.no_grad():
        _, attn = model(X, lengths)
        return attn.squeeze().numpy()


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


def compute_instance_metrics(attention, lengths, slice_labels):
    """Compute instance-level metrics from attention weights and slice labels."""
    attn_sums, aurocs, auprcs, max_corrects = [], [], [], []
    start = 0
    for i, length in enumerate(lengths):
        attn = attention[start:start + length]
        labels = slice_labels[i]
        start += length

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
    }


def evaluate_seed(experiments_dir, dataset_dir, labels_csv, numpy_dir, model_type, seed):
    """Evaluate instance-level metrics for one model_type + seed."""
    model_path, val_auroc = find_best_model(experiments_dir, model_type, seed)
    if not model_path or not os.path.exists(model_path):
        return None

    X, lengths, y = load_test_data(dataset_dir, seed)
    model = load_vap_model(model_path, X.shape[1], model_type)
    attention = get_attention(model, X, lengths)
    _, slice_labels = get_test_slice_labels(labels_csv, numpy_dir, seed)

    metrics = compute_instance_metrics(attention, lengths, slice_labels)
    metrics['val_auroc'] = val_auroc
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate VAP instance-level attention metrics')
    parser.add_argument('--experiments_dir', default=EXPERIMENTS_DIR, type=str)
    parser.add_argument('--dataset_dir', default=DATASET_DIR, type=str)
    parser.add_argument('--labels_csv', default=LABELS_CSV, type=str)
    parser.add_argument('--numpy_dir', default=NUMPY_DIR, type=str)
    args = parser.parse_args()

    print("=" * 80)
    print("VAP Instance-Level Attention Metrics (RSNA)")
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
            )
            if result:
                all_results.append(result)
                print(f"  Seed {seed}:")
                print(f"    Val AUROC:            {result['val_auroc']:.4f}")
                print(f"    Attn sum on positive: {result['attn_sum']:.4f}")
                print(f"    Instance AUROC:       {result['auroc']:.4f}")
                print(f"    Instance AUPRC:       {result['auprc']:.4f}")
                print(f"    Max attn correct:     {result['max_correct']:.4f}")
            else:
                print(f"  Seed {seed}: no model found")

        if all_results:
            print(f"\n  {model_type} Summary ({len(all_results)} seeds):")
            print(f"  {'Metric':<28} {'Mean':>10} {'Std':>10}")
            print(f"  {'-'*48}")
            for key, name in [
                ('attn_sum', 'Attn sum on positive'),
                ('auroc', 'Instance AUROC'),
                ('auprc', 'Instance AUPRC'),
                ('max_correct', 'Max attn correct'),
            ]:
                vals = [r[key] for r in all_results]
                print(f"  {name:<28} {np.mean(vals):>10.4f} {np.std(vals):>10.4f}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
