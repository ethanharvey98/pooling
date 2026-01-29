#!/usr/bin/env python3
"""
Evaluate attention metrics for different pooling methods on RSNA dataset.

Usage:
    python scripts/evaluate_attention_metrics.py
"""

import ast
import glob
import os
import sys

import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

sys.path.append('src')
import models


# Configuration
EXPERIMENTS_DIR = '/cluster/tufts/hugheslab/dloevl01/pooling/experiments/RSNA/embedding_level=True'
DATASET_DIR = '/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16'
LABELS_CSV = '/cluster/tufts/hugheslab/datasets/RSNA/labels.csv'
NUMPY_DIR = '/cluster/tufts/hugheslab/datasets/RSNA_numpy'
EMBEDDING_LEVEL = True
SEEDS = [1001, 2001, 3001]
METHODS = ['ABMIL', 'TransMIL', 'SmAP']
BASELINES = ['Middle12', 'Gaussian']  # Baselines that don't require trained models
GAUSSIAN_SIGMA = 1.0  # Default sigma for Gaussian baseline


def find_best_model(experiments_dir, pooling, seed):
    """Find best model by validation AUROC."""
    import pandas as pd
    pattern = os.path.join(experiments_dir, f"*pooling={pooling}*seed={seed}*.csv")
    best_val_auroc, best_file = -1, None

    for csv_file in glob.glob(pattern):
        try:
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            print(f"  Warning: Empty CSV file skipped: {csv_file}")
            continue
        valid_df = df[df['val_auroc'] <= df['train_auroc']]
        if valid_df.empty:
            continue
        idx = valid_df['val_auroc'].idxmax()
        if df.loc[idx, 'val_auroc'] > best_val_auroc:
            best_val_auroc = df.loc[idx, 'val_auroc']
            best_file = csv_file.replace('.csv', '.pt')

    return best_file, best_val_auroc


def load_model(model_path, in_features, pooling, embedding_level):
    """Load model from checkpoint."""
    if embedding_level:
        model = models.PoolClf(in_features, 1, pooling)
    else:
        model = models.ClfPool(in_features, 1, pooling)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def get_attention(model, X, lengths, embedding_level):
    """Get attention weights from model."""
    with torch.no_grad():
        _, attn = model(X, lengths)
        if embedding_level:
            return attn.squeeze().numpy()
        return torch.sigmoid(model.clf(X)).squeeze().numpy()


def get_test_slice_labels(labels_csv, numpy_dir, seed):
    """Get test set slice labels."""
    import pandas as pd
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


def get_middle12_attention(length):
    """
    Get attention weights for Middle 12 baseline.
    Places uniform attention (1/12) on the middle 12 slices, 0 elsewhere.
    If scan has fewer than 12 slices, uses all slices uniformly.
    """
    attn = np.zeros(length)
    n_middle = min(12, length)
    start_idx = (length - n_middle) // 2
    end_idx = start_idx + n_middle
    attn[start_idx:end_idx] = 1.0 / n_middle
    return attn


def get_gaussian_attention(length, sigma=1.0):
    """
    Get Gaussian-distributed attention weights centered on middle.
    Normalized to sum to 1.
    """
    center = (length - 1) / 2.0
    indices = np.arange(length)
    attn = norm.pdf(indices, loc=center, scale=sigma)
    return attn / attn.sum()


def evaluate_seed(experiments_dir, dataset_dir, labels_csv, numpy_dir, pooling, seed, embedding_level):
    """Evaluate a single seed for a pooling method."""
    model_path, val_auroc = find_best_model(experiments_dir, pooling, seed)
    if not model_path:
        return None

    test_data = torch.load(f'{dataset_dir}/seed={seed}/test.pth', map_location='cpu', weights_only=False)
    X, lengths, y = test_data['X'], test_data['lengths'], test_data['y']

    model = load_model(model_path, X.shape[1], pooling, embedding_level)
    attention = get_attention(model, X, lengths, embedding_level)
    _, slice_labels = get_test_slice_labels(labels_csv, numpy_dir, seed)

    attn_sum, aurocs, auprcs, max_correct, max_attns = [], [], [], [], []
    start = 0
    for i, length in enumerate(lengths):
        attn = attention[start:start + length]
        labels = slice_labels[i]
        start += length

        if labels.sum() == 0:
            continue

        attn_sum.append(attn[labels == 1].sum())
        if len(np.unique(labels)) > 1:
            aurocs.append(roc_auc_score(labels, attn))
            auprcs.append(average_precision_score(labels, attn))
        max_correct.append(labels[np.argmax(attn)] == 1)
        max_attns.append(attn.max())

    return {
        'val_auroc': val_auroc,
        'attn_sum': np.mean(attn_sum),
        'auroc': np.mean(aurocs),
        'auprc': np.mean(auprcs),
        'max_correct': np.mean(max_correct),
        'max_attn': np.mean(max_attns)
    }


def evaluate_baseline_seed(dataset_dir, labels_csv, numpy_dir, baseline, seed):
    """Evaluate a single seed for a baseline method (no trained model required)."""
    test_data = torch.load(f'{dataset_dir}/seed={seed}/test.pth', map_location='cpu', weights_only=False)
    lengths = test_data['lengths']
    _, slice_labels = get_test_slice_labels(labels_csv, numpy_dir, seed)

    attn_sum, aurocs, auprcs, max_correct, max_attns = [], [], [], [], []

    for i, length in enumerate(lengths):
        labels = slice_labels[i]

        if labels.sum() == 0:
            continue

        # Generate baseline attention based on method
        if baseline == 'Middle12':
            attn = get_middle12_attention(length)
        elif baseline == 'Gaussian':
            attn = get_gaussian_attention(length, sigma=GAUSSIAN_SIGMA)
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        attn_sum.append(attn[labels == 1].sum())
        if len(np.unique(labels)) > 1:
            aurocs.append(roc_auc_score(labels, attn))
            auprcs.append(average_precision_score(labels, attn))
        max_correct.append(labels[np.argmax(attn)] == 1)
        max_attns.append(attn.max())

    return {
        'attn_sum': np.mean(attn_sum),
        'auroc': np.mean(aurocs),
        'auprc': np.mean(auprcs),
        'max_correct': np.mean(max_correct),
        'max_attn': np.mean(max_attns)
    }


def main():
    print("=" * 80)
    print("Evaluating Attention Metrics on RSNA Dataset")
    print("=" * 80)
    print()

    # Evaluate trained models
    for method in METHODS:
        print(f"\n{'='*80}")
        print(f"Method: {method}")
        print('='*80)

        results = []
        for seed in SEEDS:
            result = evaluate_seed(
                EXPERIMENTS_DIR, DATASET_DIR, LABELS_CSV, NUMPY_DIR,
                method, seed, EMBEDDING_LEVEL
            )
            if result:
                results.append(result)
                print(f"Seed {seed}:")
                print(f"  Val AUROC:            {result['val_auroc']:.4f}")
                print(f"  Attn sum on positive: {result['attn_sum']:.4f}")
                print(f"  AUROC:                {result['auroc']:.4f}")
                print(f"  AUPRC:                {result['auprc']:.4f}")
                print(f"  Max attn correct:     {result['max_correct']:.4f}")
                print(f"  Max attention:        {result['max_attn']:.4f}")
            else:
                print(f"Seed {seed}: No model found")

        if results:
            results = np.array([[r['attn_sum'], r['auroc'], r['auprc'], r['max_correct'], r['max_attn']] for r in results])
            print(f"\n{method} Summary ({len(results)} seeds):")
            print(f"{'Metric':<30} {'Mean':>10} {'Std':>10}")
            print("-" * 50)
            for i, name in enumerate(['Attention sum on positive', 'AUROC', 'AUPRC', 'Max attention correct', 'Max attention']):
                print(f"{name:<30} {results[:, i].mean():>10.4f} {results[:, i].std():>10.4f}")
        else:
            print(f"\nNo {method} models found")

    # Evaluate baselines
    for baseline in BASELINES:
        print(f"\n{'='*80}")
        print(f"Baseline: {baseline}")
        print('='*80)

        results = []
        for seed in SEEDS:
            result = evaluate_baseline_seed(
                DATASET_DIR, LABELS_CSV, NUMPY_DIR, baseline, seed
            )
            results.append(result)
            print(f"Seed {seed}:")
            print(f"  Attn sum on positive: {result['attn_sum']:.4f}")
            print(f"  AUROC:                {result['auroc']:.4f}")
            print(f"  AUPRC:                {result['auprc']:.4f}")
            print(f"  Max attn correct:     {result['max_correct']:.4f}")
            print(f"  Max attention:        {result['max_attn']:.4f}")

        results = np.array([[r['attn_sum'], r['auroc'], r['auprc'], r['max_correct'], r['max_attn']] for r in results])
        print(f"\n{baseline} Summary ({len(results)} seeds):")
        print(f"{'Metric':<30} {'Mean':>10} {'Std':>10}")
        print("-" * 50)
        for i, name in enumerate(['Attention sum on positive', 'AUROC', 'AUPRC', 'Max attention correct', 'Max attention']):
            print(f"{name:<30} {results[:, i].mean():>10.4f} {results[:, i].std():>10.4f}")

    print("\n" + "="*80)
    print("Evaluation complete")
    print("="*80)


if __name__ == '__main__':
    main()
