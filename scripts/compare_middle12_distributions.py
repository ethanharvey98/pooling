#!/usr/bin/env python3
"""
Compare Middle 12 attention distributions: uniform vs Gaussian.

Tests on the validation set to compare:
- Middle 12 with uniform attention (1/12 for each of 12 middle slices)
- Middle 12 with Gaussian-distributed attention (bell curve centered on middle)

Usage:
    python scripts/compare_middle12_distributions.py
"""

import ast
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import torch

sys.path.append('src')


# Configuration
OUTPUT_DIR = 'figures'
DATASET_DIR = '/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16'
LABELS_CSV = '/cluster/tufts/hugheslab/datasets/RSNA/labels.csv'
NUMPY_DIR = '/cluster/tufts/hugheslab/datasets/RSNA_numpy'
SEEDS = [1001, 2001, 3001]


def get_val_slice_labels(labels_csv, numpy_dir, seed):
    """Get validation set slice labels, sorted by Study ID for consistent ordering."""
    import pandas as pd
    df = pd.read_csv(labels_csv)
    df['scan_label'] = df['Any'].apply(lambda x: 1 if any(ast.literal_eval(x)) else 0)
    df['path'] = df['Study ID'].apply(lambda x: f'{numpy_dir}/{x}.npz')
    df = df[df['path'].apply(os.path.exists)]

    # Sort by Study ID to ensure consistent ordering with encoded data
    df = df.sort_values('Study ID').reset_index(drop=True)

    # Split: first get test set, then split remainder into train/val
    train_val_ids, _, train_val_labels, _ = train_test_split(
        df['Study ID'], df['scan_label'],
        test_size=1/6, random_state=seed, stratify=df['scan_label']
    )
    train_val_df = df[df['Study ID'].isin(train_val_ids)]

    # Split train_val into train and val (1/5 of train_val = 1/6 of total for val)
    _, val_ids, _, _ = train_test_split(
        train_val_df['Study ID'], train_val_df['scan_label'],
        test_size=1/5, random_state=seed, stratify=train_val_df['scan_label']
    )
    # Sort val_df by Study ID to match the order used during encoding
    val_df = df[df['Study ID'].isin(val_ids)].sort_values('Study ID')
    return val_df['Study ID'].values, val_df['Any'].apply(lambda x: np.array(ast.literal_eval(x))).values


def get_middle12_uniform(length):
    """
    Get uniform attention weights for Middle 12.
    Places uniform attention (1/12) on the middle 12 slices, 0 elsewhere.
    """
    attn = np.zeros(length)
    n_middle = min(12, length)
    start_idx = (length - n_middle) // 2
    end_idx = start_idx + n_middle
    attn[start_idx:end_idx] = 1.0 / n_middle
    return attn


def get_middle12_gaussian(length, sigma_slices=3.0):
    """
    Get Gaussian-distributed attention weights centered on middle.

    The Gaussian is centered at the middle of the scan with standard deviation
    sigma_slices. Attention is normalized to sum to 1.

    Args:
        length: Number of slices in scan
        sigma_slices: Standard deviation in number of slices (default: 3.0)
    """
    center = (length - 1) / 2.0
    indices = np.arange(length)
    attn = norm.pdf(indices, loc=center, scale=sigma_slices)
    attn = attn / attn.sum()  # Normalize to sum to 1
    return attn


def plot_attention_distributions(length, sigma_values, output_dir):
    """
    Plot attention distributions for uniform and Gaussian methods.

    Args:
        length: Number of slices to use for the example
        sigma_values: List of sigma values for Gaussian distributions
        output_dir: Directory to save the plot
    """
    plt.rcParams.update({"font.size": 10})

    slice_indices = np.arange(1, length + 1)
    center = (length - 1) / 2.0

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot uniform
    uniform_attn = get_middle12_uniform(length)
    ax.plot(slice_indices, uniform_attn, label='Uniform (Middle 12)',
            linewidth=2, linestyle='--', color='black')

    # Plot Gaussian for each sigma
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sigma_values)))
    for sigma, color in zip(sigma_values, colors):
        gaussian_attn = get_middle12_gaussian(length, sigma_slices=sigma)
        # Compute actual mean and variance of the distribution
        mean = np.sum(slice_indices * gaussian_attn)
        var = np.sum(((slice_indices - mean) ** 2) * gaussian_attn)
        ax.plot(slice_indices, gaussian_attn,
                label=rf'Gaussian ($\sigma$={sigma}, $\mu$={mean:.1f}, $\sigma^2$={var:.1f})',
                linewidth=2, color=color)

    ax.set_xlabel(r'Slice index $j$')
    ax.set_ylabel(r'Attention $a_j$')
    ax.set_xlim(1, length)
    ax.set_title(f'Attention Distributions (scan length = {length})')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # Add vertical line at center
    ax.axvline(x=center + 1, color='gray', linestyle=':', alpha=0.5, label='_center')

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/middle12_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/middle12_distributions.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir}/middle12_distributions.png")
    print(f"Saved: {output_dir}/middle12_distributions.pdf")
    plt.close()


def evaluate_baseline(lengths, slice_labels, attention_fn, **kwargs):
    """Evaluate a baseline attention method."""
    attn_sum, aurocs, auprcs, max_correct = [], [], [], []
    mismatches = 0

    for i, length in enumerate(lengths):
        labels = slice_labels[i]

        # Check for length mismatch between encoded data and labels
        if len(labels) != length:
            mismatches += 1
            continue

        if labels.sum() == 0:
            continue

        attn = attention_fn(length, **kwargs)

        attn_sum.append(attn[labels == 1].sum())
        if len(np.unique(labels)) > 1:
            aurocs.append(roc_auc_score(labels, attn))
            auprcs.append(average_precision_score(labels, attn))
        max_correct.append(labels[np.argmax(attn)] == 1)

    if mismatches > 0:
        print(f"  WARNING: {mismatches}/{len(lengths)} scans had length mismatches (skipped)")

    return {
        'attn_sum': np.mean(attn_sum),
        'auroc': np.mean(aurocs),
        'auprc': np.mean(auprcs),
        'max_correct': np.mean(max_correct),
    }


def main():
    print("=" * 80)
    print("Comparing Middle 12 Distributions: Uniform vs Gaussian")
    print("Evaluation on VALIDATION set")
    print("=" * 80)
    print()

    # Test different sigma values for Gaussian
    sigma_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    # Plot attention distributions for visualization
    example_length = 36  # Typical scan length
    print(f"Plotting attention distributions for example scan (length={example_length})...")
    plot_attention_distributions(example_length, sigma_values, OUTPUT_DIR)
    print()

    all_results = {'Uniform': []}
    for sigma in sigma_values:
        all_results[f'Gaussian(σ={sigma})'] = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print('='*60)

        # Load validation data
        val_data = torch.load(f'{DATASET_DIR}/seed={seed}/val.pth', map_location='cpu', weights_only=False)
        lengths = val_data['lengths']
        _, slice_labels = get_val_slice_labels(LABELS_CSV, NUMPY_DIR, seed)

        print(f"Validation set: {len(lengths)} scans")
        print()

        # Evaluate uniform
        result = evaluate_baseline(lengths, slice_labels, get_middle12_uniform)
        all_results['Uniform'].append(result)
        print(f"Middle 12 Uniform:")
        print(f"  Attn sum on positive: {result['attn_sum']:.4f}")
        print(f"  AUROC:                {result['auroc']:.4f}")
        print(f"  AUPRC:                {result['auprc']:.4f}")
        print(f"  Max attn correct:     {result['max_correct']:.4f}")
        print()

        # Evaluate Gaussian with different sigma values
        for sigma in sigma_values:
            result = evaluate_baseline(lengths, slice_labels, get_middle12_gaussian, sigma_slices=sigma)
            all_results[f'Gaussian(σ={sigma})'].append(result)
            # Compute mean and variance for a typical scan in this set
            median_length = int(np.median(lengths))
            example_attn = get_middle12_gaussian(median_length, sigma_slices=sigma)
            indices = np.arange(1, median_length + 1)
            dist_mean = np.sum(indices * example_attn)
            dist_var = np.sum(((indices - dist_mean) ** 2) * example_attn)
            print(f"Middle 12 Gaussian (σ={sigma}, μ={dist_mean:.1f}, σ²={dist_var:.1f}):")
            print(f"  Attn sum on positive: {result['attn_sum']:.4f}")
            print(f"  AUROC:                {result['auroc']:.4f}")
            print(f"  AUPRC:                {result['auprc']:.4f}")
            print(f"  Max attn correct:     {result['max_correct']:.4f}")
            print()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL SEEDS")
    print("=" * 80)
    print()

    metrics = ['attn_sum', 'auroc', 'auprc', 'max_correct']
    metric_names = ['Attn Sum', 'AUROC', 'AUPRC', 'Max Correct']

    # Header
    print(f"{'Method':<25}", end='')
    for name in metric_names:
        print(f"{name:>15}", end='')
    print()
    print("-" * 85)

    # Results for each method
    for method, results in all_results.items():
        print(f"{method:<25}", end='')
        for metric in metrics:
            values = [r[metric] for r in results]
            mean = np.mean(values)
            std = np.std(values)
            print(f"{mean:>7.4f}±{std:.4f}", end='')
        print()

    # Find best method for each metric
    print()
    print("Best method by metric:")
    print("-" * 40)
    for metric, name in zip(metrics, metric_names):
        best_method = None
        best_mean = -1
        for method, results in all_results.items():
            mean = np.mean([r[metric] for r in results])
            if mean > best_mean:
                best_mean = mean
                best_method = method
        print(f"  {name}: {best_method} ({best_mean:.4f})")

    print("\n" + "=" * 80)
    print("Comparison complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
