#!/usr/bin/env python3
"""
Compare attention weights across different pooling methods on a single scan.

Usage:
    python scripts/compare_attention_methods.py
"""

import ast
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.append('src')
import models


# Configuration
EXPERIMENTS_DIR = '/cluster/tufts/hugheslab/dloevl01/pooling/experiments/RSNA/embedding_level=True'
DATASET_DIR = '/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16'
LABELS_CSV = '/cluster/tufts/hugheslab/datasets/RSNA/labels.csv'
NUMPY_DIR = '/cluster/tufts/hugheslab/datasets/RSNA_numpy'
EMBEDDING_LEVEL = True
SEED = 1001
SCAN_SELECTION_SEED = 42  # For consistent random scan selection
METHODS = ['ABMIL', 'TransMIL', 'SmAP']
OUTPUT_DIR = 'figures'


def find_best_model(experiments_dir, pooling, seed):
    """Find best model by validation AUROC."""
    pattern = os.path.join(experiments_dir, f"*pooling={pooling}*seed={seed}*.csv")
    best_val_auroc, best_file = -1, None

    for csv_file in glob.glob(pattern):
        import pandas as pd
        df = pd.read_csv(csv_file)
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


def select_random_positive_scan(scan_ids, slice_labels, y, seed):
    """Select a random positive scan from test set."""
    pos_indices = [i for i, label in enumerate(y) if label.item() == 1]
    np.random.seed(seed)
    idx = pos_indices[np.random.choice(len(pos_indices))]
    return idx, scan_ids[idx], slice_labels[idx]


def get_mean_attention(length):
    """Get uniform attention weights (mean pooling baseline)."""
    return np.ones(length) / length


def plot_line_comparison(attentions, labels, scan_id, output_dir):
    """Plot line comparison of all methods."""
    slice_nums = np.arange(1, len(labels) + 1)
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Ground truth
    ax1.plot(slice_nums, labels, color='black', linewidth=3.5, alpha=0.8)
    ax1.plot(slice_nums, labels, color='#FF6B6B', linewidth=2, alpha=0.8, label='Ground Truth')
    ax1.set_xlabel('Slice Number', fontsize=12)
    ax1.set_ylabel('Ground Truth', color='#FF6B6B', fontsize=12)
    ax1.set_xlim(1, len(labels))
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis='y', labelcolor='#FF6B6B')

    # Attention weights (normalized)
    ax2 = ax1.twinx()
    colors = {'ABMIL': '#4A90E2', 'TransMIL': '#50C878', 'SmAP': '#9B59B6'}

    for method, attn in attentions.items():
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        ax2.plot(slice_nums, attn_norm, color=colors[method], linewidth=2, alpha=0.7, label=method)

    ax2.set_ylabel('Normalized Attention', fontsize=12)
    ax2.set_ylim(-0.05, 1.05)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{scan_id}_line.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{scan_id}_line.png")
    plt.close()


def plot_grid_visualization(attentions, labels, scan_id, ct_slices, output_dir):
    """Plot grid visualization with CT scans and attention for all methods."""
    n_methods = len(attentions) + 1  # +1 for ground truth
    sample_idx = np.linspace(0, len(labels) - 1, 10, dtype=int)

    fig, axes = plt.subplots(n_methods, 10, figsize=(15, 2 * n_methods))

    # Row 0: Ground truth
    for i, si in enumerate(sample_idx):
        axes[0, i].imshow(ct_slices[si], cmap='gray')
        if labels[si] == 1:
            axes[0, i].imshow(np.ones((*ct_slices[si].shape, 4)) * [1, 0, 0, 0.3])
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[0, i].set_xlabel(si + 1, fontsize=8)
        for spine in axes[0, i].spines.values():
            spine.set_edgecolor('red' if labels[si] == 1 else 'black')
            spine.set_linewidth(2 if labels[si] == 1 else 0.5)
    axes[0, 0].set_ylabel('Ground Truth', fontsize=10, fontweight='bold')

    # Remaining rows: Each method
    for row_idx, (method, attn) in enumerate(attentions.items(), start=1):
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        for i, si in enumerate(sample_idx):
            axes[row_idx, i].imshow(ct_slices[si], cmap='gray')
            overlay = np.ones((*ct_slices[si].shape, 4)) * [1, 0, 0, attn_norm[si] * 0.5]
            axes[row_idx, i].imshow(overlay)
            axes[row_idx, i].set_xticks([])
            axes[row_idx, i].set_yticks([])
            axes[row_idx, i].set_xlabel(si + 1, fontsize=8)

        axes[row_idx, 0].set_ylabel(method, fontsize=10, fontweight='bold')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{scan_id}_grid.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{scan_id}_grid.png")
    plt.close()


def plot_ground_truth_only(labels, scan_id, ct_slices, output_dir):
    """Plot only ground truth CT scans."""
    sample_idx = np.linspace(0, len(labels) - 1, 10, dtype=int)

    fig, axes = plt.subplots(1, 10, figsize=(15, 2))

    for i, si in enumerate(sample_idx):
        axes[i].imshow(ct_slices[si], cmap='gray')
        if labels[si] == 1:
            axes[i].imshow(np.ones((*ct_slices[si].shape, 4)) * [1, 0, 0, 0.3])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_xlabel(si + 1, fontsize=8)
        for spine in axes[i].spines.values():
            spine.set_edgecolor('red' if labels[si] == 1 else 'black')
            spine.set_linewidth(2 if labels[si] == 1 else 0.5)
    axes[0].set_ylabel('Ground Truth', fontsize=10, fontweight='bold')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{scan_id}_ground_truth.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{scan_id}_ground_truth.png")
    plt.close()


def plot_combined(attentions, labels, scan_id, ct_slices, output_dir):
    """Plot combined figure with ground truth scans on top and line plot below."""
    sample_idx = np.linspace(0, len(labels) - 1, 10, dtype=int)
    slice_nums = np.arange(1, len(labels) + 1)

    # Create figure with 2 rows: top for CT scans, bottom for line plot
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.3)

    # Top: Ground truth CT scans
    gs_top = gs[0].subgridspec(1, 10, wspace=0.05)
    for i, si in enumerate(sample_idx):
        ax = fig.add_subplot(gs_top[i])
        ax.imshow(ct_slices[si], cmap='gray')
        if labels[si] == 1:
            ax.imshow(np.ones((*ct_slices[si].shape, 4)) * [1, 0, 0, 0.3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(si + 1, fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('red' if labels[si] == 1 else 'black')
            spine.set_linewidth(2 if labels[si] == 1 else 0.5)

    # Add (a) label
    fig.text(0.02, 0.75, '(a)', fontsize=14, fontweight='bold')

    # Bottom: Line plot
    ax_line = fig.add_subplot(gs[1])

    # Ground truth
    ax_line.plot(slice_nums, labels, color='black', linewidth=3.5, alpha=0.8)
    ax_line.plot(slice_nums, labels, color='#FF6B6B', linewidth=2, alpha=0.8, label='Ground Truth')
    ax_line.set_xlabel('Slice Number', fontsize=12)
    ax_line.set_ylabel('Ground Truth', color='#FF6B6B', fontsize=12)
    ax_line.set_xlim(1, len(labels))
    ax_line.set_ylim(-0.05, 1.05)
    ax_line.tick_params(axis='y', labelcolor='#FF6B6B')

    # Attention weights
    ax2 = ax_line.twinx()
    colors = {'ABMIL': '#4A90E2', 'TransMIL': '#50C878', 'SmAP': '#9B59B6'}

    for method, attn in attentions.items():
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        ax2.plot(slice_nums, attn_norm, color=colors[method], linewidth=2, alpha=0.7, label=method)

    ax2.set_ylabel('Normalized Attention', fontsize=12)
    ax2.set_ylim(-0.05, 1.05)

    # Legend
    lines1, labels1 = ax_line.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_line.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    # Add (b) label
    fig.text(0.02, 0.45, '(b)', fontsize=14, fontweight='bold')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{scan_id}_combined.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{scan_id}_combined.png")
    plt.close()


def main():
    print("=" * 80)
    print("Comparing Attention Across Methods on RSNA Dataset")
    print("=" * 80)
    print()

    # Load data
    test_data = torch.load(f'{DATASET_DIR}/seed={SEED}/test.pth', map_location='cpu', weights_only=False)
    X, lengths, y = test_data['X'], test_data['lengths'], test_data['y']
    scan_ids, slice_labels = get_test_slice_labels(LABELS_CSV, NUMPY_DIR, SEED)

    # Select random positive scan
    idx, scan_id, labels = select_random_positive_scan(scan_ids, slice_labels, y, SCAN_SELECTION_SEED)
    start = sum(lengths[:idx])
    length = lengths[idx]

    print(f"Selected scan: {scan_id}")
    print(f"  {len(labels)} slices, {labels.sum()} positive")
    print()

    # Load CT slices
    ct_data = np.load(f'{NUMPY_DIR}/{scan_id}.npz')['arr_0']
    ct_slices = ct_data[0].transpose(2, 0, 1)
    ct_slices = np.clip(ct_slices, -100, 300)
    ct_slices = (ct_slices + 100) / 400

    # Get attention for each method
    attentions = {}

    # Load methods
    for method in ['ABMIL', 'TransMIL', 'SmAP']:
        model_path, _ = find_best_model(EXPERIMENTS_DIR, method, SEED)
        if model_path:
            model = load_model(model_path, X.shape[1], method, EMBEDDING_LEVEL)
            attention = get_attention(model, X, lengths, EMBEDDING_LEVEL)
            attentions[method] = attention[start:start + length]
            print(f"{method}: loaded")
        else:
            print(f"{method}: not found, skipping")

    print()
    print("Creating visualizations...")

    # Create line plot
    plot_line_comparison(attentions, labels, scan_id, OUTPUT_DIR)

    # Create grid visualization with all methods
    plot_grid_visualization(attentions, labels, scan_id, ct_slices, OUTPUT_DIR)

    # Create ground truth only visualization
    plot_ground_truth_only(labels, scan_id, ct_slices, OUTPUT_DIR)

    # Create combined visualization
    plot_combined(attentions, labels, scan_id, ct_slices, OUTPUT_DIR)

    print()
    print("=" * 80)
    print("Comparison complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
