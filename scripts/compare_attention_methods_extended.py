#!/usr/bin/env python3
"""
Compare attention weights across different pooling methods on a single scan.
Extended version with GuidedABMIL and BayesianABMIL methods.

Usage:
    python scripts/compare_attention_methods_extended.py
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.append('scripts')
from rsna_utils import (
    EXPERIMENTS_DIR, DATASET_DIR, LABELS_CSV, NUMPY_DIR, EMBEDDING_LEVEL,
    find_best_model, load_model, get_attention, get_test_slice_labels, load_test_data
)


# Script-specific configuration
SEED = 1001
SCAN_SELECTION_SEED = 42  # For consistent random scan selection
OUTPUT_DIR = 'figures_extended'

# Methods that use find_best_model
SEARCH_METHODS = ['ABMIL', 'TransMIL', 'SmAP']

# Methods with fixed paths per seed
FIXED_PATHS = {
    'GuidedABMIL': {
        1001: '/cluster/tufts/hugheslab/eharve06/pooling/experiments/GuidedL1_beta=10.0/alpha=0.0001_criterion=GuidedL1_lr=0.01_pooling=ABMIL_seed=1001.pt',
        2001: '/cluster/tufts/hugheslab/eharve06/pooling/experiments/GuidedL1_beta=10.0/alpha=1e-06_criterion=GuidedL1_lr=0.1_pooling=ABMIL_seed=2001.pt',
        3001: '/cluster/tufts/hugheslab/eharve06/pooling/experiments/GuidedL1_beta=10.0/alpha=1e-05_criterion=GuidedL1_lr=0.1_pooling=ABMIL_seed=3001.pt',
    },
    'BayesianABMIL': {
        1001: '/cluster/tufts/hugheslab/eharve06/pooling/experiments/BayesianABMIL/alpha=0.0001_criterion=L1_lr=0.1_pooling=BayesianABMIL_seed=1001.pt',
        2001: '/cluster/tufts/hugheslab/eharve06/pooling/experiments/BayesianABMIL/alpha=0.001_criterion=L1_lr=0.001_pooling=BayesianABMIL_seed=2001.pt',
        3001: '/cluster/tufts/hugheslab/eharve06/pooling/experiments/BayesianABMIL/alpha=0.0001_criterion=L1_lr=0.1_pooling=BayesianABMIL_seed=3001.pt',
    },
}

# All methods to plot (order matters for legend)
ALL_METHODS = ['ABMIL', 'TransMIL', 'SmAP', 'GuidedABMIL', 'BayesianABMIL']


def select_random_positive_scan(scan_ids, slice_labels, y, seed):
    """Select a random positive scan from test set."""
    pos_indices = [i for i, label in enumerate(y) if label.item() == 1]
    np.random.seed(seed)
    idx = pos_indices[np.random.choice(len(pos_indices))]
    return idx, scan_ids[idx], slice_labels[idx]


def plot_combined(attentions, labels, scan_id, ct_slices, output_dir):
    """Plot combined figure with ground truth scans on top and line plot below."""
    # Save and restore rcParams to avoid affecting other plots
    original_font_size = plt.rcParams.get('font.size', 10)
    plt.rcParams.update({"font.size": 10})

    n_slices = len(labels)
    ncols, nrows = 10, 2

    fig = plt.figure(figsize=(1 * ncols, 2 * nrows))
    gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, hspace=0.3)
    axs_top = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    ax_bottom = fig.add_subplot(gs[1, :])

    # Select 10 evenly spaced slices (1-indexed like the mockup)
    indices = np.round(np.linspace(start=1, stop=n_slices, num=ncols)).astype(int)

    # Top row: CT slices with red borders on positive slices
    for i, j in enumerate(indices):
        axs_top[i].imshow(ct_slices[j - 1], cmap='gray')
        axs_top[i].set_xlabel(rf"${j}$")
        axs_top[i].set_xticks([])
        axs_top[i].set_yticks([])

        for spine in axs_top[i].spines.values():
            if labels[j - 1] == 1:
                spine.set_visible(True)
                spine.set_color("#D62728")
                spine.set_linewidth(2)
            else:
                spine.set_visible(False)

    # Bottom plot: attention weights comparison
    slice_nums = np.arange(1, n_slices + 1)

    # Ground truth normalized to sum to 1
    gt_normalized = labels / (np.sum(labels) + 1e-8)
    ax_bottom.plot(slice_nums, gt_normalized, color="#1F77B4", label="Ground truth", linewidth=3)
    ax_bottom.fill_between(slice_nums, gt_normalized, alpha=0.3, color="#1F77B4")

    # Attention weights (already normalized via softmax in the model)
    colors = {
        'ABMIL': '#D62728',
        'TransMIL': '#9467BD',
        'SmAP': '#8C564B',
        'GuidedABMIL': '#7F7F7F',
        'BayesianABMIL': '#E377C2',
    }
    for method, attn in attentions.items():
        ax_bottom.plot(slice_nums, attn, color=colors[method], label=method, linewidth=3)

    ax_bottom.set_xlabel(r"Slice index $j$")
    ax_bottom.set_ylabel(r"Attention $a_{ij}$")
    ax_bottom.set_xlim(1, n_slices)
    ax_bottom.legend(loc="upper right")
    ax_bottom.grid()

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{scan_id}_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{scan_id}_combined.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir}/{scan_id}_combined.png")
    print(f"Saved: {output_dir}/{scan_id}_combined.pdf")
    plt.close()

    # Restore original font size
    plt.rcParams.update({"font.size": original_font_size})


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


def plot_line_comparison(attentions, labels, scan_id, output_dir):
    """Plot line comparison of all methods."""
    slice_nums = np.arange(1, len(labels) + 1)
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Ground truth
    ax1.plot(slice_nums, labels, color='#FF6B6B', linewidth=2.5, label='Ground Truth')
    ax1.set_xlabel('Slice Number', fontsize=12)
    ax1.set_ylabel('Ground Truth', color='#FF6B6B', fontsize=12)
    ax1.set_xlim(1, len(labels))
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis='y', labelcolor='#FF6B6B')

    # Attention weights (normalized)
    ax2 = ax1.twinx()
    colors = {
        'ABMIL': '#D62728',
        'TransMIL': '#9467BD',
        'SmAP': '#8C564B',
        'GuidedABMIL': '#7F7F7F',
        'BayesianABMIL': '#E377C2',
    }

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


def main():
    print("=" * 80)
    print("Comparing Attention Across Methods (Extended)")
    print("=" * 80)
    print()

    # Load data
    X, lengths, y = load_test_data(DATASET_DIR, SEED)
    scan_ids, slice_labels = get_test_slice_labels(LABELS_CSV, NUMPY_DIR, SEED)

    # Select random positive scan
    idx, scan_id, labels = select_random_positive_scan(scan_ids, slice_labels, y, SCAN_SELECTION_SEED)
    labels = np.array(labels)
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

    # Load methods that use search
    for method in SEARCH_METHODS:
        model_path, _ = find_best_model(EXPERIMENTS_DIR, method, SEED)
        if model_path:
            model = load_model(model_path, X.shape[1], method, EMBEDDING_LEVEL)
            attention = get_attention(model, X, lengths, EMBEDDING_LEVEL)
            attentions[method] = attention[start:start + length]
            print(f"{method}: loaded from {model_path}")
        else:
            print(f"{method}: not found, skipping")

    # Load methods with fixed paths
    for method, seed_paths in FIXED_PATHS.items():
        if SEED in seed_paths:
            model_path = seed_paths[SEED]
            if os.path.exists(model_path):
                # GuidedABMIL uses ABMIL architecture
                arch = 'ABMIL' if method == 'GuidedABMIL' else method
                model = load_model(model_path, X.shape[1], arch, EMBEDDING_LEVEL)
                attention = get_attention(model, X, lengths, EMBEDDING_LEVEL)
                attentions[method] = attention[start:start + length]
                print(f"{method}: loaded from {model_path}")
            else:
                print(f"{method}: path not found: {model_path}")
        else:
            print(f"{method}: no path for seed {SEED}")

    print()
    print("Creating visualizations...")

    # Create combined visualization
    plot_combined(attentions, labels, scan_id, ct_slices, OUTPUT_DIR)

    # Create grid visualization
    plot_grid_visualization(attentions, labels, scan_id, ct_slices, OUTPUT_DIR)

    # Create line plot
    plot_line_comparison(attentions, labels, scan_id, OUTPUT_DIR)

    print()
    print("=" * 80)
    print("Comparison complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
