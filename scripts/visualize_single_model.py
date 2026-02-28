#!/usr/bin/env python3
"""
Visualize attention weights for a single model checkpoint.

Similar to compare_attention_methods.py but takes a specific model file path
like evaluate_single_model.py does.

Usage:
    python scripts/visualize_single_model.py model.pt
    python scripts/visualize_single_model.py model.pt --method SmAP --seed 1001
    python scripts/visualize_single_model.py model.pt --scan-id ID_abc123
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.append('scripts')
from rsna_utils import (
    DATASET_DIR, LABELS_CSV, NUMPY_DIR, EMBEDDING_LEVEL,
    load_model, get_attention, get_test_slice_labels, load_test_data
)


def select_random_positive_scan(scan_ids, slice_labels, y, seed):
    """Select a random positive scan from test set."""
    pos_indices = [i for i, label in enumerate(y) if label.item() == 1]
    np.random.seed(seed)
    idx = pos_indices[np.random.choice(len(pos_indices))]
    return idx, scan_ids[idx], slice_labels[idx]


def find_scan_by_id(scan_ids, slice_labels, target_id):
    """Find scan by ID."""
    for idx, scan_id in enumerate(scan_ids):
        if scan_id == target_id:
            return idx, scan_id, slice_labels[idx]
    raise ValueError(f"Scan ID {target_id} not found in test set")


def plot_combined(attention, labels, scan_id, ct_slices, output_path, method):
    """Plot combined figure with ground truth scans on top and line plot below.

    Exact visual from compare_attention_methods.py plot_combined function.
    """
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

    # Top row: CT slices with blue borders on positive slices
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
    # Use same color scheme as compare_attention_methods.py
    colors = {'ABMIL': '#9467BD', 'TransMIL': '#8C564B', 'SmAP': '#E377C2', 'BayesianABMIL': '#17BECF'}
    color = colors.get(method, '#9467BD')
    ax_bottom.plot(slice_nums, attention, color=color, label=method, linewidth=3)

    ax_bottom.set_xlabel(r"Slice index $j$")
    ax_bottom.set_ylabel(r"Attention $a_{ij}$")
    ax_bottom.set_xlim(1, n_slices)
    ax_bottom.legend(loc="upper right")
    ax_bottom.grid()

    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")
    plt.close()

    # Restore original font size
    plt.rcParams.update({"font.size": original_font_size})


def plot_grid(attention, labels, scan_id, ct_slices, output_path, method):
    """Plot grid visualization with CT scans and attention overlay."""
    sample_idx = np.linspace(0, len(labels) - 1, 10, dtype=int)
    attn_norm = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

    fig, axes = plt.subplots(2, 10, figsize=(15, 4))

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

    # Row 1: Attention overlay
    for i, si in enumerate(sample_idx):
        axes[1, i].imshow(ct_slices[si], cmap='gray')
        overlay = np.ones((*ct_slices[si].shape, 4)) * [1, 0, 0, attn_norm[si] * 0.5]
        axes[1, i].imshow(overlay)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        axes[1, i].set_xlabel(si + 1, fontsize=8)
    axes[1, 0].set_ylabel(method, fontsize=10, fontweight='bold')

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_line(attention, labels, scan_id, output_path, method):
    """Plot line comparison of ground truth and attention."""
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
    attn_norm = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    ax2.plot(slice_nums, attn_norm, color='#4A90E2', linewidth=2, alpha=0.7, label=method)
    ax2.set_ylabel('Normalized Attention', fontsize=12)
    ax2.set_ylim(-0.05, 1.05)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.title(f'Scan: {scan_id}', fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize attention weights for a single model checkpoint'
    )
    parser.add_argument('model_path', type=str, help='Path to .pt model file')
    parser.add_argument('--method', type=str, default='ABMIL',
                        choices=['ABMIL', 'TransMIL', 'SmAP', 'BayesianABMIL'],
                        help='Pooling method (default: ABMIL)')
    parser.add_argument('--seed', type=int, default=1001,
                        help='Random seed for test split (default: 1001)')
    parser.add_argument('--scan-id', type=str, default=None,
                        help='Specific scan ID to visualize (default: random positive)')
    parser.add_argument('--scan-seed', type=int, default=42,
                        help='Seed for random scan selection (default: 42)')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Output directory for figures (default: figures)')
    parser.add_argument('--embedding-level', action='store_true', default=True,
                        help='Use embedding-level approach (default: True)')
    parser.add_argument('--dataset-dir', type=str, default=DATASET_DIR,
                        help=f'Dataset directory (default: {DATASET_DIR})')
    parser.add_argument('--labels-csv', type=str, default=LABELS_CSV,
                        help=f'Labels CSV path (default: {LABELS_CSV})')
    parser.add_argument('--numpy-dir', type=str, default=NUMPY_DIR,
                        help=f'Numpy directory (default: {NUMPY_DIR})')
    args = parser.parse_args()

    print("=" * 80)
    print("Visualizing Attention Weights")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Method: {args.method}")
    print(f"Seed: {args.seed}")
    print()

    # Load test data
    X, lengths, y = load_test_data(args.dataset_dir, args.seed)
    scan_ids, slice_labels = get_test_slice_labels(args.labels_csv, args.numpy_dir, args.seed)

    # Select scan
    if args.scan_id:
        idx, scan_id, labels = find_scan_by_id(scan_ids, slice_labels, args.scan_id)
    else:
        idx, scan_id, labels = select_random_positive_scan(scan_ids, slice_labels, y, args.scan_seed)

    labels = np.array(labels)
    print(f"Selected scan: {scan_id}")
    print(f"  {len(labels)} slices, {labels.sum()} positive")
    print()

    # Calculate attention slice range
    start = sum(lengths[:idx])
    length = lengths[idx]

    # Load CT slices
    ct_data = np.load(f'{args.numpy_dir}/{scan_id}.npz')['arr_0']
    ct_slices = ct_data[0].transpose(2, 0, 1)
    ct_slices = np.clip(ct_slices, -100, 300)
    ct_slices = (ct_slices + 100) / 400

    # Load model and get attention
    model = load_model(args.model_path, X.shape[1], args.method, args.embedding_level)
    attention = get_attention(model, X, lengths, args.embedding_level)
    scan_attention = attention[start:start + length]

    print(f"Attention shape: {scan_attention.shape}")
    print(f"Attention range: [{scan_attention.min():.4f}, {scan_attention.max():.4f}]")
    print()

    # Create visualizations
    print("Creating visualizations...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Combined plot (like compare_attention_methods.py)
    plot_combined(
        scan_attention, labels, scan_id, ct_slices,
        f'{args.output_dir}/{scan_id}_{args.method}_combined.png',
        args.method
    )

    # Grid plot
    plot_grid(
        scan_attention, labels, scan_id, ct_slices,
        f'{args.output_dir}/{scan_id}_{args.method}_grid.png',
        args.method
    )

    # Line plot
    plot_line(
        scan_attention, labels, scan_id,
        f'{args.output_dir}/{scan_id}_{args.method}_line.png',
        args.method
    )

    print()
    print("=" * 80)
    print("Visualization complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
