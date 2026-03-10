#!/usr/bin/env python3
"""
Compare VAP attention weights across methods on a single scan, with uncertainty bands.

Usage:
    python scripts/compare_vap_attention_methods.py
    python scripts/compare_vap_attention_methods.py --seed 2001 --mc_samples 30
"""

import argparse
import ast
import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import models


# Defaults
EXPERIMENTS_DIR = '/cluster/tufts/hugheslab/dloevl01/Feb_2026/March_2026/VAP_Tests/pooling'
DATASET_DIR = '/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16'
LABELS_CSV = '/cluster/tufts/hugheslab/datasets/RSNA/labels.csv'
NUMPY_DIR = '/cluster/tufts/hugheslab/datasets/RSNA_numpy'
SEED = 1001
SCAN_SELECTION_SEED = 42
MODEL_TYPES = ['VAPGaussian', 'VAPGaussianCenter', 'VAPBernoulli', 'VAPGaussianSparse']
OUTPUT_DIR = 'figures/vap_attention'


def find_best_model(experiments_dir, model_type, seed):
    """Find best model .pt file by val_auroc."""
    pattern = os.path.join(experiments_dir, f"model={model_type}_*seed={seed}*.csv")
    best_val_auroc, best_file = -1, None
    for csv_file in glob.glob(pattern):
        try:
            df = pd.read_csv(csv_file)
        except Exception:
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


def load_test_data(dataset_dir, seed):
    test_data = torch.load(f'{dataset_dir}/seed={seed}/test.pth', map_location='cpu', weights_only=False)
    return test_data['X'], test_data['lengths'], test_data['y']


def get_test_slice_labels(labels_csv, numpy_dir, seed):
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


def get_attention_with_uncertainty(model, X, lengths, mc_samples):
    """Get MC attention samples: returns (mc_samples, N_total) array."""
    unc = model.predict_with_uncertainty(X, lengths, n_samples=mc_samples)
    # attn_samples shape: (mc_samples, N_total, 1)
    return unc['attn_samples'].squeeze(-1).numpy()


def get_deterministic_attention(model, X, lengths):
    """Get deterministic attention weights."""
    with torch.no_grad():
        _, attn = model(X, lengths)
    return attn.squeeze(-1).numpy()


def select_random_positive_scan(scan_ids, slice_labels, y, seed):
    pos_indices = [i for i, label in enumerate(y) if label.item() == 1]
    np.random.seed(seed)
    idx = pos_indices[np.random.choice(len(pos_indices))]
    return idx, scan_ids[idx], slice_labels[idx]


def plot_combined_with_uncertainty(attentions, uncertainties, labels, scan_id, ct_slices, output_dir):
    """Combined figure: CT slices on top, attention lines with uncertainty bands below."""
    original_font_size = plt.rcParams.get('font.size', 10)
    plt.rcParams.update({"font.size": 10})

    n_slices = len(labels)
    ncols, nrows = 10, 2

    fig = plt.figure(figsize=(1 * ncols, 2 * nrows))
    gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, hspace=0.3)
    axs_top = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    ax_bottom = fig.add_subplot(gs[1, :])

    indices = np.round(np.linspace(start=1, stop=n_slices, num=ncols)).astype(int)

    # Top row: CT slices
    for i, j in enumerate(indices):
        axs_top[i].imshow(ct_slices[j - 1], cmap='gray')
        axs_top[i].set_xlabel(rf"${j}$")
        axs_top[i].set_xticks([])
        axs_top[i].set_yticks([])
        for spine in axs_top[i].spines.values():
            if labels[j - 1] == 1:
                spine.set_visible(True)
                spine.set_color("#1F77B4")
                spine.set_linewidth(2)
            else:
                spine.set_visible(False)

    # Bottom plot: attention with uncertainty bands
    slice_nums = np.arange(1, n_slices + 1)

    # Ground truth normalized
    gt_normalized = labels / (np.sum(labels) + 1e-8)
    ax_bottom.plot(slice_nums, gt_normalized, color="#1F77B4", label="Ground truth", linewidth=3)
    ax_bottom.fill_between(slice_nums, gt_normalized, alpha=0.3, color="#1F77B4")

    colors = {
        'VAPGaussian': '#D62728',
        'VAPGaussianCenter': '#FF7F0E',
        'VAPBernoulli': '#9467BD',
        'VAPGaussianSparse': '#8C564B',
    }

    for method, attn_mean in attentions.items():
        color = colors.get(method, '#333333')
        ax_bottom.plot(slice_nums, attn_mean, color=color, label=method, linewidth=2)
        if method in uncertainties:
            attn_std = uncertainties[method]
            ax_bottom.fill_between(
                slice_nums,
                attn_mean - attn_std,
                attn_mean + attn_std,
                alpha=0.2, color=color,
            )

    ax_bottom.set_xlabel(r"Slice index $j$")
    ax_bottom.set_ylabel(r"Attention $a_{ij}$")
    ax_bottom.set_xlim(1, n_slices)
    ax_bottom.legend(loc="upper right", fontsize=8)
    ax_bottom.grid()

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{scan_id}_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{scan_id}_combined.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir}/{scan_id}_combined.png")
    print(f"Saved: {output_dir}/{scan_id}_combined.pdf")
    plt.close()
    plt.rcParams.update({"font.size": original_font_size})


def plot_line_with_uncertainty(attentions, uncertainties, labels, scan_id, output_dir):
    """Line plot with twin axes: ground truth + normalized attention with uncertainty bands."""
    slice_nums = np.arange(1, len(labels) + 1)
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Ground truth
    ax1.plot(slice_nums, labels, color='#FF6B6B', linewidth=2.5, label='Ground Truth')
    ax1.set_xlabel('Slice Number', fontsize=12)
    ax1.set_ylabel('Ground Truth', color='#FF6B6B', fontsize=12)
    ax1.set_xlim(1, len(labels))
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis='y', labelcolor='#FF6B6B')

    ax2 = ax1.twinx()
    colors = {
        'VAPGaussian': '#D62728',
        'VAPGaussianCenter': '#FF7F0E',
        'VAPBernoulli': '#9467BD',
        'VAPGaussianSparse': '#8C564B',
    }

    for method, attn in attentions.items():
        color = colors.get(method, '#333333')
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        ax2.plot(slice_nums, attn_norm, color=color, linewidth=2, alpha=0.7, label=method)

        if method in uncertainties:
            std = uncertainties[method]
            std_norm = std / (attn.max() - attn.min() + 1e-8)
            ax2.fill_between(
                slice_nums,
                np.clip(attn_norm - std_norm, 0, 1),
                np.clip(attn_norm + std_norm, 0, 1),
                alpha=0.15, color=color,
            )

    ax2.set_ylabel('Normalized Attention', fontsize=12)
    ax2.set_ylim(-0.05, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{scan_id}_line.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{scan_id}_line.png")
    plt.close()


def plot_grid_with_uncertainty(attentions, uncertainties, labels, scan_id, ct_slices, output_dir):
    """Grid: CT slices with attention heatmap overlay, plus uncertainty bar below each row."""
    n_methods = len(attentions) + 1  # +1 for ground truth row
    sample_idx = np.linspace(0, len(labels) - 1, 10, dtype=int)

    fig, axes = plt.subplots(n_methods * 2 - 1, 10, figsize=(15, 1.5 * (n_methods * 2 - 1)),
                             gridspec_kw={'height_ratios': [2] + [2, 1] * (n_methods - 1) + [2]})
    # Simpler approach: just use n_methods rows, uncertainty as bar thickness
    plt.close()

    # Simpler grid: one row per method + ground truth
    fig, axes = plt.subplots(n_methods, 10, figsize=(15, 2 * n_methods))
    if n_methods == 1:
        axes = axes[np.newaxis, :]

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

    for row_idx, (method, attn) in enumerate(attentions.items(), start=1):
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        has_unc = method in uncertainties
        if has_unc:
            std = uncertainties[method]
            std_norm = std / (attn.max() - attn.min() + 1e-8)

        for i, si in enumerate(sample_idx):
            axes[row_idx, i].imshow(ct_slices[si], cmap='gray')
            # Overlay attention as red alpha
            overlay = np.ones((*ct_slices[si].shape, 4)) * [1, 0, 0, attn_norm[si] * 0.5]
            axes[row_idx, i].imshow(overlay)
            axes[row_idx, i].set_xticks([])
            axes[row_idx, i].set_yticks([])
            # Show uncertainty as border thickness (thicker = more uncertain)
            if has_unc:
                lw = 1 + std_norm[si] * 8  # scale border width by uncertainty
                for spine in axes[row_idx, i].spines.values():
                    spine.set_edgecolor('#FFD700')
                    spine.set_linewidth(lw)
            axes[row_idx, i].set_xlabel(si + 1, fontsize=8)

        axes[row_idx, 0].set_ylabel(method, fontsize=10, fontweight='bold')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{scan_id}_grid.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{scan_id}_grid.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare VAP attention methods with uncertainty')
    parser.add_argument('--experiments_dir', default=EXPERIMENTS_DIR, type=str)
    parser.add_argument('--dataset_dir', default=DATASET_DIR, type=str)
    parser.add_argument('--labels_csv', default=LABELS_CSV, type=str)
    parser.add_argument('--numpy_dir', default=NUMPY_DIR, type=str)
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--scan_seed', default=SCAN_SELECTION_SEED, type=int)
    parser.add_argument('--mc_samples', default=20, type=int)
    parser.add_argument('--output_dir', default=OUTPUT_DIR, type=str)
    parser.add_argument('--methods', nargs='+', default=MODEL_TYPES)
    args = parser.parse_args()

    print("=" * 80)
    print("VAP Attention Comparison with Uncertainty")
    print(f"  MC samples: {args.mc_samples}")
    print("=" * 80)

    X, lengths, y = load_test_data(args.dataset_dir, args.seed)
    scan_ids, slice_labels = get_test_slice_labels(args.labels_csv, args.numpy_dir, args.seed)

    idx, scan_id, labels = select_random_positive_scan(scan_ids, slice_labels, y, args.scan_seed)
    start = sum(lengths[:idx])
    length = lengths[idx]

    print(f"\nSelected scan: {scan_id}")
    print(f"  {len(labels)} slices, {labels.sum()} positive\n")

    # Load CT slices
    ct_data = np.load(f'{args.numpy_dir}/{scan_id}.npz')['arr_0']
    ct_slices = ct_data[0].transpose(2, 0, 1)
    ct_slices = np.clip(ct_slices, -100, 300)
    ct_slices = (ct_slices + 100) / 400

    attentions = {}
    uncertainties = {}

    for method in args.methods:
        model_path, val_auroc = find_best_model(args.experiments_dir, method, args.seed)
        if not model_path or not os.path.exists(model_path):
            print(f"{method}: not found, skipping")
            continue

        model = load_vap_model(model_path, X.shape[1], method)

        # MC samples for uncertainty
        attn_samples = get_attention_with_uncertainty(model, X, lengths, args.mc_samples)
        # Extract this scan's attention: (mc_samples, length)
        scan_attn = attn_samples[:, start:start + length]
        attn_mean = scan_attn.mean(axis=0)
        attn_std = scan_attn.std(axis=0)

        attentions[method] = attn_mean
        uncertainties[method] = attn_std

        print(f"{method}: loaded (val_auroc={val_auroc:.4f})")
        print(f"  mean attn std: {attn_std.mean():.6f}, max attn std: {attn_std.max():.6f}")

    if not attentions:
        print("No models found. Exiting.")
        return

    print("\nCreating visualizations...")

    plot_combined_with_uncertainty(attentions, uncertainties, labels, scan_id, ct_slices, args.output_dir)
    plot_line_with_uncertainty(attentions, uncertainties, labels, scan_id, args.output_dir)
    plot_grid_with_uncertainty(attentions, uncertainties, labels, scan_id, ct_slices, args.output_dir)

    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80)


if __name__ == '__main__':
    main()
