#!/usr/bin/env python3
"""
Visualize attention weights or logit-level probabilities against true slice labels.

For embedding-level models (ABMIL, TransMIL, SmAP): shows attention weights
For logit-level models: shows sigmoid probabilities

Usage:
    python visualize_attention.py \
        --experiments_dir=/path/to/experiments \
        --dataset_dir=/path/to/encoded_data \
        --labels_csv=/path/to/labels.csv \
        --numpy_dir=/path/to/numpy_files \
        --pooling=ABMIL \
        --seed=1001 \
        --embedding_level \
        --output=attention_viz.png
"""
import argparse
import ast
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import models


def find_best_model(experiments_dir, pooling, seed):
    """Find the best model (by val_auroc) for a given pooling method and seed."""
    pattern = os.path.join(experiments_dir, f"*pooling={pooling}*seed={seed}*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        raise ValueError(f"No CSV files found matching pattern: {pattern}")

    best_val_auroc = -1
    best_file = None
    best_epoch = None

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'val_auroc' not in df.columns:
                continue

            # Filter to only rows where val_auroc <= train_auroc (not overfitting)
            if 'train_auroc' in df.columns:
                valid_df = df[df['val_auroc'] <= df['train_auroc']]
                if valid_df.empty:
                    continue
            else:
                valid_df = df

            idx = valid_df['val_auroc'].idxmax()
            val_auroc = df.loc[idx, 'val_auroc']
            epoch = df.loc[idx, 'epoch'] if 'epoch' in df.columns else idx

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_file = csv_file.replace('.csv', '.pt')
                best_epoch = epoch

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

    if best_file is None or not os.path.exists(best_file):
        raise ValueError(f"No valid model found for pooling={pooling}, seed={seed}")

    print(f"Best model: {os.path.basename(best_file)}")
    print(f"  Val AUROC: {best_val_auroc:.4f}, Epoch: {best_epoch}")

    return best_file


def load_model(model_path, in_features, pooling, embedding_level):
    """Load a trained model from checkpoint."""
    if embedding_level:
        model = models.PoolClf(in_features=in_features, out_features=1, pooling=pooling)
    else:
        model = models.ClfPool(in_features=in_features, out_features=1, pooling=pooling)

    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_attention_weights(model, embeddings, lengths, embedding_level):
    """Get attention weights or probabilities from model."""
    with torch.no_grad():
        logits, attn_weights = model(embeddings, lengths)

        if embedding_level:
            # For embedding-level, attn_weights are the attention values
            return attn_weights.squeeze().numpy()
        else:
            # For logit-level, use sigmoid of the per-instance logits before pooling
            per_instance_logits = model.clf(embeddings)
            probs = torch.sigmoid(per_instance_logits).squeeze().numpy()
            return probs


def recreate_test_split(labels_csv, numpy_dir, seed):
    """Recreate the train/val/test split to get test scan IDs in order."""
    labels_df = pd.read_csv(labels_csv)

    # Same logic as encode_rsna_full.py
    labels_df['scan_label'] = labels_df['Any'].apply(lambda x: 1 if any(ast.literal_eval(x)) else 0)
    labels_df['path'] = labels_df['Study ID'].apply(lambda x: f'{numpy_dir}/{x}.npz')

    # Filter to only scans with existing npz files
    labels_df = labels_df[labels_df['path'].apply(os.path.exists)]

    # Recreate the split
    ids, id_labels = labels_df['Study ID'], labels_df['scan_label']
    train_and_val_ids, test_ids, _, _ = train_test_split(
        ids, id_labels, test_size=1/6, random_state=seed, stratify=id_labels
    )

    test_df = labels_df[labels_df['Study ID'].isin(test_ids)]

    # Return in the same order as the test_ids (which matches encoding order)
    test_scan_ids = test_df['Study ID'].values
    test_slice_labels = test_df['Any'].apply(lambda x: ast.literal_eval(x)).values

    return test_scan_ids, test_slice_labels


def find_positive_window(slice_labels, context=3):
    """Find a continuous section of positive slices with context on both ends."""
    positive_indices = np.where(np.array(slice_labels) == 1)[0]

    if len(positive_indices) == 0:
        return None, None

    # Find the longest continuous run of positives
    runs = []
    start = positive_indices[0]
    end = positive_indices[0]

    for i in range(1, len(positive_indices)):
        if positive_indices[i] == positive_indices[i-1] + 1:
            end = positive_indices[i]
        else:
            runs.append((start, end))
            start = positive_indices[i]
            end = positive_indices[i]
    runs.append((start, end))

    # Get the longest run
    longest_run = max(runs, key=lambda x: x[1] - x[0])

    # Add context
    window_start = max(0, longest_run[0] - context)
    window_end = min(len(slice_labels), longest_run[1] + context + 1)

    return window_start, window_end


def visualize_scan(slice_labels, attention_weights, scan_id, output_path, window_start, window_end):
    """Create visualization with two rows: ground truth and attention heatmap."""
    # Extract window
    window_labels = slice_labels[window_start:window_end]
    window_attention = attention_weights[window_start:window_end]

    # Renormalize attention for this window
    window_attention_norm = (window_attention - window_attention.min()) / (window_attention.max() - window_attention.min() + 1e-8)

    n_slices = len(window_labels)

    fig, axes = plt.subplots(2, 1, figsize=(max(12, n_slices * 0.8), 4))

    # Row 1: Ground truth labels (shaded red for positive)
    ax1 = axes[0]
    for i, label in enumerate(window_labels):
        color = 'red' if label == 1 else 'lightgray'
        alpha = 0.3 if label == 1 else 0.3
        ax1.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5, alpha=alpha))

    ax1.set_xlim(0, n_slices)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_xticks(np.arange(n_slices) + 0.5)
    ax1.set_xticklabels([str(window_start + i + 1) for i in range(n_slices)], fontsize=8)
    ax1.set_yticks([])
    ax1.set_title('Ground Truth (Red = Positive Slice)', fontsize=12)

    # Row 2: Attention heatmap
    ax2 = axes[1]
    cmap = plt.cm.Reds
    for i, attn in enumerate(window_attention_norm):
        color = cmap(attn)
        ax2.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5))

    ax2.set_xlim(0, n_slices)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.set_xticks(np.arange(n_slices) + 0.5)
    ax2.set_xticklabels([str(window_start + i + 1) for i in range(n_slices)], fontsize=8)
    ax2.set_yticks([])
    ax2.set_title('Attention Weights (Normalized for Window)', fontsize=12)

    # Add colorbar for attention
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Attention', fontsize=10)

    plt.suptitle(f'Scan: {scan_id} (Slices {window_start+1}-{window_end})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize attention weights vs ground truth')
    parser.add_argument('--experiments_dir', required=True, help='Directory with experiment results')
    parser.add_argument('--dataset_dir', required=True, help='Directory with encoded dataset (without seed)')
    parser.add_argument('--labels_csv', required=True, help='Path to labels.csv with slice-level labels')
    parser.add_argument('--numpy_dir', required=True, help='Directory with numpy files')
    parser.add_argument('--pooling', required=True, choices=['ABMIL', 'TransMIL', 'SmAP'], help='Pooling method')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--embedding_level', action='store_true', help='Use embedding-level model')
    parser.add_argument('--output', default='attention_viz.png', help='Output image path')
    parser.add_argument('--scan_idx', type=int, default=0, help='Index of positive scan to visualize (among positive test scans)')
    parser.add_argument('--context', type=int, default=3, help='Number of context slices on each side')
    args = parser.parse_args()

    # Find and load the best model
    model_path = find_best_model(args.experiments_dir, args.pooling, args.seed)

    # Load test data
    test_data = torch.load(f'{args.dataset_dir}/seed={args.seed}/test.pth', map_location='cpu', weights_only=False)
    X = test_data['X']
    lengths = test_data['lengths']
    y = test_data['y']

    in_features = X.shape[1]

    # Load model
    model = load_model(model_path, in_features, args.pooling, args.embedding_level)

    # Get attention weights for all test samples
    attention_weights = get_attention_weights(model, X, lengths, args.embedding_level)

    # Recreate the test split to get scan IDs and slice-level labels
    test_scan_ids, test_slice_labels = recreate_test_split(args.labels_csv, args.numpy_dir, args.seed)

    # Find positive scans in test set
    positive_scan_indices = [i for i, label in enumerate(y) if label.item() == 1]
    print(f"Found {len(positive_scan_indices)} positive scans in test set")

    if args.scan_idx >= len(positive_scan_indices):
        print(f"Only {len(positive_scan_indices)} positive scans, using index 0")
        args.scan_idx = 0

    scan_idx = positive_scan_indices[args.scan_idx]
    scan_id = test_scan_ids[scan_idx]
    slice_labels = np.array(test_slice_labels[scan_idx])

    # Get the slice range for this scan in the flattened attention array
    start_idx = sum(lengths[:scan_idx])
    end_idx = start_idx + lengths[scan_idx]
    scan_attention = attention_weights[start_idx:end_idx]

    print(f"\nVisualizing scan {scan_id} (index {scan_idx})")
    print(f"  {lengths[scan_idx]} slices, {slice_labels.sum()} positive")
    print(f"  Attention range: [{scan_attention.min():.4f}, {scan_attention.max():.4f}]")

    # Find window around positive slices
    window_start, window_end = find_positive_window(slice_labels, context=args.context)

    if window_start is not None:
        visualize_scan(slice_labels, scan_attention, scan_id, args.output, window_start, window_end)
    else:
        print("No positive slices found in this scan (unexpected for positive scan)")


if __name__ == "__main__":
    main()
