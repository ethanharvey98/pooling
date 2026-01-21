#!/usr/bin/env python3
"""
Evaluate attention weights against ground truth slice labels.

Computes for each test bag:
1. Sum of attention on positive slices: attention[labels == 1].sum()
2. AUROC of attention vs labels: roc_auc_score(labels, attention)
3. Whether max attention is on a positive slice: labels[argmax(attention)] == 1

Reports mean ± std across 3 seeds.

Usage:
    python evaluate_attention.py \
        --experiments_dir=/path/to/experiments \
        --dataset_dir=/path/to/encoded_data \
        --labels_csv=/path/to/labels.csv \
        --numpy_dir=/path/to/numpy_files \
        --pooling=ABMIL \
        --embedding_level
"""
import argparse
import ast
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import models


def find_best_model(experiments_dir, pooling, seed):
    """Find the best model (by val_auroc) for a given pooling method and seed."""
    pattern = os.path.join(experiments_dir, f"*pooling={pooling}*seed={seed}*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        return None

    best_val_auroc = -1
    best_file = None

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'val_auroc' not in df.columns:
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
                best_file = csv_file.replace('.csv', '.pt')

        except Exception:
            continue

    if best_file and os.path.exists(best_file):
        return best_file
    return None


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
    """Get attention weights from model."""
    with torch.no_grad():
        _, attn_weights = model(embeddings, lengths)

        if embedding_level:
            return attn_weights.squeeze().numpy()
        else:
            per_instance_logits = model.clf(embeddings)
            return torch.sigmoid(per_instance_logits).squeeze().numpy()


def get_test_data(labels_csv, numpy_dir, seed):
    """Get test scan IDs and slice labels."""
    labels_df = pd.read_csv(labels_csv)
    labels_df['scan_label'] = labels_df['Any'].apply(lambda x: 1 if any(ast.literal_eval(x)) else 0)
    labels_df['path'] = labels_df['Study ID'].apply(lambda x: f'{numpy_dir}/{x}.npz')
    labels_df = labels_df[labels_df['path'].apply(os.path.exists)]

    ids, id_labels = labels_df['Study ID'], labels_df['scan_label']
    _, test_ids, _, _ = train_test_split(ids, id_labels, test_size=1/6, random_state=seed, stratify=id_labels)

    test_df = labels_df[labels_df['Study ID'].isin(test_ids)]
    slice_labels = test_df['Any'].apply(lambda x: np.array(ast.literal_eval(x))).values

    return slice_labels


def evaluate_seed(experiments_dir, dataset_dir, labels_csv, numpy_dir, pooling, seed, embedding_level):
    """Evaluate attention for one seed."""
    # Find and load model
    model_path = find_best_model(experiments_dir, pooling, seed)
    if model_path is None:
        return None

    # Load test data
    test_data = torch.load(f'{dataset_dir}/seed={seed}/test.pth', map_location='cpu', weights_only=False)
    X, lengths, y = test_data['X'], test_data['lengths'], test_data['y']

    # Load model
    model = load_model(model_path, X.shape[1], pooling, embedding_level)

    # Get attention weights
    attention = get_attention_weights(model, X, lengths, embedding_level)

    # Get slice labels
    slice_labels = get_test_data(labels_csv, numpy_dir, seed)

    # Calculate metrics for each positive bag
    attn_sum_on_pos = []
    aurocs = []
    max_attn_correct = []

    start_idx = 0
    for i, length in enumerate(lengths):
        end_idx = start_idx + length
        bag_attention = attention[start_idx:end_idx]
        bag_labels = slice_labels[i]

        # Only evaluate positive bags (bags with at least one positive slice)
        if bag_labels.sum() > 0:
            # Metric 1: Sum of attention on positive slices
            attn_sum_on_pos.append(bag_attention[bag_labels == 1].sum())

            # Metric 2: AUROC of attention vs labels
            if len(np.unique(bag_labels)) > 1:  # Need both classes for AUROC
                aurocs.append(roc_auc_score(bag_labels, bag_attention))

            # Metric 3: Is max attention on a positive slice?
            max_attn_correct.append(bag_labels[np.argmax(bag_attention)] == 1)

        start_idx = end_idx

    return {
        'attn_sum_on_pos': np.mean(attn_sum_on_pos),
        'auroc': np.mean(aurocs),
        'max_attn_correct': np.mean(max_attn_correct),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate attention weights')
    parser.add_argument('--experiments_dir', required=True)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--labels_csv', required=True)
    parser.add_argument('--numpy_dir', required=True)
    parser.add_argument('--pooling', required=True, choices=['ABMIL', 'TransMIL', 'SmAP'])
    parser.add_argument('--embedding_level', action='store_true')
    args = parser.parse_args()

    seeds = [1001, 2001, 3001]
    results = []

    print(f"\nEvaluating {args.pooling} ({'embedding' if args.embedding_level else 'logit'} level)\n")

    for seed in seeds:
        result = evaluate_seed(
            args.experiments_dir, args.dataset_dir, args.labels_csv,
            args.numpy_dir, args.pooling, seed, args.embedding_level
        )
        if result:
            results.append(result)
            print(f"Seed {seed}:")
            print(f"  Attention sum on positive slices: {result['attn_sum_on_pos']:.4f}")
            print(f"  AUROC (attention vs labels):      {result['auroc']:.4f}")
            print(f"  Max attention on positive slice:  {result['max_attn_correct']:.4f}")
            print()

    if results:
        print("=" * 50)
        print("SUMMARY (mean ± std across seeds)")
        print("=" * 50)

        for metric in ['attn_sum_on_pos', 'auroc', 'max_attn_correct']:
            values = [r[metric] for r in results]
            mean, std = np.mean(values), np.std(values)
            print(f"{metric}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
