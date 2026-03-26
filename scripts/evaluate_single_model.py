#!/usr/bin/env python3
"""
Evaluate attention metrics for a single model checkpoint on RSNA dataset.

Usage:
    python scripts/evaluate_single_model.py model.pt
    python scripts/evaluate_single_model.py model.pt --method TransMIL --seed 2001
"""

import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from rsna_utils import (
    DATASET_DIR, LABELS_CSV, NUMPY_DIR, EMBEDDING_LEVEL,
    load_model, get_attention, get_test_slice_labels, load_test_data
)


def evaluate_model(model_path, method, seed, embedding_level):
    """Evaluate attention metrics for a single model."""
    X, lengths, y = load_test_data(DATASET_DIR, seed)

    model = load_model(model_path, X.shape[1], method, embedding_level)
    attention = get_attention(model, X, lengths, embedding_level)
    _, slice_labels = get_test_slice_labels(LABELS_CSV, NUMPY_DIR, seed)

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
        'attn_sum': np.mean(attn_sum),
        'auroc': np.mean(aurocs),
        'auprc': np.mean(auprcs),
        'max_correct': np.mean(max_correct),
        'max_attn': np.mean(max_attns)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate attention metrics for a single model')
    parser.add_argument('model_path', type=str, help='Path to .pt model file')
    parser.add_argument('--method', type=str, default='ABMIL',
                        choices=['ABMIL', 'TransMIL', 'SmAP', 'InstanceClassifier'],
                        help='Pooling method (default: ABMIL)')
    parser.add_argument('--seed', type=int, default=1001,
                        help='Random seed for test split (default: 1001)')
    parser.add_argument('--embedding-level', action='store_true', default=True,
                        help='Use embedding-level approach (default: True)')
    args = parser.parse_args()

    print(f"Model: {args.model_path}")
    print(f"Method: {args.method}")
    print(f"Seed: {args.seed}")
    print()

    result = evaluate_model(args.model_path, args.method, args.seed, args.embedding_level)

    print(f"Attention sum on positive: {result['attn_sum']:.4f}")
    print(f"AUROC:                     {result['auroc']:.4f}")
    print(f"AUPRC:                     {result['auprc']:.4f}")
    print(f"Max attention correct:     {result['max_correct']:.4f}")
    print(f"Max attention:             {result['max_attn']:.4f}")


if __name__ == '__main__':
    main()
