#!/usr/bin/env python3
"""
Evaluate attention metrics for a single model checkpoint on RSNA dataset.

Usage:
    python scripts/evaluate_single_model.py model.pt
    python scripts/evaluate_single_model.py model.pt --method BernoulliVAP --seed 2001
    python scripts/evaluate_single_model.py model.pt --method BernoulliVAP --supervised --seed 1001
"""

import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from rsna_utils import (
    DATASET_DIR, DATASET_DIR_WITH_LABELS, LABELS_CSV, NUMPY_DIR,
    load_model, get_attention, get_test_slice_labels,
    load_test_data, load_test_data_with_instance_labels
)


def evaluate_model(model_path, method, seed, embedding_level, supervised):
    if supervised:
        X, lengths, y, instance_labels = load_test_data_with_instance_labels(DATASET_DIR_WITH_LABELS, seed)
        slice_labels_from_data = instance_labels
    else:
        X, lengths, y = load_test_data(DATASET_DIR, seed)
        slice_labels_from_data = None

    pool_kwargs = dict(pi_max=0.5, sigma=0.25, tau=0.5) if method == 'BernoulliVAP' else {}
    model = load_model(model_path, X.shape[1], method, embedding_level, **pool_kwargs)
    attention = get_attention(model, X, lengths, embedding_level)

    # Get ground truth slice labels
    if slice_labels_from_data is not None:
        gt_slice_labels = [np.array(sl) for sl in slice_labels_from_data]
    else:
        _, gt_slice_labels = get_test_slice_labels(LABELS_CSV, NUMPY_DIR, seed)

    attn_sum, aurocs, auprcs, max_correct, max_attns = [], [], [], [], []
    start = 0
    for i, length in enumerate(lengths):
        attn = attention[start:start + length]
        labels = gt_slice_labels[i]
        # Truncate to match if needed
        n = min(len(attn), len(labels))
        attn, labels = attn[:n], labels[:n]
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
                        choices=['ABMIL', 'TransMIL', 'SmAP', 'BernoulliVAP'],
                        help='Pooling method (default: ABMIL)')
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--embedding-level', action='store_true', default=True)
    parser.add_argument('--supervised', action='store_true', default=False,
                        help='Use dataset with instance labels (for supervised BernoulliVAP)')
    args = parser.parse_args()

    print(f"Model:      {args.model_path}")
    print(f"Method:     {args.method}")
    print(f"Seed:       {args.seed}")
    print(f"Supervised: {args.supervised}")
    print()

    result = evaluate_model(args.model_path, args.method, args.seed, args.embedding_level, args.supervised)

    print(f"Attention sum on positive: {result['attn_sum']:.4f}")
    print(f"AUROC:                     {result['auroc']:.4f}")
    print(f"AUPRC:                     {result['auprc']:.4f}")
    print(f"Max attention correct:     {result['max_correct']:.4f}")
    print(f"Max attention:             {result['max_attn']:.4f}")


if __name__ == '__main__':
    main()
