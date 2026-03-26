#!/usr/bin/env python3
"""
Evaluate instance-level attention metrics for THUMOS14.

Usage:
    python scripts/evaluate_thumos14.py model.pt \
        --dataset_dir /path/to/THUMOS14_encoded/binary/seed=1001 \
        --method ABMIL
"""

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import models


def load_model(model_path, in_features, pooling, embedding_level=True):
    if embedding_level:
        model = models.PoolClf(in_features, 1, pooling)
    else:
        model = models.ClfPool(in_features, 1, pooling)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def get_attention(model, X, lengths):
    with torch.no_grad():
        _, attn = model(X, lengths)
        return attn.squeeze().numpy()


def evaluate(model_path, dataset_dir, method, embedding_level):
    test_data = torch.load(f'{dataset_dir}/test.pth', map_location='cpu', weights_only=False)
    X, lengths, y = test_data['X'], test_data['lengths'], test_data['y']

    assert 'lengths_y' in test_data, \
        "test.pth missing 'lengths_y' — run convert_thumos14.py with instance labels first"
    instance_labels = test_data['lengths_y']

    model = load_model(model_path, X.shape[1], method, embedding_level)
    attention = get_attention(model, X, lengths)

    attn_sum, aurocs, auprcs, max_correct = [], [], [], []
    start = 0
    for i, length in enumerate(lengths):
        attn = attention[start:start + length]
        labels = np.array(instance_labels[i][:length])
        start += length

        if labels.sum() == 0:
            continue

        attn_sum.append(attn[labels == 1].sum())
        if len(np.unique(labels)) > 1:
            aurocs.append(roc_auc_score(labels, attn))
            auprcs.append(average_precision_score(labels, attn))
        max_correct.append(labels[np.argmax(attn)] == 1)

    return {
        'attn_sum': np.mean(attn_sum),
        'auroc': np.mean(aurocs),
        'auprc': np.mean(auprcs),
        'max_correct': np.mean(max_correct),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate THUMOS14 instance-level attention')
    parser.add_argument('model_path', type=str)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--method', type=str, default='ABMIL',
                        choices=['ABMIL', 'TransMIL', 'SmAP', 'Mean', 'Max'])
    parser.add_argument('--embedding-level', action='store_true', default=True)
    args = parser.parse_args()

    print(f"Model:       {args.model_path}")
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Method:      {args.method}")
    print()

    result = evaluate(args.model_path, args.dataset_dir, args.method, args.embedding_level)

    print(f"Attention sum on positive: {result['attn_sum']:.4f}")
    print(f"Instance AUROC:            {result['auroc']:.4f}")
    print(f"Instance AUPRC:            {result['auprc']:.4f}")
    print(f"Max attention correct:     {result['max_correct']:.4f}")


if __name__ == '__main__':
    main()
