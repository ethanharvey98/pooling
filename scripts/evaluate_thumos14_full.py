"""
Full evaluation for THUMOS14: find best model by val AUROC, report bag-level
test AUROC, then evaluate instance-level attention metrics.

Usage:
    python scripts/evaluate_thumos14_full.py \
        --experiments_dir /path/to/experiments/THUMOS14/class=7_Diving \
        --dataset_dir /cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14_encoded/class=7_Diving

    # Specific method:
    python scripts/evaluate_thumos14_full.py \
        --experiments_dir /path/to/experiments/THUMOS14/class=7_Diving \
        --dataset_dir /cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14_encoded/class=7_Diving \
        --method ABMIL
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import models


def find_best_model(experiments_dir, seed, method=None):
    """Find best model by val_auroc (where val < train). Optionally filter by method name."""
    if method:
        pattern = os.path.join(experiments_dir, f"*pooling={method}*seed={seed}*.csv")
    else:
        pattern = os.path.join(experiments_dir, f"*seed={seed}*.csv")

    best_val, best_test, best_file, best_epoch = -1, None, None, None

    for csv_file in sorted(glob.glob(pattern)):
        try:
            df = pd.read_csv(csv_file, index_col=0)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if df.empty:
            continue
        valid = df[df['train_auroc'] > df['val_auroc']]
        if valid.empty:
            continue
        idx = valid['val_auroc'].idxmax()
        if df.loc[idx, 'val_auroc'] > best_val:
            best_val = df.loc[idx, 'val_auroc']
            best_test = df.loc[idx, 'test_auroc']
            best_file = csv_file.replace('.csv', '.pt')
            best_epoch = idx

    return best_file, best_val, best_test, best_epoch


def evaluate_instance(model_path, dataset_dir, seed, pooling='ABMIL'):
    """Evaluate instance-level attention metrics."""
    test_data = torch.load(f'{dataset_dir}/seed={seed}/test.pth', map_location='cpu', weights_only=False)
    X, lengths, y = test_data['X'], test_data['lengths'], test_data['y']

    if 'lengths_y' not in test_data:
        print("  No instance labels (lengths_y) in test.pth — skipping instance eval")
        return None

    instance_labels = test_data['lengths_y']

    model = models.PoolClf(X.shape[1], 1, pooling)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    with torch.no_grad():
        _, attn = model(X, lengths)
    attention = attn.squeeze().numpy()

    attn_sum, aurocs, auprcs, max_correct = [], [], [], []
    start = 0
    for i, length in enumerate(lengths):
        a = attention[start:start + length]
        labels = np.array(instance_labels[i][:length])
        start += length

        if labels.sum() == 0:
            continue

        attn_sum.append(a[labels == 1].sum())
        if len(np.unique(labels)) > 1:
            aurocs.append(roc_auc_score(labels, a))
            auprcs.append(average_precision_score(labels, a))
        max_correct.append(labels[np.argmax(a)] == 1)

    return {
        'attn_sum': np.mean(attn_sum),
        'auroc': np.mean(aurocs) if aurocs else float('nan'),
        'auprc': np.mean(auprcs) if auprcs else float('nan'),
        'max_correct': np.mean(max_correct),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--method', type=str, default=None, help='Filter by method name (e.g. ABMIL)')
    args = parser.parse_args()

    seeds = [1001, 2001, 3001]

    # Detect methods present
    csv_files = glob.glob(f'{args.experiments_dir}/*.csv')
    methods_found = set()
    for f in csv_files:
        name = os.path.basename(f)
        if 'criterion=GuidedL1' in name:
            methods_found.add('GuidedABMIL')
        elif 'pooling=ABMIL' in name:
            methods_found.add('ABMIL')
        elif 'pooling=Mean' in name:
            methods_found.add('Mean')

    if args.method:
        methods_found = {args.method}

    for method_name in sorted(methods_found):
        print(f"\n{'='*70}")
        print(f"Method: {method_name}")
        print(f"{'='*70}")

        # Map method_name to filter pattern
        if method_name == 'GuidedABMIL':
            filter_pattern = 'GuidedL1'
        else:
            filter_pattern = method_name

        bag_aurocs = []
        inst_results = []

        for seed in seeds:
            # Find best model for this seed
            if method_name == 'GuidedABMIL':
                pattern = os.path.join(args.experiments_dir, f"*criterion=GuidedL1*seed={seed}*.csv")
            else:
                pattern = os.path.join(args.experiments_dir, f"*pooling={method_name}*seed={seed}*.csv")
                # Exclude GuidedL1 from ABMIL results
                if method_name == 'ABMIL':
                    all_files = glob.glob(pattern)
                    pattern_files = [f for f in all_files if 'GuidedL1' not in f]
                else:
                    pattern_files = glob.glob(pattern)

            best_val, best_test, best_file, best_epoch = -1, None, None, None

            search_files = glob.glob(pattern) if method_name == 'GuidedABMIL' else pattern_files
            for csv_file in search_files:
                try:
                    df = pd.read_csv(csv_file, index_col=0)
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                    continue
                if df.empty:
                    continue
                valid = df[df['train_auroc'] > df['val_auroc']]
                if valid.empty:
                    continue
                idx = valid['val_auroc'].idxmax()
                if df.loc[idx, 'val_auroc'] > best_val:
                    best_val = df.loc[idx, 'val_auroc']
                    best_test = df.loc[idx, 'test_auroc']
                    best_file = csv_file.replace('.csv', '.pt')
                    best_epoch = idx

            if best_file is None:
                print(f"\n  Seed {seed}: No valid results found")
                continue

            bag_aurocs.append(best_test)
            print(f"\n  Seed {seed}:")
            print(f"    Best file:  {os.path.basename(best_file)}")
            print(f"    Epoch:      {best_epoch}")
            print(f"    Val AUROC:  {best_val:.4f}")
            print(f"    Test AUROC: {best_test:.4f}")

            # Instance-level evaluation
            if os.path.exists(best_file):
                inst = evaluate_instance(best_file, args.dataset_dir, seed)
                if inst:
                    inst_results.append(inst)
                    print(f"    Instance AUROC:    {inst['auroc']:.4f}")
                    print(f"    Instance AUPRC:    {inst['auprc']:.4f}")
                    print(f"    Max correct:       {inst['max_correct']:.4f}")
                    print(f"    Attn sum positive: {inst['attn_sum']:.4f}")
            else:
                print(f"    (model .pt not found — skipping instance eval)")

        if bag_aurocs:
            print(f"\n  SUMMARY ({len(bag_aurocs)} seeds):")
            print(f"    Bag Test AUROC:      {np.mean(bag_aurocs):.4f} +/- {np.std(bag_aurocs):.4f}")
        if inst_results:
            print(f"    Instance AUROC:      {np.mean([r['auroc'] for r in inst_results]):.4f} +/- {np.std([r['auroc'] for r in inst_results]):.4f}")
            print(f"    Instance AUPRC:      {np.mean([r['auprc'] for r in inst_results]):.4f} +/- {np.std([r['auprc'] for r in inst_results]):.4f}")
            print(f"    Max correct:         {np.mean([r['max_correct'] for r in inst_results]):.4f} +/- {np.std([r['max_correct'] for r in inst_results]):.4f}")


if __name__ == '__main__':
    main()
