"""
Find best results across THUMOS14 per-class experiments.

Usage:
    python scripts/find_best_results_thumos14.py /path/to/experiments/THUMOS14
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

base_dir = sys.argv[1] if len(sys.argv) > 1 else '/cluster/tufts/hugheslab/dloevl01/action_recognition_mil/pooling/experiments/THUMOS14'

# Find all class directories
class_dirs = sorted(glob.glob(f'{base_dir}/class=*'))
if not class_dirs:
    print(f"No class directories found in {base_dir}")
    sys.exit(1)

methods = {}  # method_name -> {class_name -> [test_auroc per seed]}

for class_dir in class_dirs:
    class_name = os.path.basename(class_dir)

    for csv_file in sorted(glob.glob(f'{class_dir}/*.csv')):
        model_name = os.path.basename(csv_file).replace('.csv', '')

        try:
            df = pd.read_csv(csv_file, index_col=0)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue

        if df.empty:
            continue

        # Extract method info (everything except seed)
        parts = model_name.rsplit('_seed=', 1)
        if len(parts) != 2:
            continue
        method_key = parts[0]

        # Best val_auroc where val < train
        valid = df[df['train_auroc'] > df['val_auroc']]
        if valid.empty:
            continue
        best_epoch = valid['val_auroc'].idxmax()
        test_auroc = df.loc[best_epoch, 'test_auroc']

        if method_key not in methods:
            methods[method_key] = {}
        if class_name not in methods[method_key]:
            methods[method_key][class_name] = []
        methods[method_key][class_name].append(test_auroc)

# Find best method per class (by mean test_auroc across seeds)
print(f"{'Class':<30s} {'Best Method':<60s} {'Seeds':>5s} {'Mean AUROC':>10s} {'Std':>8s}")
print('-' * 120)

all_class_results = {}
for class_dir in class_dirs:
    class_name = os.path.basename(class_dir)
    best_mean, best_method, best_scores = -1, None, None

    for method_key, class_dict in methods.items():
        if class_name not in class_dict:
            continue
        scores = class_dict[class_name]
        if len(scores) >= 3:  # require all 3 seeds
            mean = np.mean(scores)
            if mean > best_mean:
                best_mean = mean
                best_method = method_key
                best_scores = scores

    if best_method is not None:
        std = np.std(best_scores)
        print(f"{class_name:<30s} {best_method:<60s} {len(best_scores):>5d} {best_mean:>10.4f} {std:>8.4f}")
        all_class_results[class_name] = best_mean

if all_class_results:
    print('-' * 120)
    overall_mean = np.mean(list(all_class_results.values()))
    print(f"{'OVERALL MEAN':<30s} {'':60s} {'':>5s} {overall_mean:>10.4f}")
