#!/usr/bin/env python3
"""
Analyze timing results from RSNA timing experiments.

Usage:
    python scripts/analyze_timing_results.py
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('src')
import models
import torch


# Configuration
EXPERIMENTS_DIR = '/cluster/tufts/hugheslab/dloevl01/pooling/experiments/RSNA/timing'
METHODS = ['Max', 'Mean', 'ABMIL', 'TransMIL', 'SmAP']
OUTPUT_DIR = 'figures'


def count_parameters(pooling, in_features=768):
    """Count parameters for a given pooling method."""
    if pooling == 'Max' or pooling == 'Mean':
        # Just a linear classifier: in_features x 1 + bias
        return in_features + 1
    elif pooling == 'ABMIL':
        # ABMIL: attention network + classifier
        model = models.PoolClf(in_features, 1, pooling)
    elif pooling == 'TransMIL':
        # TransMIL: transformer + classifier
        model = models.PoolClf(in_features, 1, pooling)
    elif pooling == 'SmAP':
        # SmAP: spatial attention + classifier
        model = models.PoolClf(in_features, 1, pooling)
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_timing_results(experiments_dir, method):
    """Load all timing results for a given method."""
    pattern = os.path.join(experiments_dir, f"*pooling={method}*timing.csv")
    results = []

    for csv_file in glob.glob(pattern):
        try:
            df = pd.read_csv(csv_file)
            if 'train_sec_per_epoch' in df.columns:
                # Get total time for one complete hyperparameter search
                total_time = df['train_sec_per_epoch'].sum()
                results.append({
                    'method': method,
                    'file': os.path.basename(csv_file),
                    'total_time_hours': total_time / 3600,
                    'total_epochs': len(df)
                })
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")

    return results


def main():
    print("=" * 80)
    print("Analyzing Timing Results")
    print("=" * 80)
    print()

    all_results = []
    method_stats = {}

    for method in METHODS:
        results = load_timing_results(EXPERIMENTS_DIR, method)
        all_results.extend(results)

        if results:
            times = [r['total_time_hours'] for r in results]
            method_stats[method] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times),
                'params': count_parameters(method)
            }

            print(f"{method}:")
            print(f"  Parameters: {method_stats[method]['params']:,}")
            print(f"  Experiments: {len(results)}")
            print(f"  Mean time per hyperparam search: {method_stats[method]['mean']:.2f} ± {method_stats[method]['std']:.2f} hours")
            print(f"  Min: {method_stats[method]['min']:.2f} hours")
            print(f"  Max: {method_stats[method]['max']:.2f} hours")
            print()
        else:
            print(f"{method}: No results found")
            print()

    # Create plot (exclude Max)
    if method_stats:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter out Max
        methods = [m for m in method_stats.keys() if m != 'Max']
        params = [method_stats[m]['params'] for m in methods]
        times = [method_stats[m]['mean'] for m in methods]
        stds = [method_stats[m]['std'] for m in methods]

        # Sort by number of parameters
        sorted_indices = np.argsort(params)
        methods = [methods[i] for i in sorted_indices]
        params = [params[i] for i in sorted_indices]
        times = [times[i] for i in sorted_indices]
        stds = [stds[i] for i in sorted_indices]

        # Define colors: Mean green, ABMIL red, TransMIL purple, SmAP brown
        colors = {'Mean': '#2ECC71', 'ABMIL': '#D62728',
                  'TransMIL': '#9467BD', 'SmAP': '#8C564B'}

        # Plot each method with different color
        for i, method in enumerate(methods):
            color = colors.get(method, '#95A5A6')
            ax.errorbar(params[i], times[i], yerr=stds[i],
                       fmt='o', capsize=5, capthick=2, markersize=8,
                       color=color, ecolor=color)

        # Add method labels next to dots
        for i, method in enumerate(methods):
            ax.annotate(method, (params[i], times[i]),
                       textcoords="offset points", xytext=(0,10), ha='center')

        ax.set_xlabel('Number of Parameters', fontsize=12)
        ax.set_ylabel('Training Time per Hyperparameter Search (hours)', fontsize=12)
        ax.set_title('Method Complexity vs Training Time', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Log scale if parameter counts vary widely
        if max(params) / min(params) > 10:
            ax.set_xscale('log')

        plt.tight_layout()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = f'{OUTPUT_DIR}/timing_vs_params.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    print()
    print("=" * 80)
    print("Analysis complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
