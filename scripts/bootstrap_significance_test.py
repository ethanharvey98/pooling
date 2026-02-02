#!/usr/bin/env python3
"""
Bootstrap significance testing for comparing pooling methods on RSNA dataset.

Usage:
    python scripts/bootstrap_significance_test.py --method1 TransMIL --method2 Mean --seed 1001 --n_bootstrap 1000
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.append('src')
import models


def find_best_model(experiments_dir, pooling, seed):
    """Find best model by validation AUROC."""
    pattern = os.path.join(experiments_dir, f"*pooling={pooling}*seed={seed}*.csv")
    best_val_auroc, best_file = -1, None

    for csv_file in glob.glob(pattern):
        import pandas as pd
        df = pd.read_csv(csv_file)
        valid_df = df[df['val_auroc'] <= df['train_auroc']]
        if valid_df.empty:
            continue
        idx = valid_df['val_auroc'].idxmax()
        if df.loc[idx, 'val_auroc'] > best_val_auroc:
            best_val_auroc = df.loc[idx, 'val_auroc']
            best_file = csv_file.replace('.csv', '.pt')

    return best_file, best_val_auroc


def load_model(model_path, in_features, pooling, embedding_level):
    """Load model from checkpoint."""
    if embedding_level:
        model = models.PoolClf(in_features, 1, pooling)
    else:
        model = models.ClfPool(in_features, 1, pooling)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def get_predictions(model, X, lengths):
    """Get scan-level predictions."""
    with torch.no_grad():
        logits, _ = model(X, lengths)
        probs = torch.sigmoid(logits).squeeze().numpy()
    return probs


def get_mean_predictions(X, lengths):
    """Get mean pooling predictions (uniform average)."""
    # For mean pooling, we need a simple classifier on top of mean-pooled features
    # Since we don't have a trained mean pooling model, we'll compute mean pooling
    # and use it as a baseline. But actually, we should load the Mean model too.
    raise NotImplementedError("Mean pooling predictions need a trained model")


def bootstrap_auroc_difference(y_true, preds1, preds2, n_bootstrap=1000, random_state=42):
    """
    Bootstrap test for difference in AUROC between two methods.

    Returns:
        auroc1_samples: Bootstrap AUROC samples for method 1
        auroc2_samples: Bootstrap AUROC samples for method 2
        diff_samples: Bootstrap AUROC difference samples (method1 - method2)
        ci_lower: 2.5th percentile of difference
        ci_upper: 97.5th percentile of difference
    """
    np.random.seed(random_state)
    n_samples = len(y_true)

    auroc1_samples = []
    auroc2_samples = []
    diff_samples = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        y_boot = y_true[indices]
        preds1_boot = preds1[indices]
        preds2_boot = preds2[indices]

        # Compute AUROC for each method
        auroc1 = roc_auc_score(y_boot, preds1_boot)
        auroc2 = roc_auc_score(y_boot, preds2_boot)

        auroc1_samples.append(auroc1)
        auroc2_samples.append(auroc2)
        diff_samples.append(auroc1 - auroc2)

    auroc1_samples = np.array(auroc1_samples)
    auroc2_samples = np.array(auroc2_samples)
    diff_samples = np.array(diff_samples)

    # 95% confidence interval
    ci_lower = np.percentile(diff_samples, 2.5)
    ci_upper = np.percentile(diff_samples, 97.5)

    return auroc1_samples, auroc2_samples, diff_samples, ci_lower, ci_upper


def plot_bootstrap_histogram(diff_samples, ci_lower, ci_upper, epsilon, prob_greater_epsilon, method1, method2, output_path):
    """Plot histogram of bootstrap AUROC differences with confidence intervals and epsilon threshold."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(diff_samples, bins=50, color='#4A90E2', alpha=0.7, edgecolor='black')

    # Confidence interval lines
    ax.axvline(ci_lower, color='red', linestyle='--', linewidth=2, label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    ax.axvline(ci_upper, color='red', linestyle='--', linewidth=2)

    # Zero line (no difference)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='No difference')

    # Mean difference
    mean_diff = diff_samples.mean()
    ax.axvline(mean_diff, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.4f}')

    # Epsilon threshold line
    ax.axvline(epsilon, color='purple', linestyle=':', linewidth=2.5,
               label=f'ε = {epsilon:.3f}\nPr(Δ > ε) = {prob_greater_epsilon:.1%}')

    ax.set_xlabel(f'AUROC Difference ({method1} - {method2})', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Bootstrap Distribution of AUROC Difference\n{method1} vs {method2}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Bootstrap significance testing for pooling methods')
    parser.add_argument('--method1', default='TransMIL', help='First pooling method (default: TransMIL)', type=str)
    parser.add_argument('--method2', default='Mean', help='Second pooling method (default: Mean)', type=str)
    parser.add_argument('--seed', default=1001, help='Random seed for train/test split (default: 1001)', type=int)
    parser.add_argument('--n_bootstrap', default=1000, help='Number of bootstrap samples (default: 1000)', type=int)
    parser.add_argument('--epsilon', default=0.001, help='Minimum meaningful difference threshold (default: 0.001)', type=float)
    parser.add_argument('--experiments_dir', default='/cluster/tufts/hugheslab/dloevl01/pooling/experiments/RSNA/embedding_level=True',
                        help='Directory containing experiments', type=str)
    parser.add_argument('--dataset_dir', default='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16',
                        help='Directory containing datasets', type=str)
    parser.add_argument('--embedding_level', action='store_true', default=True,
                        help='Use embedding-level approach (default: True)')
    parser.add_argument('--output_dir', default='figures', help='Output directory for plots (default: figures)', type=str)
    args = parser.parse_args()

    print("=" * 80)
    print(f"Bootstrap Significance Test: {args.method1} vs {args.method2}")
    print("=" * 80)
    print(f"Seed: {args.seed}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Epsilon (minimum meaningful difference): {args.epsilon}")
    print()

    # Load test data
    test_data = torch.load(f'{args.dataset_dir}/seed={args.seed}/test.pth',
                          map_location='cpu', weights_only=False)
    X, lengths, y = test_data['X'], test_data['lengths'], test_data['y']
    y = y.numpy().flatten()

    print(f"Test set: {len(y)} scans")
    print()

    # Load method 1 model
    print(f"Loading {args.method1} model...")
    model1_path, val_auroc1 = find_best_model(args.experiments_dir, args.method1, args.seed)
    if not model1_path:
        print(f"Error: {args.method1} model not found")
        return

    model1 = load_model(model1_path, X.shape[1], args.method1, args.embedding_level)
    preds1 = get_predictions(model1, X, lengths)
    auroc1 = roc_auc_score(y, preds1)
    print(f"  Val AUROC: {val_auroc1:.4f}")
    print(f"  Test AUROC: {auroc1:.4f}")
    print()

    # Load method 2 model
    print(f"Loading {args.method2} model...")
    model2_path, val_auroc2 = find_best_model(args.experiments_dir, args.method2, args.seed)
    if not model2_path:
        print(f"Error: {args.method2} model not found")
        return

    model2 = load_model(model2_path, X.shape[1], args.method2, args.embedding_level)
    preds2 = get_predictions(model2, X, lengths)
    auroc2 = roc_auc_score(y, preds2)
    print(f"  Val AUROC: {val_auroc2:.4f}")
    print(f"  Test AUROC: {auroc2:.4f}")
    print()

    # Bootstrap test
    print("Running bootstrap test...")
    auroc1_samples, auroc2_samples, diff_samples, ci_lower, ci_upper = bootstrap_auroc_difference(
        y, preds1, preds2, n_bootstrap=args.n_bootstrap, random_state=args.seed
    )

    # Compute Pr(Delta > epsilon)
    prob_greater_epsilon = np.mean(diff_samples > args.epsilon)

    # Find epsilon where 95% of samples are greater (5th percentile)
    eps_95 = np.percentile(diff_samples, 5)

    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print(f"{args.method1} AUROC: {auroc1:.4f} (bootstrap mean: {auroc1_samples.mean():.4f} ± {auroc1_samples.std():.4f})")
    print(f"{args.method2} AUROC: {auroc2:.4f} (bootstrap mean: {auroc2_samples.mean():.4f} ± {auroc2_samples.std():.4f})")
    print(f"Difference: {auroc1 - auroc2:.4f}")
    print(f"Bootstrap mean difference: {diff_samples.mean():.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    print()
    print(f"Pr(Δ > ε={args.epsilon}): {prob_greater_epsilon:.1%}")
    print(f"We are {prob_greater_epsilon:.1%} confident that {args.method1} improves AUROC by at least {args.epsilon}")
    print()
    print(f"ε at 95% confidence: {eps_95:.4f}")
    print(f"We are 95% confident that {args.method1} improves AUROC by at least {eps_95:.4f}")

    if ci_lower > 0:
        print(f"\n{args.method1} is significantly better than {args.method2} (p < 0.05)")
    elif ci_upper < 0:
        print(f"\n{args.method2} is significantly better than {args.method1} (p < 0.05)")
    else:
        print(f"\nNo significant difference between {args.method1} and {args.method2} (p >= 0.05)")

    print()

    # Plot histogram
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = f"{args.output_dir}/bootstrap_{args.method1}_vs_{args.method2}_seed{args.seed}.png"
    plot_bootstrap_histogram(diff_samples, ci_lower, ci_upper, args.epsilon, prob_greater_epsilon,
                            args.method1, args.method2, output_path)

    print()
    print("=" * 80)
    print("Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
