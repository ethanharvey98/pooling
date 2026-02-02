#!/usr/bin/env python3
"""
Evaluate ensemble methods on RSNA test set.

Ensembles:
1. Best ABMIL + Best Mean
2. Best Mean + Best Max + Best ABMIL
3. Top 5 Mean models
4. Top 5 ABMIL models

Usage:
    python scripts/evaluate_ensembles.py
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rsna_utils import (
    EXPERIMENTS_DIR, DATASET_DIR, EMBEDDING_LEVEL, SEEDS,
    load_model, get_predictions, load_test_data
)


def find_best_models(experiments_dir, pooling, seed, top_k=1):
    """Find top K best models by validation AUROC."""
    pattern = os.path.join(experiments_dir, f"*pooling={pooling}*seed={seed}*.csv")

    model_results = []
    for csv_file in glob.glob(pattern):
        try:
            df = pd.read_csv(csv_file)
            valid_df = df[df['val_auroc'] <= df['train_auroc']]
            if valid_df.empty:
                continue

            idx = valid_df['val_auroc'].idxmax()
            val_auroc = df.loc[idx, 'val_auroc']
            model_path = csv_file.replace('.csv', '.pt')

            if os.path.exists(model_path):
                model_results.append({
                    'path': model_path,
                    'val_auroc': val_auroc,
                    'csv': csv_file
                })
        except Exception:
            continue

    # Sort by val_auroc descending and take top K
    model_results.sort(key=lambda x: x['val_auroc'], reverse=True)
    return model_results[:top_k]


def ensemble_predict(models, X, lengths, method='soft'):
    """
    Ensemble predictions from multiple models.

    Args:
        models: List of models
        X: Input features
        lengths: Sequence lengths
        method: 'soft' (average probabilities) or 'hard' (majority vote)
    """
    all_preds = []
    for model in models:
        preds = get_predictions(model, X, lengths)
        all_preds.append(preds)

    all_preds = np.array(all_preds)

    if method == 'soft':
        # Average probabilities
        return np.mean(all_preds, axis=0)
    elif method == 'hard':
        # Majority vote: threshold at 0.5, then average votes
        hard_preds = (all_preds > 0.5).astype(float)
        return np.mean(hard_preds, axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def evaluate_ensemble(experiments_dir, dataset_dir, pooling_methods, seed, top_k=1, method='soft', name="Ensemble"):
    """Evaluate an ensemble of models."""
    # Load test data
    X, lengths, y = load_test_data(dataset_dir, seed)
    y = y.numpy().flatten()

    # Load models for each pooling method
    all_models = []
    for pooling in pooling_methods:
        best_models = find_best_models(experiments_dir, pooling, seed, top_k)

        for model_info in best_models:
            model = load_model(model_info['path'], X.shape[1], pooling, EMBEDDING_LEVEL)
            all_models.append(model)
            print(f"  Loaded {pooling}: {os.path.basename(model_info['csv'])} (val AUROC: {model_info['val_auroc']:.4f})")

    if not all_models:
        return None

    # Get ensemble predictions
    ensemble_preds = ensemble_predict(all_models, X, lengths, method=method)
    test_auroc = roc_auc_score(y, ensemble_preds)

    return test_auroc


def main():
    print("=" * 80)
    print("Evaluating Ensemble Methods on RSNA Test Set")
    print("=" * 80)
    print()

    ensembles = {
        'ABMIL + Mean': {
            'pooling_methods': ['ABMIL', 'Mean'],
            'top_k': 1
        },
        'Mean + Max + ABMIL': {
            'pooling_methods': ['Mean', 'Max', 'ABMIL'],
            'top_k': 1
        },
        'Top 5 Mean': {
            'pooling_methods': ['Mean'],
            'top_k': 5
        },
        'Top 5 ABMIL': {
            'pooling_methods': ['ABMIL'],
            'top_k': 5
        }
    }

    for ensemble_name, config in ensembles.items():
        for method in ['soft', 'hard']:
            print(f"\n{'='*80}")
            print(f"Ensemble: {ensemble_name} ({method} voting)")
            print('='*80)

            results = []
            for seed in SEEDS:
                print(f"\nSeed {seed}:")
                test_auroc = evaluate_ensemble(
                    EXPERIMENTS_DIR, DATASET_DIR,
                    config['pooling_methods'], seed,
                    top_k=config['top_k'],
                    method=method,
                    name=ensemble_name
                )

                if test_auroc is not None:
                    results.append(test_auroc)
                    print(f"  Test AUROC: {test_auroc:.4f}")
                else:
                    print(f"  No models found")

            if results:
                print(f"\n{ensemble_name} ({method}) Summary ({len(results)} seeds):")
                print(f"  Mean: {np.mean(results):.4f} ± {np.std(results):.4f}")

    print("\n" + "="*80)
    print("Evaluation complete")
    print("="*80)


if __name__ == '__main__':
    main()
