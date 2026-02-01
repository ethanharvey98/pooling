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
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.append('src')
import models


# Configuration
EXPERIMENTS_DIR = '/cluster/tufts/hugheslab/dloevl01/pooling/experiments/RSNA/embedding_level=True'
DATASET_DIR = '/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16'
EMBEDDING_LEVEL = True
SEEDS = [1001, 2001, 3001]


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
        except Exception as e:
            continue

    # Sort by val_auroc descending and take top K
    model_results.sort(key=lambda x: x['val_auroc'], reverse=True)
    return model_results[:top_k]


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
    """Get predictions from a single model."""
    with torch.no_grad():
        logits, _ = model(X, lengths)
        probs = torch.sigmoid(logits).squeeze().numpy()
    return probs


def ensemble_predict(models, X, lengths):
    """Average predictions from multiple models."""
    all_preds = []
    for model in models:
        preds = get_predictions(model, X, lengths)
        all_preds.append(preds)
    return np.mean(all_preds, axis=0)


def evaluate_ensemble(experiments_dir, dataset_dir, pooling_methods, seed, top_k=1, name="Ensemble"):
    """Evaluate an ensemble of models."""
    # Load test data
    test_data = torch.load(f'{dataset_dir}/seed={seed}/test.pth', map_location='cpu', weights_only=False)
    X, lengths, y = test_data['X'], test_data['lengths'], test_data['y']
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
    ensemble_preds = ensemble_predict(all_models, X, lengths)
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
        print(f"\n{'='*80}")
        print(f"Ensemble: {ensemble_name}")
        print('='*80)

        results = []
        for seed in SEEDS:
            print(f"\nSeed {seed}:")
            test_auroc = evaluate_ensemble(
                EXPERIMENTS_DIR, DATASET_DIR,
                config['pooling_methods'], seed,
                top_k=config['top_k'],
                name=ensemble_name
            )

            if test_auroc is not None:
                results.append(test_auroc)
                print(f"  Test AUROC: {test_auroc:.4f}")
            else:
                print(f"  No models found")

        if results:
            print(f"\n{ensemble_name} Summary ({len(results)} seeds):")
            print(f"  Mean: {np.mean(results):.4f} ± {np.std(results):.4f}")

    print("\n" + "="*80)
    print("Evaluation complete")
    print("="*80)


if __name__ == '__main__':
    main()
