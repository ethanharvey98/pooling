#!/usr/bin/env python3
"""
Shared utilities for RSNA evaluation scripts.

This module provides common functions used across multiple RSNA analysis scripts:
- find_best_model: Find best model by validation AUROC
- load_model: Load model from checkpoint
- get_attention: Get attention weights from model
- get_test_slice_labels: Get test set slice labels
- get_predictions: Get scan-level predictions
"""

import ast
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.append('src')
import models


# Default configuration constants
EXPERIMENTS_DIR = '/cluster/tufts/hugheslab/dloevl01/pooling/experiments/RSNA/embedding_level=True'
DATASET_DIR = '/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16'
LABELS_CSV = '/cluster/tufts/hugheslab/datasets/RSNA/labels.csv'
NUMPY_DIR = '/cluster/tufts/hugheslab/datasets/RSNA_numpy'
EMBEDDING_LEVEL = True
SEEDS = [1001, 2001, 3001]


def find_best_model(experiments_dir, pooling, seed):
    """
    Find best model by validation AUROC.

    Args:
        experiments_dir: Directory containing experiment CSV files
        pooling: Pooling method name (e.g., 'ABMIL', 'TransMIL', 'Mean')
        seed: Random seed used for the experiment

    Returns:
        Tuple of (model_path, val_auroc) or (None, -1) if not found
    """
    pattern = os.path.join(experiments_dir, f"*pooling={pooling}*seed={seed}*.csv")
    best_val_auroc, best_file = -1, None

    for csv_file in glob.glob(pattern):
        try:
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            print(f"  Warning: Empty CSV file skipped: {csv_file}")
            continue
        valid_df = df[df['val_auroc'] <= df['train_auroc']]
        if valid_df.empty:
            continue
        idx = valid_df['val_auroc'].idxmax()
        if df.loc[idx, 'val_auroc'] > best_val_auroc:
            best_val_auroc = df.loc[idx, 'val_auroc']
            best_file = csv_file.replace('.csv', '.pt')

    return best_file, best_val_auroc


def load_model(model_path, in_features, pooling, embedding_level=True):
    """
    Load model from checkpoint.

    Args:
        model_path: Path to the .pt model file
        in_features: Number of input features
        pooling: Pooling method name
        embedding_level: Whether to use embedding-level (PoolClf) or bag-level (ClfPool)

    Returns:
        Loaded model in eval mode
    """
    if pooling == 'InstanceClassifier':
        model = models.InstanceClassifier(in_features, 1)
    elif embedding_level:
        model = models.PoolClf(in_features, 1, pooling)
    else:
        model = models.ClfPool(in_features, 1, pooling)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def get_attention(model, X, lengths, embedding_level=True):
    """
    Get attention weights from model.

    Args:
        model: Loaded model
        X: Input embeddings tensor
        lengths: Tuple of sequence lengths
        embedding_level: Whether using embedding-level approach

    Returns:
        Numpy array of attention weights
    """
    with torch.no_grad():
        _, attn = model(X, lengths)
        if isinstance(model, models.InstanceClassifier):
            return attn.squeeze().numpy()
        if embedding_level:
            return attn.squeeze().numpy()
        return torch.sigmoid(model.clf(X)).squeeze().numpy()


def get_predictions(model, X, lengths):
    """
    Get scan-level predictions from model.

    Args:
        model: Loaded model
        X: Input embeddings tensor
        lengths: Tuple of sequence lengths

    Returns:
        Numpy array of prediction probabilities
    """
    with torch.no_grad():
        logits, _ = model(X, lengths)
        probs = torch.sigmoid(logits).squeeze().numpy()
    return probs


def get_test_slice_labels(labels_csv, numpy_dir, seed):
    """
    Get test set slice labels.

    This function reconstructs the exact same test split used during encoding
    by using the same CSV, filtering, and train_test_split parameters.

    Args:
        labels_csv: Path to labels.csv
        numpy_dir: Directory containing numpy files
        seed: Random seed for reproducible split

    Returns:
        Tuple of (scan_ids, slice_labels) where slice_labels is array of arrays
    """
    df = pd.read_csv(labels_csv)
    df['scan_label'] = df['Any'].apply(lambda x: 1 if any(ast.literal_eval(x)) else 0)
    df['path'] = df['Study ID'].apply(lambda x: f'{numpy_dir}/{x}.npz')
    df = df[df['path'].apply(os.path.exists)]

    _, test_ids, _, _ = train_test_split(
        df['Study ID'], df['scan_label'],
        test_size=1/6, random_state=seed, stratify=df['scan_label']
    )
    test_df = df[df['Study ID'].isin(test_ids)]
    return test_df['Study ID'].values, test_df['Any'].apply(lambda x: np.array(ast.literal_eval(x))).values


def load_test_data(dataset_dir, seed):
    """
    Load test data from encoded .pth file.

    Args:
        dataset_dir: Base directory for encoded data
        seed: Random seed

    Returns:
        Tuple of (X, lengths, y) tensors
    """
    test_data = torch.load(f'{dataset_dir}/seed={seed}/test.pth', map_location='cpu', weights_only=False)
    return test_data['X'], test_data['lengths'], test_data['y']
