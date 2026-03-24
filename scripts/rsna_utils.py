#!/usr/bin/env python3
"""
Shared utilities for RSNA evaluation scripts.
"""

import ast
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import models


EXPERIMENTS_DIR = '/cluster/tufts/hugheslab/dloevl01/pooling/experiments/RSNA/embedding_level=True'
DATASET_DIR = '/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16'
DATASET_DIR_WITH_LABELS = '/cluster/tufts/hugheslab/datasets/encoded_RSNA_full_dataset_with_lengths_labels/ViT_B_16'
LABELS_CSV = '/cluster/tufts/hugheslab/datasets/RSNA/labels.csv'
NUMPY_DIR = '/cluster/tufts/hugheslab/datasets/RSNA_numpy'
SEEDS = [1001, 2001, 3001]


def find_best_model(experiments_dir, pooling, seed):
    pattern = os.path.join(experiments_dir, f"*pooling={pooling}*seed={seed}*.csv")
    best_val_auroc, best_file = -1, None

    for csv_file in glob.glob(pattern):
        try:
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            continue
        valid_df = df[df['val_auroc'] <= df['train_auroc']]
        if valid_df.empty:
            continue
        idx = valid_df['val_auroc'].idxmax()
        if df.loc[idx, 'val_auroc'] > best_val_auroc:
            best_val_auroc = df.loc[idx, 'val_auroc']
            best_file = csv_file.replace('.csv', '.pt')

    return best_file, best_val_auroc


def load_model(model_path, in_features, pooling, embedding_level=True, **pool_kwargs):
    if embedding_level:
        model = models.PoolClf(in_features, 1, pooling, **pool_kwargs)
    else:
        model = models.ClfPool(in_features, 1, pooling, **pool_kwargs)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def get_attention(model, X, lengths, embedding_level=True):
    with torch.no_grad():
        _, attn = model(X, lengths)
        if embedding_level:
            return attn.squeeze().numpy()
        return torch.sigmoid(model.clf(X)).squeeze().numpy()


def get_predictions(model, X, lengths):
    with torch.no_grad():
        logits, _ = model(X, lengths)
        probs = torch.sigmoid(logits).squeeze().numpy()
    return probs


def get_test_slice_labels(labels_csv, numpy_dir, seed):
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
    test_data = torch.load(f'{dataset_dir}/seed={seed}/test.pth', map_location='cpu', weights_only=False)
    return test_data['X'], test_data['lengths'], test_data['y']


def load_test_data_with_instance_labels(dataset_dir, seed):
    test_data = torch.load(f'{dataset_dir}/seed={seed}/test.pth', map_location='cpu', weights_only=False)
    lengths = test_data['lengths']
    lengths_y = test_data['lengths_y']
    instance_labels = [ly[:l] for ly, l in zip(lengths_y, lengths)]
    return test_data['X'], lengths, test_data['y'], instance_labels
