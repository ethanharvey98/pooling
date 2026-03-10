#!/usr/bin/env python3
"""
Find best VAP hyperparameters by model type, based on validation AUROC.
Optionally use MC uncertainty to filter out most uncertain scans and re-evaluate bag-level AUROC.

Usage:
    python find_best_vap_results.py /path/to/results
    python find_best_vap_results.py /path/to/results --uncertain --mc_samples 20 --filter_pct 0.2
"""

import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import models


DATASET_DIR = '/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16'
MODEL_TYPES = ['VAPGaussian', 'VAPGaussianCenter', 'VAPBernoulli', 'VAPGaussianSparse']
SEEDS = [1001, 2001, 3001]


def find_best_for_seed(csv_files):
    """Find the best model (by val_auroc) among a list of CSVs and return metrics."""
    best_val_auroc = -1
    best_info = None

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'val_auroc' not in df.columns or 'test_auroc' not in df.columns:
                continue

            if 'train_auroc' in df.columns:
                valid_df = df[df['val_auroc'] <= df['train_auroc']]
                if valid_df.empty:
                    continue
            else:
                valid_df = df

            idx = valid_df['val_auroc'].idxmax()
            val_auroc = df.loc[idx, 'val_auroc']

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_info = {
                    'file': os.path.basename(csv_file),
                    'pt_file': csv_file.replace('.csv', '.pt'),
                    'epoch': df.loc[idx, 'epoch'] if 'epoch' in df.columns else idx,
                    'val_auroc': val_auroc,
                    'test_auroc': df.loc[idx, 'test_auroc'],
                    'train_auroc': df.loc[idx, 'train_auroc'] if 'train_auroc' in df.columns else None,
                    'test_kl': df.loc[idx, 'test_kl'] if 'test_kl' in df.columns else None,
                }
        except Exception:
            continue

    return best_info


def load_vap_model(model_path, in_features, model_type):
    """Load a VAP model from checkpoint."""
    if model_type in ('VAPGaussian', 'VAPGaussianCenter'):
        model = models.VAPGaussianMIL(in_features=in_features, out_features=1)
    elif model_type == 'VAPBernoulli':
        model = models.VAPBernoulliMIL(in_features=in_features, out_features=1)
    elif model_type == 'VAPGaussianSparse':
        model = models.VAPGaussianSparseMIL(in_features=in_features, out_features=1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def load_test_data(dataset_dir, seed):
    test_data = torch.load(f'{dataset_dir}/seed={seed}/test.pth', map_location='cpu', weights_only=False)
    return test_data['X'], test_data['lengths'], test_data['y']


def evaluate_with_uncertainty(model, X, lengths, y, mc_samples, filter_pct):
    """Run MC inference with three uncertainty strategies."""
    unc = model.predict_with_uncertainty(X, lengths, n_samples=mc_samples)
    logits_mean = unc['logits_mean'].squeeze().numpy()
    logits_var = unc['logits_var'].squeeze().numpy()
    attn_samples = unc['attn_samples']  # (mc_samples, N_total, 1)
    probs = 1.0 / (1.0 + np.exp(-logits_mean))
    labels = y.squeeze().numpy()

    # All scans (baseline)
    auroc_all = roc_auc_score(labels, probs)

    # Filter out top filter_pct most uncertain
    threshold = np.percentile(logits_var, (1 - filter_pct) * 100)
    keep = logits_var <= threshold
    if len(np.unique(labels[keep])) > 1:
        auroc_filtered = roc_auc_score(labels[keep], probs[keep])
    else:
        auroc_filtered = float('nan')

    # Strategy 1: Mean-pool uncertain bags
    # For uncertain bags, re-pool with uniform weights and re-classify
    uncertain_mask = logits_var > threshold
    probs_meanpool = probs.copy()
    clf_weight = model.clf.weight.detach()
    clf_bias = model.clf.bias.detach()
    start = 0
    for i, length in enumerate(lengths):
        if uncertain_mask[i]:
            x_bag = X[start:start + length]
            # Mean pool instead of attention pool
            pooled = x_bag.mean(dim=0, keepdim=True)
            logit = (pooled @ clf_weight.T + clf_bias).item()
            probs_meanpool[i] = 1.0 / (1.0 + np.exp(-logit))
        start += length
    auroc_meanpool = roc_auc_score(labels, probs_meanpool)

    # Strategy 2: Zero out most uncertain instances within each bag
    # Per-instance attention variance across MC samples
    attn_var = attn_samples.var(dim=0).squeeze().numpy()  # (N_total,)
    probs_zeroed = np.zeros(len(labels))
    start = 0
    for i, length in enumerate(lengths):
        x_bag = X[start:start + length]
        inst_var = attn_var[start:start + length]

        # Zero out top filter_pct most uncertain instances in this bag
        n_zero = max(1, int(length * filter_pct))
        zero_idx = np.argsort(inst_var)[-n_zero:]  # highest variance instances
        # Get mean attention across MC samples
        mean_attn = attn_samples[:, start:start + length, 0].mean(dim=0).numpy()
        mean_attn[zero_idx] = 0.0
        attn_sum = mean_attn.sum()
        if attn_sum > 0:
            mean_attn = mean_attn / attn_sum
        else:
            mean_attn = np.ones(length) / length

        # Re-pool with modified attention
        pooled = torch.from_numpy(mean_attn).unsqueeze(1).float() * x_bag
        pooled = pooled.sum(dim=0, keepdim=True)
        logit = (pooled @ clf_weight.T + clf_bias).item()
        probs_zeroed[i] = 1.0 / (1.0 + np.exp(-logit))
        start += length
    auroc_zeroed = roc_auc_score(labels, probs_zeroed)

    # Strategy 3: Use slice uncertainty as attention logits
    # High variance = uncertain = attend more (var_as_logits)
    # Low variance = certain = attend more (neg_var_as_logits)
    probs_var_attn = np.zeros(len(labels))
    probs_neg_var_attn = np.zeros(len(labels))
    start = 0
    for i, length in enumerate(lengths):
        x_bag = X[start:start + length]
        inst_var = attn_var[start:start + length]

        # Softmax over variance -> attend to uncertain slices
        var_logits = torch.from_numpy(inst_var).float()
        var_weights = torch.nn.functional.softmax(var_logits, dim=0).unsqueeze(1)
        pooled = (var_weights * x_bag).sum(dim=0, keepdim=True)
        logit = (pooled @ clf_weight.T + clf_bias).item()
        probs_var_attn[i] = 1.0 / (1.0 + np.exp(-logit))

        # Softmax over negative variance -> attend to certain slices
        neg_var_weights = torch.nn.functional.softmax(-var_logits, dim=0).unsqueeze(1)
        pooled = (neg_var_weights * x_bag).sum(dim=0, keepdim=True)
        logit = (pooled @ clf_weight.T + clf_bias).item()
        probs_neg_var_attn[i] = 1.0 / (1.0 + np.exp(-logit))

        start += length
    auroc_var_attn = roc_auc_score(labels, probs_var_attn)
    auroc_neg_var_attn = roc_auc_score(labels, probs_neg_var_attn)

    # Strategy 4: Adaptive softmax temperature from bag-level uncertainty
    # tau = 1 + scale * normalized_var  (uncertain bags get higher temp -> softer attention)
    # Get the learned attention logits (mu) from the model
    with torch.no_grad():
        _, _ = model(X, lengths)  # populate _mu
    attn_logits_mu = model.pool._mu.squeeze().numpy()  # (N_total,)

    # Normalize bag-level variance to [0, 1] range
    var_min, var_max = logits_var.min(), logits_var.max()
    if var_max > var_min:
        norm_var = (logits_var - var_min) / (var_max - var_min)
    else:
        norm_var = np.zeros_like(logits_var)

    probs_adaptive_temp = np.zeros(len(labels))
    start = 0
    for i, length in enumerate(lengths):
        x_bag = X[start:start + length]
        mu_bag = torch.from_numpy(attn_logits_mu[start:start + length]).float()

        # tau: certain bags -> ~1 (sharp), uncertain bags -> higher (softer)
        tau = 1.0 + 4.0 * norm_var[i]  # ranges from 1 to 5
        attn_w = torch.nn.functional.softmax(mu_bag / tau, dim=0).unsqueeze(1)
        pooled = (attn_w * x_bag).sum(dim=0, keepdim=True)
        logit = (pooled @ clf_weight.T + clf_bias).item()
        probs_adaptive_temp[i] = 1.0 / (1.0 + np.exp(-logit))
        start += length
    auroc_adaptive_temp = roc_auc_score(labels, probs_adaptive_temp)

    return {
        'auroc_all': auroc_all,
        'auroc_filtered': auroc_filtered,
        'auroc_meanpool': auroc_meanpool,
        'auroc_zeroed': auroc_zeroed,
        'auroc_var_attn': auroc_var_attn,
        'auroc_neg_var_attn': auroc_neg_var_attn,
        'auroc_adaptive_temp': auroc_adaptive_temp,
        'n_total': len(labels),
        'n_kept': int(keep.sum()),
        'n_uncertain': int(uncertain_mask.sum()),
        'var_mean': float(logits_var.mean()),
        'var_threshold': float(threshold),
    }


def main():
    parser = argparse.ArgumentParser(description='Find best VAP results by model type')
    parser.add_argument('folder', type=str, help='Path to folder containing CSV result files')
    parser.add_argument('--dataset_dir', default=DATASET_DIR, type=str)
    parser.add_argument('--uncertain', action='store_true', default=False,
                        help='Re-evaluate best models with MC uncertainty filtering')
    parser.add_argument('--mc_samples', default=10, type=int,
                        help='Number of MC forward passes (default: 10)')
    parser.add_argument('--filter_pct', default=0.2, type=float,
                        help='Fraction of most uncertain scans to filter out (default: 0.2)')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory")
        return

    csv_files = glob.glob(os.path.join(args.folder, '*.csv'))
    if not csv_files:
        print("No CSV files found yet.")
        return

    mode_str = "with uncertainty filtering" if args.uncertain else "deterministic"
    print(f"\nSearching in: {args.folder}")
    print(f"Found {len(csv_files)} CSV files — {mode_str}")
    if args.uncertain:
        print(f"  MC samples: {args.mc_samples}, filtering top {args.filter_pct*100:.0f}% most uncertain")
    print()

    for model_type in MODEL_TYPES:
        print("=" * 80)
        print(f"  {model_type}")
        print("=" * 80)

        test_aurocs = []
        filtered_aurocs = []
        meanpool_aurocs = []
        zeroed_aurocs = []
        var_attn_aurocs = []
        neg_var_attn_aurocs = []
        adaptive_temp_aurocs = []

        for seed in SEEDS:
            seed_files = [f for f in csv_files
                          if f"model={model_type}_" in os.path.basename(f)
                          and f"seed={seed}" in os.path.basename(f)]

            if not seed_files:
                print(f"  Seed {seed}: no results yet")
                continue

            best = find_best_for_seed(seed_files)
            if best is None:
                print(f"  Seed {seed}: no valid results yet")
                continue

            test_aurocs.append(best['test_auroc'])
            kl_str = f"  KL: {best['test_kl']:.4f}" if best['test_kl'] is not None else ""
            print(f"  Seed {seed}: test_auroc={best['test_auroc']:.4f}  val_auroc={best['val_auroc']:.4f}  epoch={best['epoch']}{kl_str}")
            print(f"           {best['file']}")

            if args.uncertain and os.path.exists(best['pt_file']):
                X, lengths, y = load_test_data(args.dataset_dir, seed)
                model = load_vap_model(best['pt_file'], X.shape[1], model_type)
                unc_result = evaluate_with_uncertainty(model, X, lengths, y,
                                                       args.mc_samples, args.filter_pct)
                filtered_aurocs.append(unc_result['auroc_filtered'])
                meanpool_aurocs.append(unc_result['auroc_meanpool'])
                zeroed_aurocs.append(unc_result['auroc_zeroed'])
                var_attn_aurocs.append(unc_result['auroc_var_attn'])
                neg_var_attn_aurocs.append(unc_result['auroc_neg_var_attn'])
                adaptive_temp_aurocs.append(unc_result['auroc_adaptive_temp'])
                pct = args.filter_pct * 100
                print(f"           AUROC (all):               {unc_result['auroc_all']:.4f}")
                print(f"           AUROC (drop uncertain):    {unc_result['auroc_filtered']:.4f}  "
                      f"(kept {unc_result['n_kept']}/{unc_result['n_total']})")
                print(f"           AUROC (meanpool uncertain):{unc_result['auroc_meanpool']:.4f}  "
                      f"(meanpool {unc_result['n_uncertain']} bags)")
                print(f"           AUROC (zero uncertain inst):{unc_result['auroc_zeroed']:.4f}  "
                      f"(zeroed top {pct:.0f}% inst per bag)")
                print(f"           AUROC (var as attn logits): {unc_result['auroc_var_attn']:.4f}  "
                      f"(attend to uncertain)")
                print(f"           AUROC (-var as attn logits):{unc_result['auroc_neg_var_attn']:.4f}  "
                      f"(attend to certain)")
                print(f"           AUROC (adaptive temp):     {unc_result['auroc_adaptive_temp']:.4f}  "
                      f"(tau=1+4*norm_var)")

        if test_aurocs:
            mean = np.mean(test_aurocs)
            std = np.std(test_aurocs)
            print(f"\n  >> {model_type}: {mean:.4f} +/- {std:.4f}  (n={len(test_aurocs)})")

        if filtered_aurocs:
            print(f"  >> drop uncertain bags:     {np.mean(filtered_aurocs):.4f} +/- {np.std(filtered_aurocs):.4f}")
            print(f"  >> meanpool uncertain bags:  {np.mean(meanpool_aurocs):.4f} +/- {np.std(meanpool_aurocs):.4f}")
            print(f"  >> zero uncertain inst:      {np.mean(zeroed_aurocs):.4f} +/- {np.std(zeroed_aurocs):.4f}")
            print(f"  >> var as attn (uncertain):  {np.mean(var_attn_aurocs):.4f} +/- {np.std(var_attn_aurocs):.4f}")
            print(f"  >> -var as attn (certain):   {np.mean(neg_var_attn_aurocs):.4f} +/- {np.std(neg_var_attn_aurocs):.4f}")
            print(f"  >> adaptive temp:            {np.mean(adaptive_temp_aurocs):.4f} +/- {np.std(adaptive_temp_aurocs):.4f}")

        print()


if __name__ == "__main__":
    main()
