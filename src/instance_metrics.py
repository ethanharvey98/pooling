"""Instance-level metrics for synthetic MIL models."""
import argparse
import re

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import datasets
import models


class CenterGaussian(torch.nn.Module):
    """Baseline that assigns Gaussian attention centered at the middle of each bag."""
    def forward(self, x, lengths):
        attn_weights = []
        for x_i in torch.split(x, lengths):
            n = len(x_i)
            pos = torch.linspace(1, n - 1, n)
            w = torch.exp(-0.5 * ((pos - n / 2) / 1.0) ** 2)
            attn_weights.append((w / w.sum()).unsqueeze(1))
        return torch.zeros(len(lengths), 1), torch.cat(attn_weights)


def parse_path(path):
    """Extract experiment params from the HPC path convention."""
    path = path.replace('.pt', '')
    kv = dict(re.findall(r'([a-zA-Z]\w*)=([^_/]+)', path))
    pooling = kv.pop('pooling', 'ABMIL')
    kv.pop('criterion', None)
    params = {}
    for k, v in kv.items():
        try:
            params[k] = float(v) if '.' in v else int(v)
        except ValueError:
            continue
    return params, pooling


def get_instance_labels(dataset, i):
    """Binary ground truth: 1 for instances in [u[i], u[i]+r), else 0."""
    y_ij = np.zeros(dataset.lengths[i])
    y_ij[dataset.u[i]:dataset.u[i] + dataset.r] = 1.0
    return y_ij


def evaluate_instance_metrics(model, dataset):
    """Compute per-positive-bag instance AUROC, AUPRC, attn mass on positives."""
    model.eval()
    aurocs, auprcs, attn_corrs = [], [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            embeddings, length, label = dataset[i]
            if label == 1.0:
                _, a_i = model(embeddings, (length,))
                a_i = a_i.squeeze().numpy()
                y_ij = get_instance_labels(dataset, i)
                attn_corrs.append(a_i[y_ij == 1.0].sum())
                aurocs.append(roc_auc_score(y_ij, a_i))
                auprcs.append(average_precision_score(y_ij, a_i))
    return {
        'auroc': np.mean(aurocs), 'auprc': np.mean(auprcs),
        'attn_mass': np.mean(attn_corrs),
        'aurocs': aurocs, 'auprcs': auprcs, 'attn_corrs': attn_corrs,
    }


if __name__ == '__main__':
    from collections import defaultdict

    parser = argparse.ArgumentParser(description='Instance-level metrics for synthetic MIL models')
    parser.add_argument('paths', nargs='+', help='Model .pt checkpoint paths')
    parser.add_argument('--center-gaussian', action='store_true', help='Evaluate center Gaussian baseline instead of loading checkpoints')
    args = parser.parse_args()

    grouped = defaultdict(lambda: {'auroc': [], 'auprc': [], 'attn_mass': []})

    for path in args.paths:
        params, pooling = parse_path(path)
        test_dataset = datasets.ShiftedMeanMILDataset(
            n=params['n_test'], delta=params['delta'], r=params['r'],
            s_low=params['s_low'], s_high=params['s_high'],
            seed=params['data_seed_test'],
        )
        if args.center_gaussian:
            model = CenterGaussian()
            key = 'CenterGaussian'
        else:
            model = models.PoolClf(
                in_features=768, out_features=1, pooling=pooling,
                neighbors=params.get('neighbors', 1),
            )
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
            key = pooling
        results = evaluate_instance_metrics(model, test_dataset)
        grouped[key]['auroc'].append(results['auroc'])
        grouped[key]['auprc'].append(results['auprc'])
        grouped[key]['attn_mass'].append(results['attn_mass'])
        print(f"[seed={params['seed']}] {key}  AUROC={results['auroc']:.4f}  AUPRC={results['auprc']:.4f}  Attn mass={results['attn_mass']:.4f}")

    print("\n=== Summary (mean +/- std across seeds) ===")
    for key, metrics in grouped.items():
        for name in ('auroc', 'auprc', 'attn_mass'):
            vals = np.array(metrics[name])
            print(f"  {key} {name}: {vals.mean():.4f} +/- {vals.std():.4f}")
