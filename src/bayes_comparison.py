"""Instance-level Bayes scoring: independent vs full-bag."""
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import datasets
import utils


def independent_scores(dataset, index):
    """Score each instance alone: log p(h_j | anomalous) - log p(h_j | normal)."""
    h = dataset.h_split[index][:, 0:dataset.k]
    return torch.stack([
        torch.sum(torch.log(utils.normal_pdf(h[j], mu=dataset.delta)) - torch.log(utils.normal_pdf(h[j])))
        for j in range(dataset.lengths[index])
    ])


if __name__ == '__main__':
    delta, r, n_test = 0.5, 12, 100
    seeds = [1003, 2003, 3003]

    for name, score_fn in [
        ('Independent', independent_scores),
        ('Full-bag',    lambda ds, i: ds.p_instance_given_h(i)),  # new dataset method
    ]:
        aurocs, auprcs = [], []
        for seed in seeds:
            ds = datasets.ShiftedMeanMILDataset(n=n_test, delta=delta, r=r, seed=seed)
            seed_aurocs, seed_auprcs = [], []
            for i in range(len(ds)):
                if ds.y[i] == 1.0:
                    y_ij = np.zeros(ds.lengths[i])
                    y_ij[ds.u[i]:ds.u[i] + ds.r] = 1.0
                    scores = score_fn(ds, i).numpy()
                    seed_aurocs.append(roc_auc_score(y_ij, scores))
                    seed_auprcs.append(average_precision_score(y_ij, scores))
            aurocs.append(np.mean(seed_aurocs))
            auprcs.append(np.mean(seed_auprcs))
        print(f"{name:12s}  AUROC = {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}  AUPRC = {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}")
