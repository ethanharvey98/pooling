"""Localization AUROC/AUPRC for Centered Gaussian (empirical mu/sigma, log-space),
across all 4 datasets -- matches Table 2's "Centered Gaussian" row protocol
(roc_auc_score / average_precision_score per positive bag, macro-averaged).
"""
import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import datasets  # noqa: E402
import utils  # noqa: E402

DATA = "/cluster/tufts/hugheslab/datasets"
SEEDS = [1001, 2001, 3001]
SEMI_TEST_SEED = {1001: 1003, 2001: 2003, 3001: 3003}
M_EMP, S_EMP = 0.52, 0.115


def load_semi_test(seed):
    ds = datasets.ShiftedMeanMILDataset(n=1000, r=12, s_low=20, s_high=60, delta=0.5, seed=SEMI_TEST_SEED[seed])
    lengths_y = []
    for i, S in enumerate(ds.lengths):
        y_ij = [0.0] * int(S)
        if ds.y[i] == 1:
            u = int(ds.u[i])
            for j in range(u, min(u + ds.r, int(S))):
                y_ij[j] = 1.0
        lengths_y.append(y_ij)
    return ds, lengths_y


def load_std_test(dataset_dir, seed):
    d = torch.load(f"{dataset_dir}/seed={seed}/test.pt", map_location="cpu", weights_only=False)
    return datasets.MILTensorDataset(d["X"], d["lengths"], d["y"]), d["lengths_y"]


DATASETS = {
    "Semi-Synthetic": {"kind": "semi"},
    "Head CT": {"kind": "std", "data_dir": f"{DATA}/encoded_RSNA_ICH_full_dataset/ViT_B_16"},
    "Chest CT": {"kind": "std", "data_dir": f"{DATA}/encoded_RSNA_PE/ViT_B_16"},
    "Abdomen CT": {"kind": "std", "data_dir": f"{DATA}/encoded_RSNA_AT/ViT_B_16"},
}


def score_cg(S):
    x = torch.arange(1, S + 1, dtype=torch.float32)
    a = utils.log_normal_pdf(x, mu=M_EMP * S, sigma=S_EMP * S)
    return a.numpy()


def main():
    for tag, cfg in DATASETS.items():
        ars, aps = [], []
        for seed in SEEDS:
            if cfg["kind"] == "semi":
                ds, ly = load_semi_test(seed)
            else:
                ds, ly = load_std_test(cfg["data_dir"], seed)
            bag_ar, bag_ap = [], []
            for i in range(len(ds)):
                h_i, S_i, y_i = ds[i]
                y_ij = np.asarray(ly[i], dtype=float)
                if y_i != 1.0 or len(y_ij) != S_i or y_ij.sum() == 0 or y_ij.sum() == len(y_ij):
                    continue
                a = score_cg(int(S_i))
                bag_ar.append(roc_auc_score(y_ij, a))
                bag_ap.append(average_precision_score(y_ij, a))
            ars.append(np.mean(bag_ar))
            aps.append(np.mean(bag_ap))
        print(f"{tag}: AUROC={np.mean(ars):.3f} +/- {np.std(ars):.3f}   AUPRC={np.mean(aps):.3f} +/- {np.std(aps):.3f}")


if __name__ == "__main__":
    main()
