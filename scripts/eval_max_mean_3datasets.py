"""Max pooling / Mean pooling baselines (no checkpoint needed) for Precision@low-recall,
Semi-Synthetic / Chest CT (RSNA PE) / Abdomen CT (RSNA AT). Head CT already computed
(Max=0.396+/-0.008, Mean=0.359+/-0.009, precision_at_low_recall_rsna_ich.csv).
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import datasets  # noqa: E402

DATA = "/cluster/tufts/hugheslab/datasets"
SEEDS = [1001, 2001, 3001]
RECALLS = [0.05, 0.10, 0.25, 0.50]
SEMI_TEST_SEED = {1001: 1003, 2001: 2003, 3001: 3003}


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


DATASET_CFG = {
    "Semi-Synthetic": {"kind": "semi"},
    "Chest CT (RSNA PE)": {"kind": "std", "data_dir": f"{DATA}/encoded_RSNA_PE/ViT_B_16"},
    "Abdomen CT (RSNA AT)": {"kind": "std", "data_dir": f"{DATA}/encoded_RSNA_AT/ViT_B_16"},
}


def score_max(h, S):
    idx = torch.argmax(h, dim=0)
    return np.array([(idx == j).sum().item() for j in range(S)], dtype=float) / h.shape[1]


def score_mean(S):
    return np.full(S, 1.0 / S)


def _prec_at_recall(y, s, r):
    p, rec, _ = precision_recall_curve(y, s)
    ok = rec >= r
    return float(np.max(p[ok])) if ok.any() else float("nan")


def main():
    cols = [f"P@rec{int(r * 100)}" for r in RECALLS] + ["P@1", "AUPRC"]
    out_rows = []
    for method_name, scorer in [("Max pooling", "max"), ("Mean pooling", "mean")]:
        for tag, cfg in DATASET_CFG.items():
            agg = {c: [] for c in cols}
            for seed in SEEDS:
                if cfg["kind"] == "semi":
                    ds, ly = load_semi_test(seed)
                else:
                    ds, ly = load_std_test(cfg["data_dir"], seed)
                per_bag = {c: [] for c in cols}
                for i in range(len(ds)):
                    h_i, S_i, y_i = ds[i]
                    y_ij = np.asarray(ly[i], dtype=float)
                    if y_i != 1.0 or len(y_ij) != S_i or y_ij.sum() == 0 or y_ij.sum() == len(y_ij):
                        continue
                    a = score_max(h_i, int(S_i)) if scorer == "max" else score_mean(int(S_i))
                    for r in RECALLS:
                        per_bag[f"P@rec{int(r * 100)}"].append(_prec_at_recall(y_ij, a, r))
                    per_bag["P@1"].append(float(y_ij[int(np.argmax(a))]))
                    per_bag["AUPRC"].append(average_precision_score(y_ij, a))
                for c in cols:
                    agg[c].append(np.mean(per_bag[c]))
            row = {"method": method_name, "dataset": tag}
            row.update({c: f"{np.mean(agg[c]):.3f} +/- {np.std(agg[c]):.3f}" for c in cols})
            out_rows.append(row)
            print(f"{method_name} | {tag}: " + " | ".join(f"{c}={row[c]}" for c in cols))

    out = pd.DataFrame(out_rows)
    out_csv = "/cluster/home/zmou01/pooling/experiments/max_mean_3datasets.csv"
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
