"""Best-in-Class Ceiling for Semi-Synthetic: the analytic Bayes-optimal per-instance
posterior p(y_ij=1 | h_i) from ShiftedMeanMILDataset.p_y_j1_given_h (see
notebooks/bayes-estimator.ipynb), not a trained InstanceLevelClassifier -- since the
true generative process is known exactly for this dataset, this is a tighter ceiling
than the trained-classifier approximation used for the real CT datasets.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import datasets  # noqa: E402

SEEDS = [1001, 2001, 3001]
SEMI_TEST_SEED = {1001: 1003, 2001: 2003, 3001: 3003}
RECALLS = [0.05, 0.10, 0.25, 0.50]


def _prec_at_recall(y, s, r):
    p, rec, _ = precision_recall_curve(y, s)
    ok = rec >= r
    return float(np.max(p[ok])) if ok.any() else float("nan")


def main():
    cols = [f"P@rec{int(r * 100)}" for r in RECALLS] + ["P@1", "AUPRC"]
    agg = {c: [] for c in cols}
    for seed in SEEDS:
        ds = datasets.ShiftedMeanMILDataset(n=1000, r=12, s_low=20, s_high=60, delta=0.5, seed=SEMI_TEST_SEED[seed])
        per_bag = {c: [] for c in cols}
        for i in range(len(ds)):
            h_i, S_i, y_i = ds[i]
            if y_i != 1.0:
                continue
            u = int(ds.u[i])
            y_ij = np.zeros(int(S_i))
            y_ij[u:min(u + ds.r, int(S_i))] = 1.0
            if y_ij.sum() == 0 or y_ij.sum() == len(y_ij):
                continue
            a = ds.p_y_j1_given_h(i).numpy().flatten()
            for r in RECALLS:
                per_bag[f"P@rec{int(r * 100)}"].append(_prec_at_recall(y_ij, a, r))
            per_bag["P@1"].append(float(y_ij[int(np.argmax(a))]))
            per_bag["AUPRC"].append(average_precision_score(y_ij, a))
        for c in cols:
            agg[c].append(np.mean(per_bag[c]))
        print(f"seed {seed}: P@rec10={np.mean(per_bag['P@rec10']):.3f}")

    row = {"dataset": "Semi-Synthetic"}
    row.update({c: f"{np.mean(agg[c]):.3f} +/- {np.std(agg[c]):.3f}" for c in cols})
    print("Semi-Synthetic (Bayes ceiling): " + " | ".join(f"{c}={row[c]}" for c in cols))

    pd.DataFrame([row]).to_csv("/cluster/home/zmou01/pooling/experiments/ceiling_p_at_rec10_semi.csv", index=False)
    print("wrote /cluster/home/zmou01/pooling/experiments/ceiling_p_at_rec10_semi.csv")


if __name__ == "__main__":
    main()
