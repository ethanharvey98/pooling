"""Precision@low-recall for Additive MIL and AEM on Semi-Synthetic (newly available
checkpoints under varying_n_embedding_level=True[_beta=0.001_AEM]).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import datasets  # noqa: E402
import models  # noqa: E402

EH = "/cluster/tufts/hugheslab/eharve06/pooling/experiments"
SEEDS = [1001, 2001, 3001]
ALPHAS = ["0.0", "1e-06", "1e-05", "0.0001", "0.001", "0.01", "0.1", "1.0"]
LRS = ["0.0001", "0.001", "0.01", "0.1"]
RECALLS = [0.05, 0.10, 0.25, 0.50]
SUFFIX = "delta=0.5_r=12_s_low=20_s_high=60"
SUBDIR = {
    1001: ("data_seed_test=1003_data_seed_train=1001_data_seed_val=1002_n_test=1000_n_train=10000_n_val=2500", 1003),
    2001: ("data_seed_test=2003_data_seed_train=2001_data_seed_val=2002_n_test=1000_n_train=10000_n_val=2500", 2003),
    3001: ("data_seed_test=3003_data_seed_train=3001_data_seed_val=3002_n_test=1000_n_train=10000_n_val=2500", 3003),
}

METHODS = {
    "Additive MIL": {"dir_name": "varying_n_embedding_level=True", "pooling": "AdditiveMIL"},
    "AEM": {"dir_name": "varying_n_embedding_level=True_beta=0.001_AEM", "pooling": "ABMIL"},
}


def load_test(seed):
    _, test_seed = SUBDIR[seed]
    ds = datasets.ShiftedMeanMILDataset(n=1000, r=12, s_low=20, s_high=60, delta=0.5, seed=test_seed)
    lengths_y = []
    for i, S in enumerate(ds.lengths):
        y_ij = [0.0] * int(S)
        if ds.y[i] == 1:
            u = int(ds.u[i])
            for j in range(u, min(u + ds.r, int(S))):
                y_ij[j] = 1.0
        lengths_y.append(y_ij)
    return ds, lengths_y


def bag_dir(dir_name, seed):
    subdir, _ = SUBDIR[seed]
    return f"{EH}/{dir_name}/{SUFFIX}/{subdir}"


def select(dir_, pooling, seed):
    best = None
    for alpha in ALPHAS:
        for lr in LRS:
            name = f"alpha={alpha}_criterion=L1_lr={lr}_pooling={pooling}_seed={seed}"
            csv = f"{dir_}/{name}.csv"
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv)
            g = df[df.train_auroc > df.val_auroc]
            if g.empty:
                continue
            v = g.val_auroc.max()
            if best is None or v > best[0]:
                best = (v, name)
    return best


def _prec_at_recall(y, s, r):
    p, rec, _ = precision_recall_curve(y, s)
    ok = rec >= r
    return float(np.max(p[ok])) if ok.any() else float("nan")


def main():
    cols = [f"P@rec{int(r * 100)}" for r in RECALLS] + ["P@1", "AUPRC"]
    out_rows = []
    for method_name, cfg in METHODS.items():
        agg = {c: [] for c in cols}
        for seed in SEEDS:
            ds, ly = load_test(seed)
            d = bag_dir(cfg["dir_name"], seed)
            best = select(d, cfg["pooling"], seed)
            if best is None:
                print(f"[skip] {method_name} seed {seed}: no checkpoint")
                continue
            v, name = best
            if cfg["pooling"] == "AdditiveMIL":
                m = models.AdditiveMIL(768, 1)
            else:
                m = models.PoolClf(768, 1, pooling="ABMIL")
            sd = torch.load(f"{d}/{name}.pt", map_location="cpu")
            sd = sd.get("state_dict", sd)
            if cfg["pooling"] == "AdditiveMIL" and "pool.mlp.0.weight" in sd:
                sd = {k.replace("pool.", "", 1) if k.startswith("pool.") else k: v2 for k, v2 in sd.items()}
            elif cfg["pooling"] != "AdditiveMIL" and "mlp.0.weight" in sd:  # legacy flat ABMIL checkpoint
                sd = {("pool." + k if k.startswith("mlp.") else k): v2 for k, v2 in sd.items()}
            m.load_state_dict(sd)
            m.eval()
            per_bag = {c: [] for c in cols}
            with torch.no_grad():
                for i in range(len(ds)):
                    h_i, S_i, y_i = ds[i]
                    y_ij = np.asarray(ly[i], dtype=float)
                    if y_i != 1.0 or len(y_ij) != S_i or y_ij.sum() == 0 or y_ij.sum() == len(y_ij):
                        continue
                    _, a = m(h_i, (int(S_i),))
                    a = torch.mean(a, dim=1).cpu().numpy().flatten()
                    for r in RECALLS:
                        per_bag[f"P@rec{int(r * 100)}"].append(_prec_at_recall(y_ij, a, r))
                    per_bag["P@1"].append(float(y_ij[int(np.argmax(a))]))
                    per_bag["AUPRC"].append(average_precision_score(y_ij, a))
            for c in cols:
                agg[c].append(np.mean(per_bag[c]))
            print(f"  {method_name} seed {seed}: {name} val={v:.4f}")
        row = {"method": method_name, "dataset": "Semi-Synthetic"}
        row.update({c: f"{np.mean(agg[c]):.3f} +/- {np.std(agg[c]):.3f}" for c in cols})
        out_rows.append(row)
        print(f"{method_name}: " + " | ".join(f"{c}={row[c]}" for c in cols))
    pd.DataFrame(out_rows).to_csv("/cluster/home/zmou01/pooling/experiments/_semi_additivemil_aem_p_at_rec.csv", index=False)
    print("wrote _semi_additivemil_aem_p_at_rec.csv")


if __name__ == "__main__":
    main()
