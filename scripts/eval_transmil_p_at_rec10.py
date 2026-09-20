"""Precision@10%Recall for the TransMIL row only, across
Semi-Synthetic, Chest CT (RSNA PE), and Abdomen CT (RSNA AT) -- Head CT is already
computed (0.841 +/- 0.009, from precision_at_low_recall_rsna_ich.csv).

Uses eharve06's existing GuidedL1+ABMIL checkpoints referenced by the notebooks
(rsna_pe.ipynb / rsna_at.ipynb / synthetic_data.ipynb): the "beta=1.0" dirs. No new
training. Per (dataset, seed): select (alpha, lr) by val_auroc among epochs with
train_auroc > val_auroc (standard project rule), then score precision@recall on
positive test bags.

Run from the repo root in the neuroimg_gpu env:
    python scripts/eval_transmil_p_at_rec10.py
"""
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
DATA = "/cluster/tufts/hugheslab/datasets"
SEEDS = [1001, 2001, 3001]
ALPHAS = ["0.0", "1e-06", "1e-05", "0.0001", "0.001", "0.01", "0.1", "1.0"]
LRS = ["0.0001", "0.001", "0.01", "0.1"]
RECALLS = [0.05, 0.10, 0.25, 0.50]

SEMI_SUFFIX = "delta=0.5_r=12_s_low=20_s_high=60"
SEMI_SUBDIR = {
    1001: "data_seed_test=1003_data_seed_train=1001_data_seed_val=1002_n_test=1000_n_train=10000_n_val=2500",
    2001: "data_seed_test=2003_data_seed_train=2001_data_seed_val=2002_n_test=1000_n_train=10000_n_val=2500",
    3001: "data_seed_test=3003_data_seed_train=3001_data_seed_val=3002_n_test=1000_n_train=10000_n_val=2500",
}
SEMI_TEST_SEED = {1001: 1003, 2001: 2003, 3001: 3003}


def semi_dir(seed):
    return f"{EH}/varying_n_embedding_level=True/{SEMI_SUFFIX}/{SEMI_SUBDIR[seed]}"


DATASET_CFG = {
    "Semi-Synthetic": {"kind": "semi"},
    "Chest CT (RSNA PE)": {"kind": "std", "ng_dir": f"{EH}/RSNA_PE_embedding_level=True",
                             "data_dir": f"{DATA}/encoded_RSNA_PE/ViT_B_16"},
    "Abdomen CT (RSNA AT)": {"kind": "std", "ng_dir": f"{EH}/RSNA_AT_embedding_level=True",
                               "data_dir": f"{DATA}/encoded_RSNA_AT/ViT_B_16"},
}


def _select(dir_, seed):
    best = None
    for alpha in ALPHAS:
        for lr in LRS:
            name = f"alpha={alpha}_criterion=L1_lr={lr}_pooling=TransMIL_seed={seed}"
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
    if best is None:
        raise FileNotFoundError(f"no valid GuidedL1 ABMIL checkpoint in {dir_} for seed {seed}")
    return best


def _prec_at_recall(y, s, r):
    p, rec, _ = precision_recall_curve(y, s)
    ok = rec >= r
    return float(np.max(p[ok])) if ok.any() else float("nan")


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


def load_std_test(cfg, seed):
    d = torch.load(f"{cfg['data_dir']}/seed={seed}/test.pt", map_location="cpu", weights_only=False)
    return datasets.MILTensorDataset(d["X"], d["lengths"], d["y"]), d["lengths_y"]


def main():
    cols = [f"P@rec{int(r * 100)}" for r in RECALLS] + ["P@1", "AUPRC"]
    out_rows = []
    for tag, cfg in DATASET_CFG.items():
        print(f"\n########## {tag} ##########", flush=True)
        agg = {c: [] for c in cols}
        picks = []
        for seed in SEEDS:
            if cfg["kind"] == "semi":
                ds, ly = load_semi_test(seed)
                v, name = _select(semi_dir(seed), seed)
                ckpt = f"{semi_dir(seed)}/{name}.pt"
            else:
                ds, ly = load_std_test(cfg, seed)
                v, name = _select(cfg["ng_dir"], seed)
                ckpt = f"{cfg['ng_dir']}/{name}.pt"
            picks.append(f"seed{seed}:{name}(val={v:.4f})")

            m = models.PoolClf(768, 1, pooling="TransMIL")
            sd = torch.load(ckpt, map_location="cpu")
            m.load_state_dict(sd.get("state_dict", sd))
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
            print(f"  seed {seed}: {name} val={v:.4f}  P@rec10={np.mean(per_bag['P@rec10']):.3f}", flush=True)

        row = {"dataset": tag}
        row.update({c: f"{np.mean(agg[c]):.3f} +/- {np.std(agg[c]):.3f}" for c in cols})
        row["picks"] = " ".join(picks)
        out_rows.append(row)
        print(f"{tag}: " + " | ".join(f"{c}={row[c]}" for c in cols), flush=True)

    out = pd.DataFrame(out_rows)
    out_csv = "/cluster/home/zmou01/pooling/experiments/transmil_p_at_rec10_3datasets.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
