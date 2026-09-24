"""Precision@low-recall for CIA+L1, using the same fixed-beta-per-dataset selection
as eval_cia_localization_fixedbeta.py (beta chosen by mean val_auroc over the full
alpha x lr x seed grid; alpha/lr then selected per seed within that beta).
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

E = "/cluster/home/zmou01/pooling/experiments"
DATA = "/cluster/tufts/hugheslab/datasets"
SEEDS = [1001, 2001, 3001]
BETAS = ["0.2", "0.8", "1.0"]
RECALLS = [0.05, 0.10, 0.25, 0.50]

DATASETS = {
    "Semi-Synthetic": {"dir": f"{E}/synthetic_CIA_L1_embedding_level=True", "test_pt": None},
    "Head CT": {"dir": f"{E}/RSNA_ICH_full_dataset_CIA_L1_embedding_level=True",
                "test_pt": lambda s: f"{DATA}/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed={s}/test.pt"},
    "Chest CT": {"dir": f"{E}/RSNA_PE_CIA_L1_embedding_level=True",
                 "test_pt": lambda s: f"{DATA}/encoded_RSNA_PE/ViT_B_16/seed={s}/test.pt"},
    "Abdomen CT": {"dir": f"{E}/RSNA_AT_CIA_L1_embedding_level=True",
                   "test_pt": lambda s: f"{DATA}/encoded_RSNA_AT/ViT_B_16/seed={s}/test.pt"},
}


def load_test_dataset(tag, seed):
    if tag == "Semi-Synthetic":
        ds = datasets.ShiftedMeanMILDataset(n=1000, delta=0.5, r=12, seed=seed + 2)
        lengths_y = []
        for i, S in enumerate(ds.lengths):
            y_ij = [0.0] * S
            if ds.y[i] == 1:
                u = int(ds.u[i])
                for j in range(u, min(u + ds.r, S)):
                    y_ij[j] = 1.0
            lengths_y.append(y_ij)
        return ds, lengths_y
    d = torch.load(DATASETS[tag]["test_pt"](seed), map_location="cpu", weights_only=False)
    return datasets.MILTensorDataset(d["X"], d["lengths"], d["y"]), d["lengths_y"]


def choose_beta(exp_dir):
    best_beta, best_score = None, -1
    for beta in BETAS:
        vals = []
        for f in glob.glob(f"{exp_dir}/*beta_effect={beta}_*.csv"):
            df = pd.read_csv(f)
            g = df[df.train_auroc > df.val_auroc]
            if g.empty:
                continue
            vals.append(g.val_auroc.max())
        if vals:
            m = np.mean(vals)
            if m > best_score:
                best_score, best_beta = m, beta
    return best_beta


def select_ckpt(exp_dir, beta, seed):
    best = None
    for f in glob.glob(f"{exp_dir}/*beta_effect={beta}_*seed={seed}.csv"):
        df = pd.read_csv(f)
        g = df[df.train_auroc > df.val_auroc]
        if g.empty:
            continue
        i = int(g.val_auroc.idxmax())
        v = float(df.loc[i, "val_auroc"])
        if best is None or v > best[0]:
            best = (v, f[:-4] + ".pt")
    return best


def _prec_at_recall(y, s, r):
    p, rec, _ = precision_recall_curve(y, s)
    ok = rec >= r
    return float(np.max(p[ok])) if ok.any() else float("nan")


def main():
    cols = [f"P@rec{int(r * 100)}" for r in RECALLS] + ["P@1", "AUPRC"]
    out_rows = []
    for tag, info in DATASETS.items():
        beta = choose_beta(info["dir"])
        agg = {c: [] for c in cols}
        for seed in SEEDS:
            ds, ly = load_test_dataset(tag, seed)
            best = select_ckpt(info["dir"], beta, seed)
            if best is None:
                continue
            v, pt = best
            m = models.PoolClf(768, 1, pooling="ABMIL")
            sd = torch.load(pt, map_location="cpu")
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
        row = {"dataset": tag, "beta_effect": beta}
        row.update({c: f"{np.mean(agg[c]):.3f} +/- {np.std(agg[c]):.3f}" for c in cols})
        out_rows.append(row)
        print(f"{tag} (beta={beta}): " + " | ".join(f"{c}={row[c]}" for c in cols))
    pd.DataFrame(out_rows).to_csv(f"{E}/_cia_p_at_rec_fixedbeta.csv", index=False)
    print(f"wrote {E}/_cia_p_at_rec_fixedbeta.csv")


if __name__ == "__main__":
    main()
