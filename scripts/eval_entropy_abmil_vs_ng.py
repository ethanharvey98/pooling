"""Attention entropy of ABMIL (L1, no guidance) vs ABMIL + Normal Guidance on positive
test bags, all 4 datasets. Same checkpoints/selection as the Precision@Recall tables.
Reports raw entropy H = -sum a log a and length-normalized entropy H / log(S)
(1.0 = perfectly uniform / mean pooling, 0 = one-hot).
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import datasets  # noqa: E402
import models  # noqa: E402

EH = "/cluster/tufts/hugheslab/eharve06/pooling/experiments"
HOME_EXP = "/cluster/home/zmou01/pooling/experiments"
DATA = "/cluster/tufts/hugheslab/datasets"
SEEDS = [1001, 2001, 3001]
ALPHAS = ["0.0", "1e-06", "1e-05", "0.0001", "0.001", "0.01", "0.1", "1.0"]
LRS = ["0.0001", "0.001", "0.01", "0.1"]
SEMI_SUFFIX = "delta=0.5_r=12_s_low=20_s_high=60"
SEMI_SUBDIR = {
    1001: ("data_seed_test=1003_data_seed_train=1001_data_seed_val=1002_n_test=1000_n_train=10000_n_val=2500", 1003),
    2001: ("data_seed_test=2003_data_seed_train=2001_data_seed_val=2002_n_test=1000_n_train=10000_n_val=2500", 2003),
    3001: ("data_seed_test=3003_data_seed_train=3001_data_seed_val=3002_n_test=1000_n_train=10000_n_val=2500", 3003),
}
ICH_ABMIL_PICKS = {1001: ("0.0001", "0.001"), 2001: ("0.0001", "0.1"), 3001: ("1e-05", "0.01")}
FKL_DIR = f"{HOME_EXP}/RSNA_ICH_full_dataset_NG_variance_ablation_fkl_sweep_embedding_level=True"


def select(dir_, name_fn, seed, alphas=ALPHAS, lrs=LRS):
    best = None
    for a in alphas:
        for lr in lrs:
            name = name_fn(a, lr, seed)
            csv = f"{dir_}/{name}.csv"
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv)
            g = df[df.train_auroc > df.val_auroc]
            if g.empty:
                continue
            v = g.val_auroc.max()
            if best is None or v > best[0]:
                best = (v, f"{dir_}/{name}.pt")
    return best[1]


def ckpt_path(ds, method, seed):
    if method == "ABMIL":
        fn = lambda a, lr, s: f"alpha={a}_criterion=L1_lr={lr}_pooling=ABMIL_seed={s}"
        if ds == "Semi-Synthetic":
            d = f"{EH}/varying_n_embedding_level=True/{SEMI_SUFFIX}/{SEMI_SUBDIR[seed][0]}"
            return select(d, fn, seed)
        if ds == "Head CT":
            a, lr = ICH_ABMIL_PICKS[seed]
            return f"{EH}/RSNA_ICH_full_dataset_embedding_level=True/{fn(a, lr, seed)}.pt"
        d = f"{EH}/RSNA_PE_embedding_level=True" if ds == "Chest CT" else f"{EH}/RSNA_AT_embedding_level=True"
        return select(d, fn, seed)
    fn = lambda a, lr, s: f"alpha={a}_criterion=GuidedL1_lr={lr}_pooling=ABMIL_seed={s}"
    if ds == "Semi-Synthetic":
        d = f"{EH}/varying_n_beta=1.0_embedding_level=True/{SEMI_SUFFIX}/{SEMI_SUBDIR[seed][0]}"
        return select(d, fn, seed)
    if ds == "Head CT":
        fn2 = lambda a, lr, s: f"alpha={a}_criterion=GuidedL1_div=fkl_ng=full_lr={lr}_pooling=ABMIL_seed={s}"
        return select(FKL_DIR, fn2, seed, ["0.001", "0.0001", "1e-05"], ["0.1", "0.01", "0.001"])
    d = f"{EH}/RSNA_PE_beta=1.0_embedding_level=True" if ds == "Chest CT" else f"{EH}/RSNA_AT_beta=1.0_embedding_level=True"
    return select(d, fn, seed)


def load_test(ds, seed):
    if ds == "Semi-Synthetic":
        data = datasets.ShiftedMeanMILDataset(n=1000, r=12, s_low=20, s_high=60, delta=0.5, seed=SEMI_SUBDIR[seed][1])
        ly = []
        for i, S in enumerate(data.lengths):
            y = [0.0] * int(S)
            if data.y[i] == 1:
                u = int(data.u[i])
                for j in range(u, min(u + data.r, int(S))):
                    y[j] = 1.0
            ly.append(y)
        return data, ly
    dd = {"Head CT": "encoded_RSNA_ICH_full_dataset", "Chest CT": "encoded_RSNA_PE", "Abdomen CT": "encoded_RSNA_AT"}[ds]
    d = torch.load(f"{DATA}/{dd}/ViT_B_16/seed={seed}/test.pt", map_location="cpu", weights_only=False)
    return datasets.MILTensorDataset(d["X"], d["lengths"], d["y"]), d["lengths_y"]


def main():
    rows = []
    for ds in ["Semi-Synthetic", "Head CT", "Chest CT", "Abdomen CT"]:
        for method in ["ABMIL", "ABMIL + NG"]:
            H, Hn = [], []
            for seed in SEEDS:
                data, ly = load_test(ds, seed)
                m = models.PoolClf(768, 1, pooling="ABMIL")
                sd = torch.load(ckpt_path(ds, method, seed), map_location="cpu")
                m.load_state_dict(sd.get("state_dict", sd))
                m.eval()
                h_bag, hn_bag = [], []
                with torch.no_grad():
                    for i in range(len(data)):
                        h_i, S_i, y_i = data[i]
                        y_ij = np.asarray(ly[i], dtype=float)
                        if y_i != 1.0 or len(y_ij) != S_i or y_ij.sum() == 0 or y_ij.sum() == len(y_ij):
                            continue
                        _, a = m(h_i, (int(S_i),))
                        p = torch.mean(a, dim=1)
                        p = p / p.sum()
                        ent = float(-(p * torch.log(torch.clamp(p, min=1e-12))).sum())
                        h_bag.append(ent)
                        hn_bag.append(ent / np.log(int(S_i)))
                H.append(np.mean(h_bag))
                Hn.append(np.mean(hn_bag))
            rows.append({"dataset": ds, "method": method,
                         "entropy": f"{np.mean(H):.3f} +/- {np.std(H):.3f}",
                         "entropy/log(S)": f"{np.mean(Hn):.3f} +/- {np.std(Hn):.3f}"})
            print(f"{ds:15s} {method:11s} H={rows[-1]['entropy']}   H/log(S)={rows[-1]['entropy/log(S)']}", flush=True)
    pd.DataFrame(rows).to_csv(f"{HOME_EXP}/_entropy_abmil_vs_ng.csv", index=False)


if __name__ == "__main__":
    main()
