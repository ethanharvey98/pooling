"""Check r_hat underflow for the FROZEN-std NG modes (centered, fit_mean), where
sigma = empirical_std * S_i instead of being fit from the model's attention.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import datasets  # noqa: E402
import models  # noqa: E402
import utils  # noqa: E402

DATA = "/cluster/tufts/hugheslab/datasets"
EMP_STD = 0.115
EMP_MEAN = 0.52


def analyze(model, ds, tag, max_bags=200):
    n_bags = 0
    n_underflow_total = 0
    n_slices_total = 0
    with torch.no_grad():
        for i in range(min(len(ds), max_bags)):
            h_i, S_i, y_i = ds[i]
            if y_i != 1.0:
                continue
            S_i = int(S_i)
            j = torch.arange(1, S_i + 1, dtype=torch.float32)
            mean = EMP_MEAN * S_i
            std = EMP_STD * S_i
            r_hat = utils.normal_pdf(j, torch.tensor(mean), torch.tensor(std))
            underflow_mask = (r_hat == 0.0)
            n_underflow_total += int(underflow_mask.sum())
            n_slices_total += S_i
            n_bags += 1

    print(f"### {tag}")
    print(f"  bags analyzed: {n_bags}")
    print(f"  slices underflowed: {n_underflow_total}/{n_slices_total} ({100*n_underflow_total/max(n_slices_total,1):.1f}%)")
    print()


def pick_best_ckpt(exp_dir, pattern):
    best = None
    for f in glob.glob(f"{exp_dir}/{pattern}"):
        df = pd.read_csv(f)
        g = df[df.train_auroc > df.val_auroc]
        if g.empty:
            continue
        v = g.val_auroc.max()
        if best is None or v > best[0]:
            best = (v, f[:-4] + ".pt")
    return best


ich_dir = "/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_NG_variance_ablation_fkl_sweep_embedding_level=True"
d = torch.load(f"{DATA}/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=1001/test.pt", map_location="cpu", weights_only=False)
ds = datasets.MILTensorDataset(d["X"], d["lengths"], d["y"])

for mode in ["centered", "fit_mean"]:
    best = pick_best_ckpt(ich_dir, f"*ng={mode}*seed=1001.csv")
    if best is None:
        print(f"[skip] mode={mode}: no checkpoint")
        continue
    v, pt = best
    m = models.PoolClf(768, 1, pooling="ABMIL")
    sd = torch.load(pt, map_location="cpu")
    m.load_state_dict(sd.get("state_dict", sd))
    m.eval()
    analyze(m, ds, f"Head CT (NG {mode}, seed 1001, val={v:.4f})")

# Also just the pure formula check across a wide range of S (no model needed)
print("### Pure formula check across S (no model)")
for S in [12, 60, 100, 500, 1000, 1727]:
    j = torch.arange(1, S + 1, dtype=torch.float32)
    mean = EMP_MEAN * S
    std = EMP_STD * S
    r_hat = utils.normal_pdf(j, torch.tensor(mean), torch.tensor(std))
    n_zero = int((r_hat == 0.0).sum())
    print(f"  S={S}: underflowed {n_zero}/{S}")
