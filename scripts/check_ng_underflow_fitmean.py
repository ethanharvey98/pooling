"""Correct check for the fit_mean NG mode: mean is COMPUTED live from the model's
current attention (like fit_std=True does for std), while std stays frozen at
empirical_std * S_i.
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


def analyze(model, ds, tag, max_bags=200):
    n_bags = 0
    n_underflow_total = 0
    n_slices_total = 0
    fitted_means = []
    with torch.no_grad():
        for i in range(min(len(ds), max_bags)):
            h_i, S_i, y_i = ds[i]
            if y_i != 1.0:
                continue
            _, a = model(h_i, (int(S_i),))
            a = a.mean(dim=1)
            S_i = int(S_i)
            j = torch.arange(1, S_i + 1, dtype=torch.float32)
            sum_a = torch.clamp(a.sum(), min=1e-6)
            mean = (j * a).sum() / sum_a   # FIT from current attention
            std = EMP_STD * S_i             # FROZEN

            fitted_means.append((S_i, float(mean)))
            r_hat = utils.normal_pdf(j, mean, torch.tensor(std))
            underflow_mask = (r_hat == 0.0)
            n_underflow_total += int(underflow_mask.sum())
            n_slices_total += S_i
            n_bags += 1

    print(f"### {tag}")
    print(f"  bags analyzed: {n_bags}")
    print(f"  slices underflowed: {n_underflow_total}/{n_slices_total} ({100*n_underflow_total/max(n_slices_total,1):.1f}%)")
    Ss, means = zip(*fitted_means)
    print(f"  fitted mean/S ratio: mean={np.mean([m/S for S,m in fitted_means]):.3f}")
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

best = pick_best_ckpt(ich_dir, "*ng=fit_mean*seed=1001.csv")
if best is None:
    print("[skip] no fit_mean checkpoint")
else:
    v, pt = best
    m = models.PoolClf(768, 1, pooling="ABMIL")
    sd = torch.load(pt, map_location="cpu")
    m.load_state_dict(sd.get("state_dict", sd))
    m.eval()
    analyze(m, ds, f"Head CT (NG fit_mean, seed 1001, val={v:.4f})")
