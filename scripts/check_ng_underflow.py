"""Check whether NG's fit_std=True mode actually causes r_hat underflow on trained
checkpoints, and whether underflowed positions correspond to attention weights that
are already near-zero (Mike's hypothesis) or not.
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


def analyze(model, ds, tag, max_bags=200):
    n_bags = 0
    n_underflow_total = 0
    n_slices_total = 0
    underflow_attn = []  # attn weight at underflowed positions
    nonunderflow_attn = []
    fitted_sigmas = []
    with torch.no_grad():
        for i in range(min(len(ds), max_bags)):
            h_i, S_i, y_i = ds[i]
            if y_i != 1.0:
                continue
            _, a = model(h_i, (int(S_i),))
            a = a.mean(dim=1)  # [S]
            S_i = int(S_i)
            j = torch.arange(1, S_i + 1, dtype=torch.float32)
            sum_a = torch.clamp(a.sum(), min=1e-6)
            mean = (j * a).sum() / sum_a
            var = (j.pow(2) * a).sum() / sum_a - mean.pow(2)
            std = torch.sqrt(torch.clamp(var, min=1e-12))
            fitted_sigmas.append((S_i, float(std), float(mean)))

            r_hat = utils.normal_pdf(j, mean, std)
            underflow_mask = (r_hat == 0.0)
            n_underflow_total += int(underflow_mask.sum())
            n_slices_total += S_i
            n_bags += 1

            underflow_attn.extend(a[underflow_mask].tolist())
            nonunderflow_attn.extend(a[~underflow_mask].tolist())

    print(f"### {tag}")
    print(f"  bags analyzed: {n_bags}")
    print(f"  slices underflowed: {n_underflow_total}/{n_slices_total} ({100*n_underflow_total/max(n_slices_total,1):.1f}%)")
    if underflow_attn:
        print(f"  attn weight at underflowed positions: mean={np.mean(underflow_attn):.3e} max={np.max(underflow_attn):.3e}")
    if nonunderflow_attn:
        print(f"  attn weight at non-underflowed positions: mean={np.mean(nonunderflow_attn):.3e} min={np.min(nonunderflow_attn):.3e}")
    Ss, stds, means = zip(*fitted_sigmas)
    print(f"  fitted sigma: mean={np.mean(stds):.2f}  median={np.median(stds):.2f}  (bag S: mean={np.mean(Ss):.1f} max={np.max(Ss)})")
    print(f"  sigma/S ratio: mean={np.mean([s/S for S,s,_ in fitted_sigmas]):.4f}")
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


# ICH: fkl sweep, ng_mode=full (fit_mean=True, fit_std=True)
ich_dir = "/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_NG_variance_ablation_fkl_sweep_embedding_level=True"
best = pick_best_ckpt(ich_dir, "*ng=full*seed=1001.csv")
if best:
    v, pt = best
    m = models.PoolClf(768, 1, pooling="ABMIL")
    sd = torch.load(pt, map_location="cpu")
    m.load_state_dict(sd.get("state_dict", sd))
    m.eval()
    d = torch.load(f"{DATA}/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=1001/test.pt", map_location="cpu", weights_only=False)
    ds = datasets.MILTensorDataset(d["X"], d["lengths"], d["y"])
    analyze(m, ds, f"Head CT (NG full, seed 1001, val={v:.4f})")

# PE / AT: beta=1.0 dir (default fit_mean=True, fit_std=True)
for tag, dir_ in [
    ("Chest CT (PE)", "/cluster/tufts/hugheslab/eharve06/pooling/experiments/RSNA_PE_beta=1.0_embedding_level=True"),
    ("Abdomen CT (AT)", "/cluster/tufts/hugheslab/eharve06/pooling/experiments/RSNA_AT_beta=1.0_embedding_level=True"),
]:
    best = pick_best_ckpt(dir_, "*pooling=ABMIL_seed=1001.csv")
    if best is None:
        print(f"[skip] {tag}: no checkpoint")
        continue
    v, pt = best
    m = models.PoolClf(768, 1, pooling="ABMIL")
    sd = torch.load(pt, map_location="cpu")
    m.load_state_dict(sd.get("state_dict", sd))
    m.eval()
    dset_name = "encoded_RSNA_PE" if "PE" in tag else "encoded_RSNA_AT"
    d = torch.load(f"{DATA}/{dset_name}/ViT_B_16/seed=1001/test.pt", map_location="cpu", weights_only=False)
    ds = datasets.MILTensorDataset(d["X"], d["lengths"], d["y"])
    analyze(m, ds, f"{tag} (NG, seed 1001, val={v:.4f})")
