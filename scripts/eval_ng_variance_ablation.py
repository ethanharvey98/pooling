"""Localization evaluation for the Normal-Guidance mean/variance ablation (RSNA ICH).

For each ng_mode in {full, fit_mean, fit_std, centered}:
  - per seed, pick the lr whose checkpoint has the highest val_auroc among epochs
    with train_auroc > val_auroc (same selection rule as notebooks/rsna_ich.ipynb);
  - load the saved ABMIL checkpoint and, on positive test bags, score the
    mean-over-heads attention against per-slice lesion labels;
  - report slice-localization AUROC / AUPRC and attention mass on lesion slices
    (attn_corr), mean +/- std over seeds.

Run inside the neuroimg_gpu env from the repo root:
    python scripts/eval_ng_variance_ablation.py
"""
import ast
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import datasets  # noqa: E402
import models  # noqa: E402

EXP_DIR = "/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_NG_variance_ablation_embedding_level=True"
DATA_DIR = "/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16"
SEEDS = [1001, 2001, 3001]
LRS = ["0.1", "0.01", "0.001"]
MODES = ["full", "fit_mean", "fit_std", "centered"]
ALPHA = "0.0001"


def select_lr(mode, seed):
    """Best lr for this (mode, seed) by val_auroc s.t. train_auroc > val_auroc."""
    best_lr, best_val = None, -np.inf
    for lr in LRS:
        name = f"alpha={ALPHA}_criterion=GuidedL1_ng={mode}_lr={lr}_pooling=ABMIL_seed={seed}"
        csv = f"{EXP_DIR}/{name}.csv"
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        gen = df[df.train_auroc > df.val_auroc]
        if gen.empty:
            continue
        v = gen.val_auroc.max()
        if v > best_val:
            best_lr, best_val = lr, v
    return best_lr, best_val


def localization_for_model(name, test_dataset, test_lengths_y):
    model = models.PoolClf(in_features=768, out_features=1, pooling="ABMIL")
    ckpt = torch.load(f"{EXP_DIR}/{name}.pt", map_location="cpu")["state_dict"]
    model.load_state_dict(ckpt)
    model.eval()

    attn_corrs, aurocs, auprcs = [], [], []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            h_i, S_i, y_i = test_dataset[i]
            y_ij = np.asarray(test_lengths_y[i], dtype=float)
            if y_i != 1.0 or len(y_ij) != S_i or y_ij.sum() == 0 or y_ij.sum() == len(y_ij):
                continue
            _, a_i = model(h_i, (S_i,))
            a_i = torch.mean(a_i, dim=1).detach().cpu().numpy().flatten()
            attn_corrs.append(float(a_i[y_ij == 1.0].sum()))
            aurocs.append(roc_auc_score(y_ij, a_i))
            auprcs.append(average_precision_score(y_ij, a_i))
    return np.mean(attn_corrs), np.mean(aurocs), np.mean(auprcs)


def main():
    per_seed_data = {}
    for seed in SEEDS:
        d = torch.load(f"{DATA_DIR}/seed={seed}/test.pt", map_location="cpu", weights_only=False)
        ds = datasets.MILTensorDataset(d["X"], d["lengths"], d["y"])
        per_seed_data[seed] = (ds, d["lengths_y"])

    rows = []
    for mode in MODES:
        m = {"attn_corr": [], "auroc": [], "auprc": [], "picks": []}
        for seed in SEEDS:
            lr, val = select_lr(mode, seed)
            if lr is None:
                print(f"[skip] {mode} seed={seed}: no finished csv")
                continue
            name = f"alpha={ALPHA}_criterion=GuidedL1_ng={mode}_lr={lr}_pooling=ABMIL_seed={seed}"
            if not os.path.exists(f"{EXP_DIR}/{name}.pt"):
                print(f"[skip] {mode} seed={seed}: {name}.pt missing")
                continue
            ds, ly = per_seed_data[seed]
            ac, ar, ap = localization_for_model(name, ds, ly)
            m["attn_corr"].append(ac)
            m["auroc"].append(ar)
            m["auprc"].append(ap)
            m["picks"].append(f"seed{seed}:lr={lr}(val={val:.4f})")
        if m["auroc"]:
            rows.append(
                {
                    "ng_mode": mode,
                    "n_seeds": len(m["auroc"]),
                    "loc_auroc": f"{np.mean(m['auroc']):.3f} +/- {np.std(m['auroc']):.3f}",
                    "loc_auprc": f"{np.mean(m['auprc']):.3f} +/- {np.std(m['auprc']):.3f}",
                    "attn_corr": f"{np.mean(m['attn_corr']):.3f} +/- {np.std(m['attn_corr']):.3f}",
                    "picks": " ".join(m["picks"]),
                }
            )
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(f"{EXP_DIR}/_localization_summary.csv", index=False)
    print(f"\nwrote {EXP_DIR}/_localization_summary.csv")


if __name__ == "__main__":
    main()
