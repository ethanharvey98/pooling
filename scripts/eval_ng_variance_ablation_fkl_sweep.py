"""Localization eval for the forward-KL NG mean/variance ablation, alpha AND lr swept.

Per (ng_mode, seed): pick the (alpha, lr) in {1e-3,1e-4,1e-5} x {0.1,0.01,0.001} whose
checkpoint has the highest val_auroc among epochs with train_auroc > val_auroc (the
rule from notebooks/rsna_ich.ipynb), then score slice-localization on positive test bags
and record the bag-level test_auroc (classification) at that same selected epoch.

Run from the repo root in the neuroimg_gpu env:
    python scripts/eval_ng_variance_ablation_fkl_sweep.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import datasets  # noqa: E402
import models  # noqa: E402

EXP_DIR = "/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_NG_variance_ablation_fkl_sweep_embedding_level=True"
DATA_DIR = "/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16"
SEEDS = [1001, 2001, 3001]
ALPHAS = ["0.001", "0.0001", "1e-05"]
LRS = ["0.1", "0.01", "0.001"]
MODES = ["full", "fit_mean", "fit_std", "centered"]
DIV_TAG = "fkl"
MU_SIGMA = {"full": "fit / fit", "fit_std": "frozen / fit", "centered": "frozen / frozen", "fit_mean": "fit / frozen"}


def select(mode, seed):
    """Best (alpha, lr) for this (mode, seed) by val_auroc s.t. train_auroc > val_auroc."""
    best = None  # (val, alpha, lr, name, epoch, test_auroc)
    for alpha in ALPHAS:
        for lr in LRS:
            name = f"alpha={alpha}_criterion=GuidedL1_div={DIV_TAG}_ng={mode}_lr={lr}_pooling=ABMIL_seed={seed}"
            csv = f"{EXP_DIR}/{name}.csv"
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv)
            g = df[df.train_auroc > df.val_auroc]
            if g.empty:
                continue
            i = int(g.val_auroc.idxmax())
            v = float(df.loc[i, "val_auroc"])
            if best is None or v > best[0]:
                best = (v, alpha, lr, name, i, float(df.loc[i, "test_auroc"]))
    return best


def localization(name, test_dataset, test_lengths_y):
    model = models.PoolClf(in_features=768, out_features=1, pooling="ABMIL")
    model.load_state_dict(torch.load(f"{EXP_DIR}/{name}.pt", map_location="cpu")["state_dict"])
    model.eval()
    ac, ar, ap = [], [], []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            h_i, S_i, y_i = test_dataset[i]
            y_ij = np.asarray(test_lengths_y[i], dtype=float)
            if y_i != 1.0 or len(y_ij) != S_i or y_ij.sum() == 0 or y_ij.sum() == len(y_ij):
                continue
            _, a_i = model(h_i, (S_i,))
            a_i = torch.mean(a_i, dim=1).detach().cpu().numpy().flatten()
            ac.append(float(a_i[y_ij == 1.0].sum()))
            ar.append(roc_auc_score(y_ij, a_i))
            ap.append(average_precision_score(y_ij, a_i))
    return np.mean(ac), np.mean(ar), np.mean(ap)


def main():
    per_seed = {}
    for s in SEEDS:
        d = torch.load(f"{DATA_DIR}/seed={s}/test.pt", map_location="cpu", weights_only=False)
        per_seed[s] = (datasets.MILTensorDataset(d["X"], d["lengths"], d["y"]), d["lengths_y"])

    sel_rows = []
    sum_rows = []
    for mode in MODES:
        acs, ars, aps, tests, picks = [], [], [], [], []
        per_seed_pick = {}
        for s in SEEDS:
            best = select(mode, s)
            if best is None:
                print(f"[skip] {mode} seed={s}: nothing finished")
                per_seed_pick[s] = "n/a"
                continue
            val, alpha, lr, name, epoch, test_auroc = best
            per_seed_pick[s] = f"{alpha}, {lr}, ep {epoch}"
            if not os.path.exists(f"{EXP_DIR}/{name}.pt"):
                print(f"[skip] {name}.pt missing")
                continue
            ac, ar, ap = localization(name, *per_seed[s])
            acs.append(ac)
            ars.append(ar)
            aps.append(ap)
            tests.append(test_auroc)
            picks.append(f"seed{s}:a={alpha},lr={lr}@ep{epoch}(val={val:.4f})")
        sel_rows.append({"ng_mode": mode, "mu/sigma": MU_SIGMA[mode], **{f"seed {s}": per_seed_pick[s] for s in SEEDS}})
        if ars:
            sum_rows.append(
                {
                    "ng_mode": mode,
                    "mu/sigma": MU_SIGMA[mode],
                    "n_seeds": len(ars),
                    "classification_auroc": f"{np.mean(tests):.3f} +/- {np.std(tests):.3f}",
                    "loc_auroc": f"{np.mean(ars):.3f} +/- {np.std(ars):.3f}",
                    "loc_auprc": f"{np.mean(aps):.3f} +/- {np.std(aps):.3f}",
                    "attn_corr": f"{np.mean(acs):.3f} +/- {np.std(acs):.3f}",
                    "picks": " ".join(picks),
                }
            )

    sel_df = pd.DataFrame(sel_rows)
    sum_df = pd.DataFrame(sum_rows)
    print("=== selection table ===")
    print(sel_df.to_string(index=False))
    print("\n=== summary table ===")
    print(sum_df.to_string(index=False))
    sel_df.to_csv(f"{EXP_DIR}/_selection_1000ep.csv", index=False)
    sum_df.to_csv(f"{EXP_DIR}/_localization_summary.csv", index=False)
    print(f"\nwrote {EXP_DIR}/_selection_1000ep.csv and {EXP_DIR}/_localization_summary.csv")


if __name__ == "__main__":
    main()
