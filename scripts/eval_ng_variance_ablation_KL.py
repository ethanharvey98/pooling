"""Localization eval for the forward-KL / reverse-KL NG mean-variance ablation (RSNA ICH).

Fixed hparams (notebook headline NG-ABMIL selection): alpha=1e-4, beta(lambda)=1.0,
lr = 0.01 / 0.01 / 0.001 for seed 1001 / 2001 / 3001. So the checkpoint for each
(divergence, ng_mode, seed) is deterministic -- no lr selection needed.

Run from the repo root in the neuroimg_gpu env:
    python scripts/eval_ng_variance_ablation_KL.py
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

EXP_DIR = "/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_NG_variance_ablation_KL_embedding_level=True"
DATA_DIR = "/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16"
SEED_LR = {1001: "0.01", 2001: "0.01", 3001: "0.001"}
DIVS = [("forward kl", "fkl"), ("reverse kl", "rkl")]
MODES = ["full", "fit_mean", "fit_std", "centered"]


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
    for s in SEED_LR:
        d = torch.load(f"{DATA_DIR}/seed={s}/test.pt", map_location="cpu", weights_only=False)
        per_seed[s] = (datasets.MILTensorDataset(d["X"], d["lengths"], d["y"]), d["lengths_y"])

    rows = []
    for dname, dtag in DIVS:
        for mode in MODES:
            acs, ars, aps, best_epochs = [], [], [], []
            for s, lr in SEED_LR.items():
                name = f"alpha=0.0001_criterion=GuidedL1_div={dtag}_ng={mode}_lr={lr}_pooling=ABMIL_seed={s}"
                if not os.path.exists(f"{EXP_DIR}/{name}.pt"):
                    print(f"[skip] {name} (no .pt)")
                    continue
                csv = pd.read_csv(f"{EXP_DIR}/{name}.csv")
                g = csv[csv.train_auroc > csv.val_auroc]
                best_epochs.append(int(g.val_auroc.idxmax()) if not g.empty else -1)
                ac, ar, ap = localization(name, *per_seed[s])
                acs.append(ac)
                ars.append(ar)
                aps.append(ap)
            if ars:
                rows.append(
                    {
                        "divergence": dname,
                        "ng_mode": mode,
                        "n_seeds": len(ars),
                        "loc_auroc": f"{np.mean(ars):.3f} +/- {np.std(ars):.3f}",
                        "loc_auprc": f"{np.mean(aps):.3f} +/- {np.std(aps):.3f}",
                        "attn_corr": f"{np.mean(acs):.3f} +/- {np.std(acs):.3f}",
                        "best_val_epochs": best_epochs,
                    }
                )
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(f"{EXP_DIR}/_localization_summary.csv", index=False)
    print(f"\nwrote {EXP_DIR}/_localization_summary.csv")


if __name__ == "__main__":
    main()
