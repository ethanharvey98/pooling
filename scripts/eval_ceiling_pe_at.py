"""Precision@low-recall for Best-in-Class Ceiling (InstanceLevelClassifier), Chest CT
(RSNA PE) and Abdomen CT (RSNA AT). No Semi-Synthetic checkpoints exist for this method.
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

DATASET_CFG = {
    "Chest CT (RSNA PE)": {"ceil_dir": f"{EH}/RSNA_PE_best_possible_instance-level",
                             "data_dir": f"{DATA}/encoded_RSNA_PE/ViT_B_16", "kernel": 35},
    "Abdomen CT (RSNA AT)": {"ceil_dir": f"{EH}/RSNA_AT_best_possible_instance-level",
                               "data_dir": f"{DATA}/encoded_RSNA_AT/ViT_B_16", "kernel": 29},
}


def _select(dir_, seed):
    best = None
    for alpha in ALPHAS:
        for lr in LRS:
            name = f"alpha={alpha}_criterion=L1_lr={lr}_seed={seed}"
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
        raise FileNotFoundError(f"no valid ceiling checkpoint in {dir_} for seed {seed}")
    return best


def _prec_at_recall(y, s, r):
    p, rec, _ = precision_recall_curve(y, s)
    ok = rec >= r
    return float(np.max(p[ok])) if ok.any() else float("nan")


def main():
    cols = [f"P@rec{int(r * 100)}" for r in RECALLS] + ["P@1", "AUPRC"]
    out_rows = []
    for tag, cfg in DATASET_CFG.items():
        print(f"\n########## {tag} ##########", flush=True)
        agg = {c: [] for c in cols}
        for seed in SEEDS:
            d = torch.load(f"{cfg['data_dir']}/seed={seed}/test.pt", map_location="cpu", weights_only=False)
            ds = datasets.MILTensorDataset(d["X"], d["lengths"], d["y"])
            ly = d["lengths_y"]

            v, name = _select(cfg["ceil_dir"], seed)
            ckpt = f"{cfg['ceil_dir']}/{name}.pt"
            m = models.InstanceLevelClassifier(768, 1, kernel_size=cfg["kernel"])
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
                    out, _ = m(h_i, (int(S_i),))
                    a = torch.mean(out, dim=1).cpu().numpy().flatten()
                    for r in RECALLS:
                        per_bag[f"P@rec{int(r * 100)}"].append(_prec_at_recall(y_ij, a, r))
                    per_bag["P@1"].append(float(y_ij[int(np.argmax(a))]))
                    per_bag["AUPRC"].append(average_precision_score(y_ij, a))
            for c in cols:
                agg[c].append(np.mean(per_bag[c]))
            print(f"  seed {seed}: {name} val={v:.4f}  P@rec10={np.mean(per_bag['P@rec10']):.3f}", flush=True)

        row = {"dataset": tag}
        row.update({c: f"{np.mean(agg[c]):.3f} +/- {np.std(agg[c]):.3f}" for c in cols})
        out_rows.append(row)
        print(f"{tag}: " + " | ".join(f"{c}={row[c]}" for c in cols), flush=True)

    out = pd.DataFrame(out_rows)
    out_csv = "/cluster/home/zmou01/pooling/experiments/ceiling_p_at_rec10_pe_at.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
