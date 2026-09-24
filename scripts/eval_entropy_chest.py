"""Attention entropy (and P@1 for context) for ABMIL (no guidance) vs ABMIL + Normal
Guidance on Chest CT (RSNA PE), positive test bags only. Same checkpoint selection as the
precision-at-recall tables (best val_auroc with train_auroc > val_auroc, per seed).

  H      = -sum_j a_j log a_j          (nats)
  H_norm = H / log(S)                   (0 = one-hot, 1 = uniform; comparable across bag sizes)
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import datasets  # noqa: E402
import models  # noqa: E402

EH = "/cluster/tufts/hugheslab/eharve06/pooling/experiments"
DATA = "/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16"
SEEDS = [1001, 2001, 3001]
ALPHAS = ["0.0", "1e-06", "1e-05", "0.0001", "0.001", "0.01", "0.1", "1.0"]
LRS = ["0.0001", "0.001", "0.01", "0.1"]

METHODS = {
    "ABMIL (no guidance)": (f"{EH}/RSNA_PE_embedding_level=True", "L1"),
    "ABMIL + Normal Guidance": (f"{EH}/RSNA_PE_beta=1.0_embedding_level=True", "GuidedL1"),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def select(dir_, criterion, seed):
    best = None
    for alpha in ALPHAS:
        for lr in LRS:
            name = f"alpha={alpha}_criterion={criterion}_lr={lr}_pooling=ABMIL_seed={seed}"
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
    return best


def main():
    print("device:", device, flush=True)
    rows = []
    for method, (dir_, crit) in METHODS.items():
        agg = {"H": [], "H_norm": [], "P@1": []}
        for seed in SEEDS:
            d = torch.load(f"{DATA}/seed={seed}/test.pt", map_location="cpu", weights_only=False)
            ds = datasets.MILTensorDataset(d["X"], d["lengths"], d["y"])
            ly = d["lengths_y"]
            v, name = select(dir_, crit, seed)
            m = models.PoolClf(768, 1, pooling="ABMIL")
            sd = torch.load(f"{dir_}/{name}.pt", map_location="cpu")
            m.load_state_dict(sd.get("state_dict", sd))
            m.to(device).eval()
            Hs, Hn, p1 = [], [], []
            with torch.no_grad():
                for i in range(len(ds)):
                    h_i, S_i, y_i = ds[i]
                    y_ij = np.asarray(ly[i], dtype=float)
                    if y_i != 1.0 or len(y_ij) != S_i or y_ij.sum() == 0 or y_ij.sum() == len(y_ij):
                        continue
                    _, a = m(h_i.to(device), (int(S_i),))
                    a = torch.mean(a, dim=1).double()
                    a = a / a.sum()
                    H = float(-(a * torch.log(torch.clamp(a, min=1e-300))).sum())
                    Hs.append(H)
                    Hn.append(H / math.log(int(S_i)))
                    p1.append(float(y_ij[int(torch.argmax(a))]))
            agg["H"].append(np.mean(Hs))
            agg["H_norm"].append(np.mean(Hn))
            agg["P@1"].append(np.mean(p1))
            print(f"  {method} seed {seed}: {name} val={v:.4f}  H={np.mean(Hs):.3f}  H_norm={np.mean(Hn):.3f}  P@1={np.mean(p1):.3f}", flush=True)
        row = {"method": method}
        row.update({k: f"{np.mean(v):.3f} +/- {np.std(v):.3f}" for k, v in agg.items()})
        rows.append(row)
        print(f"{method}: " + " | ".join(f"{k}={row[k]}" for k in agg), flush=True)
    pd.DataFrame(rows).to_csv("/cluster/home/zmou01/pooling/experiments/_entropy_chest.csv", index=False)


if __name__ == "__main__":
    main()
