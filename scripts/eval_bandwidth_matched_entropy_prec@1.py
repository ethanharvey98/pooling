"""Entropy + entropy-matched smoothing control for "denoising vs genuine relocation".

For each dataset and seed, with the per-seed (alpha, lr) checkpoints picked in the notebooks
(rsna_ich / rsna_pe / rsna_at / synthetic_data):
  1. On VALIDATION positive bags, choose one Gaussian-smoothing sigma so the mean normalized
     attention entropy of smoothed ABMIL equals that of ABMIL + Normal Guidance (only bag labels
     are used, no slice labels). If ABMIL is already flatter than NG, smoothing cannot lower the
     entropy, so sigma = 0 (identity) is chosen.
  2. On TEST positive bags report normalized entropy, P@1 (pointing game) and P@rec10 for:
     ABMIL, ABMIL + NG, and entropy-matched smoothed ABMIL.

  H_norm = -sum_j a_j ln a_j / ln(S)   (0 = one-hot, 1 = uniform)
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import precision_recall_curve

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import datasets  # noqa: E402
import models  # noqa: E402

EH = "/cluster/tufts/hugheslab/eharve06/pooling/experiments"
HOME_EXP = "/cluster/home/zmou01/pooling/experiments"
DATA = "/cluster/tufts/hugheslab/datasets"
SEEDS = [1001, 2001, 3001]
SIGMA_GRID = [0.0] + list(np.geomspace(0.3, 300, 70))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEMI_SUFFIX = "delta=0.5_r=12_s_low=20_s_high=60"
SEMI_SUBDIR = {
    1001: ("data_seed_test=1003_data_seed_train=1001_data_seed_val=1002_n_test=1000_n_train=10000_n_val=2500", 1002, 1003),
    2001: ("data_seed_test=2003_data_seed_train=2001_data_seed_val=2002_n_test=1000_n_train=10000_n_val=2500", 2002, 2003),
    3001: ("data_seed_test=3003_data_seed_train=3001_data_seed_val=3002_n_test=1000_n_train=10000_n_val=2500", 3002, 3003),
}


def semi_dir(base, seed):
    return f"{EH}/{base}/{SEMI_SUFFIX}/{SEMI_SUBDIR[seed][0]}"


def from_picks(dir_fn, criterion, picks):
    def f(seed):
        a, lr = picks[seed]
        p = f"{dir_fn(seed)}/alpha={a}_criterion={criterion}_lr={lr}_pooling=ABMIL_seed={seed}.pt"
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p
    return f


def fixed(d):
    return lambda seed: d


CONFIGS = {
    "Semi-Synthetic": dict(
        kind="semi",
        abmil=from_picks(lambda s: semi_dir("varying_n_embedding_level=True", s), "L1",
                         {1001: ("0.01", "0.1"), 2001: ("0.01", "0.1"), 3001: ("0.001", "0.001")}),
        ng=from_picks(lambda s: semi_dir("varying_n_beta=1.0_embedding_level=True", s), "GuidedL1",
                      {1001: ("0.01", "0.1"), 2001: ("0.1", "0.0001"), 3001: ("0.001", "0.01")}),
    ),
    "Head CT": dict(
        kind="std", data_dir=f"{DATA}/encoded_RSNA_ICH_full_dataset/ViT_B_16",
        abmil=from_picks(fixed(f"{EH}/RSNA_ICH_full_dataset_embedding_level=True"), "L1",
                         {1001: ("0.0001", "0.001"), 2001: ("0.0001", "0.1"), 3001: ("1e-05", "0.01")}),
        ng=from_picks(fixed(f"{EH}/RSNA_ICH_full_dataset_beta=1.0_embedding_level=True"), "GuidedL1",
                      {1001: ("0.0001", "0.01"), 2001: ("0.0001", "0.01"), 3001: ("0.0001", "0.001")}),
    ),
    "Chest CT": dict(
        kind="std", data_dir=f"{DATA}/encoded_RSNA_PE/ViT_B_16",
        abmil=from_picks(fixed(f"{EH}/RSNA_PE_embedding_level=True"), "L1",
                         {1001: ("1e-05", "0.01"), 2001: ("0.001", "0.01"), 3001: ("0.001", "0.01")}),
        ng=from_picks(fixed(f"{EH}/RSNA_PE_beta=1.0_embedding_level=True"), "GuidedL1",
                      {1001: ("0.001", "0.01"), 2001: ("1e-05", "0.01"), 3001: ("0.001", "0.01")}),
    ),
    "Abdomen CT": dict(
        kind="std", data_dir=f"{DATA}/encoded_RSNA_AT/ViT_B_16",
        abmil=from_picks(fixed(f"{EH}/RSNA_AT_embedding_level=True"), "L1",
                         {1001: ("0.001", "0.1"), 2001: ("0.001", "0.1"), 3001: ("0.001", "0.01")}),
        ng=from_picks(fixed(f"{EH}/RSNA_AT_beta=1.0_embedding_level=True"), "GuidedL1",
                      {1001: ("0.001", "0.01"), 2001: ("0.0001", "0.1"), 3001: ("0.0001", "0.1")}),
    ),
}

LABELS = ["ABMIL", "ABMIL + NG", "ABMIL smoothed (entropy-matched)"]


def load_split(cfg, split, seed):
    if cfg["kind"] == "semi":
        _, val_seed, test_seed = SEMI_SUBDIR[seed]
        n, s = (2500, val_seed) if split == "val" else (1000, test_seed)
        ds = datasets.ShiftedMeanMILDataset(n=n, r=12, s_low=20, s_high=60, delta=0.5, seed=s)
        ly = []
        for i, S in enumerate(ds.lengths):
            y_ij = [0.0] * int(S)
            if ds.y[i] == 1:
                u = int(ds.u[i])
                for j in range(u, min(u + ds.r, int(S))):
                    y_ij[j] = 1.0
            ly.append(y_ij)
        return ds, ly
    d = torch.load(f"{cfg['data_dir']}/seed={seed}/{split}.pt", map_location="cpu", weights_only=False)
    return datasets.MILTensorDataset(d["X"], d["lengths"], d["y"]), d["lengths_y"]


def load_model(path):
    m = models.PoolClf(768, 1, pooling="ABMIL")
    sd = torch.load(path, map_location="cpu")
    m.load_state_dict(sd.get("state_dict", sd))
    return m.to(device).eval()


def collect(model, ds, ly):
    """Normalized attention vector + slice labels for every valid positive bag."""
    out = []
    with torch.no_grad():
        for i in range(len(ds)):
            h_i, S_i, y_i = ds[i]
            y_ij = np.asarray(ly[i], dtype=float)
            if y_i != 1.0 or len(y_ij) != S_i or y_ij.sum() == 0 or y_ij.sum() == len(y_ij) or S_i < 3:
                continue
            _, a = model(h_i.to(device), (int(S_i),))
            a = torch.mean(a, dim=1).double().cpu().numpy().flatten()
            out.append((a / a.sum(), y_ij))
    return out


def h_norm(a):
    a = np.clip(a, 1e-300, None)
    a = a / a.sum()
    return float(-(a * np.log(a)).sum() / math.log(len(a)))


def smooth(a, sigma):
    if sigma <= 0:
        return a
    b = np.clip(gaussian_filter1d(a, sigma=sigma, mode="reflect"), 1e-300, None)
    return b / b.sum()


def match_sigma(val_abmil, target):
    """Smoothing sigma whose mean normalized entropy on the validation bags is closest to target."""
    best = (1e9, 0.0)
    for s in SIGMA_GRID:
        m = np.mean([h_norm(smooth(a, s)) for a, _ in val_abmil])
        if abs(m - target) < best[0]:
            best = (abs(m - target), s)
    return best[1]


def p_at_rec(y, s, r):
    p, rec, _ = precision_recall_curve(y, s)
    ok = rec >= r
    return float(np.max(p[ok])) if ok.any() else float("nan")


def evaluate(records, transform=None):
    H, P1, PR = [], [], []
    for a, y in records:
        b = transform(a) if transform else a
        H.append(h_norm(b))
        P1.append(float(y[int(np.argmax(b))]))
        PR.append(p_at_rec(y, b, 0.10))
    return dict(H_norm=np.mean(H), P1=np.mean(P1), Prec10=np.mean(PR))


def main():
    print("device:", device, flush=True)
    rows = []
    for ds_name, cfg in CONFIGS.items():
        agg = {k: {"H_norm": [], "P1": [], "Prec10": []} for k in LABELS}
        sigmas = []
        for seed in SEEDS:
            val_ds, val_ly = load_split(cfg, "val", seed)
            test_ds, test_ly = load_split(cfg, "test", seed)
            ab_path, ng_path = cfg["abmil"](seed), cfg["ng"](seed)
            print(f"[{ds_name}] seed {seed}: ABMIL={os.path.basename(ab_path)}  NG={os.path.basename(ng_path)}", flush=True)
            m_ab, m_ng = load_model(ab_path), load_model(ng_path)

            val_ab = collect(m_ab, val_ds, val_ly)
            target = float(np.mean([h_norm(a) for a, _ in collect(m_ng, val_ds, val_ly)]))
            sigma = match_sigma(val_ab, target)
            sigmas.append(sigma)

            test_ab = collect(m_ab, test_ds, test_ly)
            res = {
                "ABMIL": evaluate(test_ab),
                "ABMIL + NG": evaluate(collect(m_ng, test_ds, test_ly)),
                "ABMIL smoothed (entropy-matched)": evaluate(test_ab, lambda a: smooth(a, sigma)),
            }
            print(f"    matched sigma={sigma:.2f} (target val H_norm={target:.3f})", flush=True)
            for k, r in res.items():
                print(f"    {k:34s} H_norm={r['H_norm']:.3f}  P@1={r['P1']:.3f}  P@rec10={r['Prec10']:.3f}", flush=True)
                for kk in agg[k]:
                    agg[k][kk].append(r[kk])

        print(f"=== {ds_name} ===", flush=True)
        for k, v in agg.items():
            row = {
                "dataset": ds_name, "method": k,
                "H_norm": f"{np.mean(v['H_norm']):.3f} +/- {np.std(v['H_norm']):.3f}",
                "P@1": f"{np.mean(v['P1']):.3f} +/- {np.std(v['P1']):.3f}",
                "P@rec10": f"{np.mean(v['Prec10']):.3f} +/- {np.std(v['Prec10']):.3f}",
                "sigma_per_seed": ",".join(f"{s:.2f}" for s in sigmas) if "smoothed" in k else "",
            }
            rows.append(row)
            print(f"  {k:34s} H_norm={row['H_norm']} | P@1={row['P@1']} | P@rec10={row['P@rec10']} {row['sigma_per_seed']}", flush=True)
    pd.DataFrame(rows).to_csv(f"{HOME_EXP}/_bandwidth_matched_notebook_picks.csv", index=False)


if __name__ == "__main__":
    main()
