"""Entropy + bandwidth-matched smoothing control for "denoising vs genuine relocation".

For each dataset and seed:
  1. ABMIL (no guidance) and ABMIL + Normal Guidance checkpoints (same picks as the
     precision-at-recall tables).
  2. On VALIDATION positive bags, choose one Gaussian-smoothing sigma so the mean
     normalized attention entropy of smoothed ABMIL equals NG's mean normalized entropy
     (only bag labels are used, no slice labels). If ABMIL is already flatter than NG,
     smoothing cannot lower entropy, so sigma = 0 (identity) is chosen.
  3. On TEST positive bags report normalized entropy, P@1 (pointing game) and P@rec10
     for: ABMIL, ABMIL + NG, entropy-matched smoothed ABMIL.

  H_norm = -sum_j a_j log a_j / log(S)   (0 = one-hot, 1 = uniform)
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
ALPHAS = ["0.0", "1e-06", "1e-05", "0.0001", "0.001", "0.01", "0.1", "1.0"]
LRS = ["0.0001", "0.001", "0.01", "0.1"]
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


def best_by_val(dir_, template, seed, alphas=ALPHAS, lrs=LRS):
    best = None
    for a in alphas:
        for lr in lrs:
            name = template.format(a=a, lr=lr, s=seed)
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
    if best is None:
        raise FileNotFoundError(f"{dir_} {template} seed {seed}")
    return best[1]


HEAD_ABMIL_PICKS = {1001: ("0.0001", "0.001"), 2001: ("0.0001", "0.1"), 3001: ("1e-05", "0.01")}


def head_abmil(seed):
    a, lr = HEAD_ABMIL_PICKS[seed]
    name = f"alpha={a}_criterion=L1_lr={lr}_pooling=ABMIL_seed={seed}"
    for d in [f"{EH}/RSNA_ICH_full_dataset_embedding_level=True",
              f"{HOME_EXP}/RSNA_ICH_full_dataset_baselines_embedding_level=True"]:
        if os.path.exists(f"{d}/{name}.pt"):
            return f"{d}/{name}.pt"
    raise FileNotFoundError(name)


L1_T = "alpha={a}_criterion=L1_lr={lr}_pooling=ABMIL_seed={s}"
NG_T = "alpha={a}_criterion=GuidedL1_lr={lr}_pooling=ABMIL_seed={s}"
FKL_T = "alpha={a}_criterion=GuidedL1_div=fkl_ng=full_lr={lr}_pooling=ABMIL_seed={s}"

CONFIGS = {
    "Semi-Synthetic": dict(
        kind="semi",
        abmil=lambda s: best_by_val(semi_dir("varying_n_embedding_level=True", s), L1_T, s),
        ng=lambda s: best_by_val(semi_dir("varying_n_beta=1.0_embedding_level=True", s), NG_T, s),
    ),
    "Head CT": dict(
        kind="std", data_dir=f"{DATA}/encoded_RSNA_ICH_full_dataset/ViT_B_16",
        abmil=head_abmil,
        ng=lambda s: best_by_val(f"{HOME_EXP}/RSNA_ICH_full_dataset_NG_variance_ablation_fkl_sweep_embedding_level=True",
                                 FKL_T, s, ["0.001", "0.0001", "1e-05"], ["0.1", "0.01", "0.001"]),
    ),
    "Chest CT": dict(
        kind="std", data_dir=f"{DATA}/encoded_RSNA_PE/ViT_B_16",
        abmil=lambda s: best_by_val(f"{EH}/RSNA_PE_embedding_level=True", L1_T, s),
        ng=lambda s: best_by_val(f"{EH}/RSNA_PE_beta=1.0_embedding_level=True", NG_T, s),
    ),
    "Abdomen CT": dict(
        kind="std", data_dir=f"{DATA}/encoded_RSNA_AT/ViT_B_16",
        abmil=lambda s: best_by_val(f"{EH}/RSNA_AT_embedding_level=True", L1_T, s),
        ng=lambda s: best_by_val(f"{EH}/RSNA_AT_beta=1.0_embedding_level=True", NG_T, s),
    ),
}


def load_split(cfg, split, seed):
    if cfg["kind"] == "semi":
        _, vs, ts = SEMI_SUBDIR[seed]
        n, s = (2500, vs) if split == "val" else (1000, ts)
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
        agg = {k: {"H_norm": [], "P1": [], "Prec10": []} for k in ["ABMIL", "ABMIL + NG", "ABMIL smoothed (entropy-matched)"]}
        sigmas = []
        for seed in SEEDS:
            val_ds, val_ly = load_split(cfg, "val", seed)
            test_ds, test_ly = load_split(cfg, "test", seed)
            m_ab, m_ng = load_model(cfg["abmil"](seed)), load_model(cfg["ng"](seed))
            val_ab, val_ng = collect(m_ab, val_ds, val_ly), collect(m_ng, val_ds, val_ly)
            target = float(np.mean([h_norm(a) for a, _ in val_ng]))
            val_ab_h = float(np.mean([h_norm(a) for a, _ in val_ab]))
            sigma = match_sigma(val_ab, target)
            sigmas.append(sigma)
            test_ab, test_ng = collect(m_ab, test_ds, test_ly), collect(m_ng, test_ds, test_ly)
            res = {
                "ABMIL": evaluate(test_ab),
                "ABMIL + NG": evaluate(test_ng),
                "ABMIL smoothed (entropy-matched)": evaluate(test_ab, lambda a: smooth(a, sigma)),
            }
            print(f"[{ds_name}] seed {seed}: val H_norm ABMIL={val_ab_h:.3f} NG={target:.3f} -> sigma={sigma:.2f}", flush=True)
            for k, r in res.items():
                print(f"    {k:34s} H_norm={r['H_norm']:.3f}  P@1={r['P1']:.3f}  P@rec10={r['Prec10']:.3f}", flush=True)
                for kk in agg[k]:
                    agg[k][kk].append(r[kk])
        for k, v in agg.items():
            rows.append({
                "dataset": ds_name, "method": k,
                "H_norm": f"{np.mean(v['H_norm']):.3f} +/- {np.std(v['H_norm']):.3f}",
                "P@1": f"{np.mean(v['P1']):.3f} +/- {np.std(v['P1']):.3f}",
                "P@rec10": f"{np.mean(v['Prec10']):.3f} +/- {np.std(v['Prec10']):.3f}",
                "sigma_per_seed": ",".join(f"{s:.2f}" for s in sigmas) if "smoothed" in k else "",
            })
        print(f"=== {ds_name} ===", flush=True)
        for r in rows[-3:]:
            print(f"  {r['method']:34s} H_norm={r['H_norm']} | P@1={r['P@1']} | P@rec10={r['P@rec10']} {r['sigma_per_seed']}", flush=True)
    pd.DataFrame(rows).to_csv(f"{HOME_EXP}/_bandwidth_matched_smoothing.csv", index=False)


if __name__ == "__main__":
    main()
