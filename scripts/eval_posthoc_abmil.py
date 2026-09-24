"""Precision@low-recall for the 3 post-hoc ABMIL variants (reusing the exact same
ABMIL-no-guidance checkpoints already selected for the "ABMIL" row):
  - Blend a_i with Centered Gaussian: a = (attn + CG_prior(sigma)) / 2, sigma selected
    on val positive bags (argmax mean AUROC over sigma in 1..12).
  - Each slice as its own bag: feed each slice singly (lengths=(1,)) through the
    trained model and use the resulting classifier output as the per-slice score.
  - Gaussian kernel smoothing: scipy.ndimage.gaussian_filter1d(attn, sigma), sigma
    selected the same way as the blend variant.
Matches the exact logic found (commented out) in notebooks/rsna_ich.ipynb cell 23 and
notebooks/rsna_pe.ipynb cell 28.
"""
import glob
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import datasets  # noqa: E402
import models  # noqa: E402
import utils  # noqa: E402

EH = "/cluster/tufts/hugheslab/eharve06/pooling/experiments"
HOME_EXP = "/cluster/home/zmou01/pooling/experiments"
DATA = "/cluster/tufts/hugheslab/datasets"
SEEDS = [1001, 2001, 3001]
RECALLS = [0.05, 0.10, 0.25, 0.50]
ALPHAS = ["0.0", "1e-06", "1e-05", "0.0001", "0.001", "0.01", "0.1", "1.0"]
LRS = ["0.0001", "0.001", "0.01", "0.1"]
SIGMAS = list(range(1, 13))

SEMI_SUFFIX = "delta=0.5_r=12_s_low=20_s_high=60"
SEMI_SUBDIR = {
    1001: ("data_seed_test=1003_data_seed_train=1001_data_seed_val=1002_n_test=1000_n_train=10000_n_val=2500", 1002, 1003),
    2001: ("data_seed_test=2003_data_seed_train=2001_data_seed_val=2002_n_test=1000_n_train=10000_n_val=2500", 2002, 2003),
    3001: ("data_seed_test=3003_data_seed_train=3001_data_seed_val=3002_n_test=1000_n_train=10000_n_val=2500", 3002, 3003),
}


def semi_dir(seed):
    subdir, _, _ = SEMI_SUBDIR[seed]
    return f"{EH}/varying_n_embedding_level=True/{SEMI_SUFFIX}/{subdir}"


DATASETS = {
    "Semi-Synthetic": {"kind": "semi"},
    "Head CT": {"kind": "std", "l1_dir": f"{EH}/RSNA_ICH_full_dataset_embedding_level=True",
                "data_dir": f"{DATA}/encoded_RSNA_ICH_full_dataset/ViT_B_16"},
    "Chest CT": {"kind": "std", "l1_dir": f"{EH}/RSNA_PE_embedding_level=True",
                 "data_dir": f"{DATA}/encoded_RSNA_PE/ViT_B_16"},
    "Abdomen CT": {"kind": "std", "l1_dir": f"{EH}/RSNA_AT_embedding_level=True",
                   "data_dir": f"{DATA}/encoded_RSNA_AT/ViT_B_16"},
}


def load_semi(n, seed):
    ds = datasets.ShiftedMeanMILDataset(n=n, r=12, s_low=20, s_high=60, delta=0.5, seed=seed)
    lengths_y = []
    for i, S in enumerate(ds.lengths):
        y_ij = [0.0] * int(S)
        if ds.y[i] == 1:
            u = int(ds.u[i])
            for j in range(u, min(u + ds.r, int(S))):
                y_ij[j] = 1.0
        lengths_y.append(y_ij)
    return ds, lengths_y


def load_std(data_dir, split, seed):
    d = torch.load(f"{data_dir}/seed={seed}/{split}.pt", map_location="cpu", weights_only=False)
    return datasets.MILTensorDataset(d["X"], d["lengths"], d["y"]), d["lengths_y"]


def select_abmil_ckpt(cfg, seed):
    if cfg["kind"] == "semi":
        d = semi_dir(seed)
    else:
        d = cfg["l1_dir"]
    best = None
    for alpha in ALPHAS:
        for lr in LRS:
            name = f"alpha={alpha}_criterion=L1_lr={lr}_pooling=ABMIL_seed={seed}"
            csv = f"{d}/{name}.csv"
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv)
            g = df[df.train_auroc > df.val_auroc]
            if g.empty:
                continue
            v = g.val_auroc.max()
            if best is None or v > best[0]:
                best = (v, f"{d}/{name}.pt")
    return best


def raw_attn(model, h, S):
    with torch.no_grad():
        _, a = model(h, (int(S),))
    return torch.mean(a, dim=1).cpu().numpy().flatten()


def each_slice_score(model, h, S):
    with torch.no_grad():
        lengths = tuple([1 for _ in range(int(S))])
        y_hat_ij, _ = model(h, lengths)
    return torch.mean(y_hat_ij, dim=1).cpu().numpy().flatten()


def cg_prior(S, sigma):
    x = torch.arange(1, S + 1, dtype=torch.float32)
    a = utils.normal_pdf(x, mu=S / 2.0, sigma=sigma)
    return (a / a.sum()).numpy()


def log_blend_cg(a1_np, S, sigma, eps=1e-12):
    """logsumexp-based blend of ABMIL attention with Centered Gaussian, in log space
    throughout (Ethan/Mike's fix): avoids the underflow the raw-space (a1+cg)/2 version
    can hit. Returns a log-score (safe to rank directly -- AUROC/AUPRC/precision-at-
    recall only depend on order, not the score's absolute scale).
        log_a_CG = log_normal_pdf(x, S/2, sigma)
        norm_log_a_CG = log_a_CG - logsumexp(log_a_CG)
        blended = logsumexp([norm_log_a_CG, log_a_ABMIL]) - log(2)
    """
    x = torch.arange(1, S + 1, dtype=torch.float32)
    log_cg = utils.log_normal_pdf(x, mu=S / 2.0, sigma=sigma)
    norm_log_cg = log_cg - torch.logsumexp(log_cg, dim=0)
    log_abmil = torch.log(torch.clamp(torch.as_tensor(a1_np, dtype=torch.float32), min=eps))
    blended = torch.logsumexp(torch.stack([norm_log_cg, log_abmil], dim=0), dim=0) - math.log(2)
    return blended.numpy()


def pick_sigma(model, val_ds, val_ly, mode):
    """mode: 'blend' or 'smooth'. Selects sigma maximizing mean AUROC on positive val bags."""
    records = []
    for i in range(len(val_ds)):
        h_i, S_i, y_i = val_ds[i]
        y_ij = np.asarray(val_ly[i], dtype=float)
        if y_i != 1.0 or len(y_ij) != S_i or y_ij.sum() == 0 or y_ij.sum() == len(y_ij):
            continue
        a1 = raw_attn(model, h_i, S_i)
        records.append((a1, int(S_i), y_ij))
    if not records:
        return 1
    best_sigma, best_auroc = 1, -1
    for sigma in SIGMAS:
        aurocs = []
        for a1, S, y_ij in records:
            if mode == "blend":
                a = log_blend_cg(a1, S, sigma)
            else:
                a = gaussian_filter1d(a1, sigma=sigma)
            aurocs.append(roc_auc_score(y_ij, a))
        m = np.mean(aurocs)
        if m > best_auroc:
            best_auroc, best_sigma = m, sigma
    return best_sigma


def _prec_at_recall(y, s, r):
    p, rec, _ = precision_recall_curve(y, s)
    ok = rec >= r
    return float(np.max(p[ok])) if ok.any() else float("nan")


def main():
    cols = [f"P@rec{int(r * 100)}" for r in RECALLS] + ["P@1", "AUPRC"]
    methods = ["blend_cg", "each_slice", "gauss_smooth"]
    out_rows = {m: [] for m in methods}

    for tag, cfg in DATASETS.items():
        agg = {m: {c: [] for c in cols} for m in methods}
        for seed in SEEDS:
            best = select_abmil_ckpt(cfg, seed)
            if best is None:
                print(f"[skip] {tag} seed {seed}: no ABMIL checkpoint")
                continue
            v, pt = best
            m = models.PoolClf(768, 1, pooling="ABMIL")
            sd = torch.load(pt, map_location="cpu")
            m.load_state_dict(sd.get("state_dict", sd))
            m.eval()

            if cfg["kind"] == "semi":
                _, val_seed, test_seed = SEMI_SUBDIR[seed]
                val_ds, val_ly = load_semi(2500, val_seed)
                test_ds, test_ly = load_semi(1000, test_seed)
            else:
                val_ds, val_ly = load_std(cfg["data_dir"], "val", seed)
                test_ds, test_ly = load_std(cfg["data_dir"], "test", seed)

            sigma_blend = pick_sigma(m, val_ds, val_ly, "blend")
            sigma_smooth = pick_sigma(m, val_ds, val_ly, "smooth")
            print(f"  {tag} seed {seed}: val={v:.4f}  blend_sigma={sigma_blend}  smooth_sigma={sigma_smooth}")

            per_bag = {mth: {c: [] for c in cols} for mth in methods}
            for i in range(len(test_ds)):
                h_i, S_i, y_i = test_ds[i]
                y_ij = np.asarray(test_ly[i], dtype=float)
                if y_i != 1.0 or len(y_ij) != S_i or y_ij.sum() == 0 or y_ij.sum() == len(y_ij):
                    continue
                a1 = raw_attn(m, h_i, S_i)
                scores = {
                    "blend_cg": log_blend_cg(a1, int(S_i), sigma_blend),
                    "each_slice": each_slice_score(m, h_i, S_i),
                    "gauss_smooth": gaussian_filter1d(a1, sigma=sigma_smooth),
                }
                for mth, a in scores.items():
                    for r in RECALLS:
                        per_bag[mth][f"P@rec{int(r * 100)}"].append(_prec_at_recall(y_ij, a, r))
                    per_bag[mth]["P@1"].append(float(y_ij[int(np.argmax(a))]))
                    per_bag[mth]["AUPRC"].append(average_precision_score(y_ij, a))
            for mth in methods:
                for c in cols:
                    agg[mth][c].append(np.mean(per_bag[mth][c]))

        for mth in methods:
            row = {"dataset": tag}
            row.update({c: f"{np.mean(agg[mth][c]):.3f} +/- {np.std(agg[mth][c]):.3f}" for c in cols})
            out_rows[mth].append(row)
            print(f"[{mth}] {tag}: " + " | ".join(f"{c}={row[c]}" for c in cols))

    for mth in methods:
        pd.DataFrame(out_rows[mth]).to_csv(f"{HOME_EXP}/_posthoc_{mth}_p_at_rec.csv", index=False)
        print(f"wrote {HOME_EXP}/_posthoc_{mth}_p_at_rec.csv")


if __name__ == "__main__":
    main()
