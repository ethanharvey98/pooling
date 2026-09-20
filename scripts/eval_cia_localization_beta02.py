"""Bag classification + localization (per-positive-bag slice AUROC/AUPRC) for the CIA+L1
alpha x beta_effect x lr sweep, across all 4 datasets, compared against Mean/Max pooling
(model-free) baselines computed the same way as the RSNA ICH precision-at-low-recall table.

For each dataset: for each seed, select the (alpha, beta_effect, lr) checkpoint with the
highest val_auroc (bag-level, same rule used everywhere else in this repo), then report the
bag-level test_auroc (classification) at that same epoch and score its factual attention on
positive test bags (localization).

Run from the repo root in the neuroimg_gpu env:
    python scripts/eval_cia_localization.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import datasets  # noqa: E402
import models  # noqa: E402

E = "/cluster/home/zmou01/pooling/experiments"
DATA = "/cluster/tufts/hugheslab/datasets"
SEEDS = [1001, 2001, 3001]

DATASETS = {
    "semi-synthetic": {
        "dir": f"{E}/synthetic_CIA_L1_embedding_level=True",
        "test_pt": None,  # generated on the fly, see below
    },
    "head CT": {
        "dir": f"{E}/RSNA_ICH_full_dataset_CIA_L1_embedding_level=True",
        "test_pt": lambda s: f"{DATA}/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed={s}/test.pt",
    },
    "chest CT (RSNA PE)": {
        "dir": f"{E}/RSNA_PE_CIA_L1_embedding_level=True",
        "test_pt": lambda s: f"{DATA}/encoded_RSNA_PE/ViT_B_16/seed={s}/test.pt",
    },
    "abdomen CT (RSNA AT)": {
        "dir": f"{E}/RSNA_AT_CIA_L1_embedding_level=True",
        "test_pt": lambda s: f"{DATA}/encoded_RSNA_AT/ViT_B_16/seed={s}/test.pt",
    },
}


def load_test_dataset(tag, seed):
    if tag == "semi-synthetic":
        ds = datasets.ShiftedMeanMILDataset(n=1000, delta=0.5, r=12, seed=seed + 2)
        # ShiftedMeanMILDataset doesn't carry lengths_y; reconstruct per-instance labels
        # from its own construction (positions u_i : u_i + r are the injected instances).
        lengths_y = []
        for i, S in enumerate(ds.lengths):
            y_ij = [0.0] * S
            if ds.y[i] == 1:
                u = int(ds.u[i])
                for j in range(u, min(u + ds.r, S)):
                    y_ij[j] = 1.0
            lengths_y.append(y_ij)
        return ds, lengths_y
    d = torch.load(DATASETS[tag]["test_pt"](seed), map_location="cpu", weights_only=False)
    return datasets.MILTensorDataset(d["X"], d["lengths"], d["y"]), d["lengths_y"]


def select_ckpt(exp_dir, seed):
    """Best (alpha, beta_effect, lr) for this seed by val_auroc s.t. train_auroc > val_auroc."""
    best = None  # (val, test_auroc, test_auprc, epoch, pt_path, name)
    for f in glob.glob(f"{exp_dir}/*beta_effect=0.2_*seed={seed}.csv"):
        df = pd.read_csv(f)
        g = df[df.train_auroc > df.val_auroc]
        if g.empty:
            continue
        i = int(g.val_auroc.idxmax())
        v = float(df.loc[i, "val_auroc"])
        if best is None or v > best[0]:
            best = (v, float(df.loc[i, "test_auroc"]), float(df.loc[i, "test_auprc"]), i, f[:-4] + ".pt", os.path.basename(f)[:-4])
    return best


def localization(pt_path, ds, lengths_y):
    m = models.PoolClf(768, 1, pooling="ABMIL")
    sd = torch.load(pt_path, map_location="cpu")
    m.load_state_dict(sd.get("state_dict", sd))
    m.eval()
    ar, ap = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            h, S, y = ds[i]
            y_ij = np.asarray(lengths_y[i], dtype=float)
            if y != 1.0 or len(y_ij) != S or y_ij.sum() == 0 or y_ij.sum() == len(y_ij):
                continue
            _, a = m(h, (S,))
            a = torch.mean(a, dim=1).numpy().flatten()
            ar.append(roc_auc_score(y_ij, a))
            ap.append(average_precision_score(y_ij, a))
    return float(np.mean(ar)), float(np.mean(ap))


def mean_max_baselines(ds, lengths_y):
    ar_mean, ap_mean, ar_max, ap_max = [], [], [], []
    for i in range(len(ds)):
        h, S, y = ds[i]
        y_ij = np.asarray(lengths_y[i], dtype=float)
        if y != 1.0 or len(y_ij) != S or y_ij.sum() == 0 or y_ij.sum() == len(y_ij):
            continue
        a_mean = np.full(int(S), 1.0 / S)
        ar_mean.append(roc_auc_score(y_ij, a_mean))
        ap_mean.append(average_precision_score(y_ij, a_mean))
        idx = torch.argmax(h, dim=0)
        a_max = np.array([(idx == j).sum().item() for j in range(int(S))], dtype=float) / h.shape[1]
        ar_max.append(roc_auc_score(y_ij, a_max))
        ap_max.append(average_precision_score(y_ij, a_max))
    return (np.mean(ar_mean), np.mean(ap_mean)), (np.mean(ar_max), np.mean(ap_max))


def main():
    summary_rows = []
    for tag, info in DATASETS.items():
        cia_test, cia_test_ap, cia_ar, cia_ap, mean_ar, mean_ap, max_ar, max_ap = [], [], [], [], [], [], [], []
        picks = []
        for s in SEEDS:
            ds, ly = load_test_dataset(tag, s)
            best = select_ckpt(info["dir"], s)
            if best is None:
                print(f"[skip] {tag} seed {s}: no checkpoint with train>val epoch")
                continue
            v, test_auroc, test_auprc, epoch, pt, name = best
            if not os.path.exists(pt):
                print(f"[skip] {name}.pt missing")
                continue
            ar, ap = localization(pt, ds, ly)
            cia_test.append(test_auroc)
            cia_test_ap.append(test_auprc)
            cia_ar.append(ar); cia_ap.append(ap)
            (mar, map_), (xar, xap) = mean_max_baselines(ds, ly)
            mean_ar.append(mar); mean_ap.append(map_)
            max_ar.append(xar); max_ap.append(xap)
            cfg = re.sub(r".*_criterion=CIA_", "", name).replace("_pooling=ABMIL", "")
            picks.append(f"seed{s}:{cfg}@ep{epoch}(val={v:.4f})")
        if not cia_ar:
            print(f"### {tag}: (no results)")
            continue
        print(f"### {tag}")
        print(f"    CIA+L1  classification AUROC {np.mean(cia_test):.3f} +/- {np.std(cia_test):.3f}   AUPRC {np.mean(cia_test_ap):.3f} +/- {np.std(cia_test_ap):.3f}")
        print(f"    CIA+L1  loc AUROC {np.mean(cia_ar):.3f} +/- {np.std(cia_ar):.3f}   AUPRC {np.mean(cia_ap):.3f} +/- {np.std(cia_ap):.3f}")
        print(f"    Mean    loc AUROC {np.mean(mean_ar):.3f} +/- {np.std(mean_ar):.3f}   AUPRC {np.mean(mean_ap):.3f} +/- {np.std(mean_ap):.3f}")
        print(f"    Max     loc AUROC {np.mean(max_ar):.3f} +/- {np.std(max_ar):.3f}   AUPRC {np.mean(max_ap):.3f} +/- {np.std(max_ap):.3f}")
        print(f"    picks: {' '.join(picks)}")
        print()
        summary_rows.append({
            "dataset": tag,
            "n_seeds": len(cia_ar),
            "classification_auroc": f"{np.mean(cia_test):.3f} +/- {np.std(cia_test):.3f}",
            "classification_auprc": f"{np.mean(cia_test_ap):.3f} +/- {np.std(cia_test_ap):.3f}",
            "loc_auroc": f"{np.mean(cia_ar):.3f} +/- {np.std(cia_ar):.3f}",
            "loc_auprc": f"{np.mean(cia_ap):.3f} +/- {np.std(cia_ap):.3f}",
            "mean_loc_auroc": f"{np.mean(mean_ar):.3f} +/- {np.std(mean_ar):.3f}",
            "max_loc_auroc": f"{np.mean(max_ar):.3f} +/- {np.std(max_ar):.3f}",
            "picks": " ".join(picks),
        })
    out = pd.DataFrame(summary_rows)
    out.to_csv(f"{E}/_cia_l1_1000ep_summary.csv", index=False)
    print(f"wrote {E}/_cia_l1_1000ep_summary.csv")


if __name__ == "__main__":
    main()
