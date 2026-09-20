"""CIA+L1 classification + localization, per dataset using a SINGLE beta_effect
chosen by highest mean val_auroc averaged over the full alpha x lr x seed grid for
that beta (not letting each seed pick its own beta). Within the chosen beta, alpha/lr
are still selected independently per seed (standard project rule).
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
BETAS = ["0.2", "0.8", "1.0"]

DATASETS = {
    "semi-synthetic": {
        "dir": f"{E}/synthetic_CIA_L1_embedding_level=True",
        "test_pt": None,
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


def choose_beta(exp_dir):
    best_beta, best_score = None, -1
    for beta in BETAS:
        vals = []
        for f in glob.glob(f"{exp_dir}/*beta_effect={beta}_*.csv"):
            df = pd.read_csv(f)
            g = df[df.train_auroc > df.val_auroc]
            if g.empty:
                continue
            vals.append(g.val_auroc.max())
        if vals:
            m = np.mean(vals)
            if m > best_score:
                best_score, best_beta = m, beta
    return best_beta, best_score


def select_ckpt(exp_dir, beta, seed):
    best = None
    for f in glob.glob(f"{exp_dir}/*beta_effect={beta}_*seed={seed}.csv"):
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


def main():
    summary_rows = []
    for tag, info in DATASETS.items():
        beta, beta_score = choose_beta(info["dir"])
        print(f"### {tag}  (chosen beta_effect={beta}, mean val_auroc={beta_score:.4f})")
        cia_test, cia_test_ap, cia_ar, cia_ap = [], [], [], []
        picks = []
        for s in SEEDS:
            ds, ly = load_test_dataset(tag, s)
            best = select_ckpt(info["dir"], beta, s)
            if best is None:
                print(f"  [skip] seed {s}: no checkpoint")
                continue
            v, test_auroc, test_auprc, epoch, pt, name = best
            ar, ap = localization(pt, ds, ly)
            cia_test.append(test_auroc); cia_test_ap.append(test_auprc)
            cia_ar.append(ar); cia_ap.append(ap)
            cfg = re.sub(r".*_criterion=CIA_", "", name).replace("_pooling=ABMIL", "")
            picks.append(f"seed{s}:{cfg}@ep{epoch}(val={v:.4f})")
        print(f"  classification AUROC {np.mean(cia_test):.3f} +/- {np.std(cia_test):.3f}   AUPRC {np.mean(cia_test_ap):.3f} +/- {np.std(cia_test_ap):.3f}")
        print(f"  loc AUROC {np.mean(cia_ar):.3f} +/- {np.std(cia_ar):.3f}   AUPRC {np.mean(cia_ap):.3f} +/- {np.std(cia_ap):.3f}")
        print(f"  picks: {' '.join(picks)}")
        print()
        summary_rows.append({
            "dataset": tag, "beta_effect": beta,
            "classification_auroc": f"{np.mean(cia_test):.3f} +/- {np.std(cia_test):.3f}",
            "classification_auprc": f"{np.mean(cia_test_ap):.3f} +/- {np.std(cia_test_ap):.3f}",
            "loc_auroc": f"{np.mean(cia_ar):.3f} +/- {np.std(cia_ar):.3f}",
            "loc_auprc": f"{np.mean(cia_ap):.3f} +/- {np.std(cia_ap):.3f}",
            "picks": " ".join(picks),
        })
    out = pd.DataFrame(summary_rows)
    out.to_csv(f"{E}/_cia_l1_fixedbeta_summary.csv", index=False)
    print(f"wrote {E}/_cia_l1_fixedbeta_summary.csv")


if __name__ == "__main__":
    main()
