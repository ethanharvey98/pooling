"""Precision at low recall for slice-level localization (RSNA ICH, positive test bags).

Motivation: a method can have a low overall localization AUROC / AUPRC yet still be
"right at the top" -- its highest-scored slices are reliably lesion slices even if it
misses the rest of the same lesion. AUROC / AUPRC summarize the whole PR curve; this
script reports its left end directly.

For each positive test bag:
  - build a per-slice score vector (attention, pooled selection, or a fixed prior)
  - PR curve vs the per-slice lesion labels (lengths_y)
  - interpolated precision at recall in RECALLS, plus P@1 (is the single top-scored
    slice a lesion?) and AUPRC
Average over positive bags, then report mean +/- std over the 3 seeds.

Methods and their notebook-selected configs are in METHODS below. Run from the repo
root in the neuroimg_gpu env:
    python scripts/eval_precision_at_low_recall.py
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
import utils  # noqa: E402

DATA_DIR = "/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16"
EH = "/cluster/tufts/hugheslab/eharve06/pooling/experiments"
HOME_EXP = "/cluster/home/zmou01/pooling/experiments"
# eharve06's checkpoints first (match the notebook exactly); the retrained ones under
# HOME_EXP are only a reproducibility cross-check (SmAP/TransMIL agree within ~0.001).
L1_DIRS = [f"{EH}/RSNA_ICH_full_dataset_embedding_level=True",
           f"{HOME_EXP}/RSNA_ICH_full_dataset_baselines_embedding_level=True"]
CEIL_DIRS = [f"{HOME_EXP}/RSNA_ICH_full_dataset_best_possible_instance-level",
             f"{EH}/RSNA_ICH_full_dataset_best_possible_instance-level"]


def _find(dirs, name):
    for d in dirs:
        p = f"{d}/{name}.pt"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)
FKL_DIR = "/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_NG_variance_ablation_fkl_sweep_embedding_level=True"
SEEDS = [1001, 2001, 3001]
RECALLS = [0.05, 0.10, 0.25, 0.50]
OUT_CSV = "/cluster/home/zmou01/pooling/experiments/precision_at_low_recall_rsna_ich.csv"

_MODEL_CACHE = {}


def _load(key, build, ckpt):
    if key not in _MODEL_CACHE:
        m = build()
        sd = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(sd.get("state_dict", sd))
        m.eval()
        _MODEL_CACHE[key] = m
    return _MODEL_CACHE[key]


# --- per-slice score functions: (h_i [S,768], S_i, seed) -> np.array [S] ------------
def score_centered_gaussian(h, S, seed):
    # log-space to avoid underflow; mu/sigma from the empirical positive-slice
    # position distribution (same parameterization as NG's "frozen" mode),
    # scaled by bag length S instead of a fixed absolute sigma=1.
    x = torch.arange(1, S + 1, dtype=torch.float32)
    a = utils.log_normal_pdf(x, mu=0.52 * S, sigma=0.115 * S)
    return a.numpy()


def score_max(h, S, seed):
    # where does max-pooling read each feature channel from
    idx = torch.argmax(h, dim=0)  # [768]
    return np.array([(idx == j).sum().item() for j in range(S)], dtype=float) / h.shape[1]


def score_mean(h, S, seed):
    return np.full(S, 1.0 / S)


def _pool_scorer(pooling, picks):
    def f(h, S, seed):
        alpha, lr = picks[seed]
        name = f"alpha={alpha}_criterion=L1_lr={lr}_pooling={pooling}_seed={seed}"
        m = _load((pooling, seed), lambda: models.PoolClf(768, 1, pooling=pooling),
                  _find(L1_DIRS, name))
        with torch.no_grad():
            _, a = m(h, (S,))
        return torch.mean(a, dim=1).cpu().numpy().flatten()
    return f


def _guided_transmil_scorer(h, S, seed):
    # forward-KL Normal Guidance on TransMIL (notebook pick: alpha=1e-4, lr=0.001)
    d = f"{EH}/RSNA_ICH_full_dataset_beta=1.0_embedding_level=True"
    name = f"alpha=0.0001_criterion=GuidedL1_lr=0.001_pooling=TransMIL_seed={seed}"
    m = _load(("gtransmil", seed), lambda: models.PoolClf(768, 1, pooling="TransMIL"),
              f"{d}/{name}.pt")
    with torch.no_grad():
        _, a = m(h, (S,))
    return torch.mean(a, dim=1).cpu().numpy().flatten()


def _fkl_ng_scorer(ng_mode):
    def pick(seed):
        best = None
        for alpha in ["0.001", "0.0001", "1e-05"]:
            for lr in ["0.1", "0.01", "0.001"]:
                name = f"alpha={alpha}_criterion=GuidedL1_div=fkl_ng={ng_mode}_lr={lr}_pooling=ABMIL_seed={seed}"
                csv = f"{FKL_DIR}/{name}.csv"
                if not os.path.exists(csv):
                    continue
                g = pd.read_csv(csv)
                g = g[g.train_auroc > g.val_auroc]
                if g.empty:
                    continue
                v = g.val_auroc.max()
                if best is None or v > best[0]:
                    best = (v, name)
        return best[1]

    def f(h, S, seed):
        name = pick(seed)
        m = _load(("ng", seed), lambda: models.PoolClf(768, 1, pooling="ABMIL"),
                  f"{FKL_DIR}/{name}.pt")
        with torch.no_grad():
            _, a = m(h, (S,))
        return torch.mean(a, dim=1).cpu().numpy().flatten()
    return f


def _ceiling_scorer(alpha):
    # best-possible instance-level: per-slice classifier logit. Feed the whole bag
    # (the 12-slice InstanceConv1d needs its context) -- matches the notebook.
    # NOTE: cell 32 of the notebook hard-codes alpha=0.0; a plain val-AUROC sweep
    # picks alpha=1e-05 (marginally higher slice-level val). The two are within noise
    # here, but the ceiling's hyperparameter selection is inconsistent with the rest
    # of the table, so both are reported.
    def f(h, S, seed):
        name = f"alpha={alpha}_criterion=L1_lr=0.01_seed={seed}"
        m = _load(("ceil", alpha, seed),
                  lambda: models.InstanceLevelClassifier(768, 1, kernel_size=12),
                  _find(CEIL_DIRS, name))
        with torch.no_grad():
            out, _ = m(h, (S,))
        return torch.mean(out, dim=1).cpu().numpy().flatten()
    return f


def _abmil_scorer(h, S, seed):
    # (alpha, lr) matching the notebook's cell-35 abmil localization array.
    # seed 2001 uses alpha=0.0001, lr=0.1 there -- NOT the alpha=1e-06, lr=0.1 that the
    # notebook's own bag-level selection (cell 20) prints. Same "hyperparameter selection
    # is inconsistent" issue as the Best-in-Class Ceiling; both lr=0.1 configs localise
    # poorly here (lr=0.01 would give ~0.76).
    alpha, lr = {1001: ("0.0001", "0.001"), 2001: ("0.0001", "0.1"), 3001: ("1e-05", "0.01")}[seed]
    name = f"alpha={alpha}_criterion=L1_lr={lr}_pooling=ABMIL_seed={seed}"
    m = _load(("abmil", seed), lambda: models.PoolClf(768, 1, pooling="ABMIL"),
              _find(L1_DIRS, name))
    with torch.no_grad():
        _, a = m(h, (S,))
    return torch.mean(a, dim=1).cpu().numpy().flatten()


METHODS = [
    ("Centered Gaussian", score_centered_gaussian),
    ("Max pooling", score_max),
    ("Mean pooling", score_mean),
    ("ABMIL (no guidance)", _abmil_scorer),
    ("Smooth Operator (SmAP)", _pool_scorer("SmAP", {1001: ("0.0001", "0.01"), 2001: ("0.0001", "0.01"), 3001: ("0.0001", "0.01")})),
    ("TransMIL", _pool_scorer("TransMIL", {1001: ("0.0", "0.01"), 2001: ("0.0001", "0.001"), 3001: ("0.0001", "0.001")})),
    ("Smooth Operator TransMIL (SmTAP)", _pool_scorer("SmTAP", {1001: ("0.0001", "0.01"), 2001: ("0.0001", "0.001"), 3001: ("0.0001", "0.001")})),
    ("TransMIL + NG", _guided_transmil_scorer),
    ("ABMIL + NG (ours)", _fkl_ng_scorer("full")),
    ("Best-in-Class Ceiling (alpha=0.0, notebook)", _ceiling_scorer("0.0")),
    ("Best-in-Class Ceiling (alpha=1e-05, val-selected)", _ceiling_scorer("1e-05")),
]


def _prec_at_recall(y, s, r):
    p, rec, _ = precision_recall_curve(y, s)
    ok = rec >= r
    return float(np.max(p[ok])) if ok.any() else float("nan")


def main():
    tests = {}
    for s in SEEDS:
        d = torch.load(f"{DATA_DIR}/seed={s}/test.pt", map_location="cpu", weights_only=False)
        tests[s] = (datasets.MILTensorDataset(d["X"], d["lengths"], d["y"]), d["lengths_y"])

    cols = [f"P@rec{int(r * 100)}" for r in RECALLS] + ["P@1", "AUPRC"]
    out_rows = []
    for name, scorer in METHODS:
        agg = {c: [] for c in cols}
        for seed in SEEDS:
            ds, ly = tests[seed]
            per_bag = {c: [] for c in cols}
            for i in range(len(ds)):
                h_i, S_i, y_i = ds[i]
                y_ij = np.asarray(ly[i], dtype=float)
                if y_i != 1.0 or len(y_ij) != S_i or y_ij.sum() == 0 or y_ij.sum() == len(y_ij):
                    continue
                a = np.asarray(scorer(h_i, int(S_i), seed), dtype=float).flatten()
                for r in RECALLS:
                    per_bag[f"P@rec{int(r * 100)}"].append(_prec_at_recall(y_ij, a, r))
                per_bag["P@1"].append(float(y_ij[int(np.argmax(a))]))
                per_bag["AUPRC"].append(average_precision_score(y_ij, a))
            for c in cols:
                agg[c].append(np.mean(per_bag[c]))
        row = {"method": name}
        row.update({c: f"{np.mean(agg[c]):.3f} +/- {np.std(agg[c]):.3f}" for c in cols})
        out_rows.append(row)
        print(f"{name:34s} | " + " | ".join(f"{row[c]:>15s}" for c in cols))

    pd.DataFrame(out_rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
