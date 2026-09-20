"""Precision at low recall for slice-level localization, Chest CT (RSNA PE) and
Abdomen CT (RSNA AT), analogous to eval_precision_at_low_recall.py (RSNA ICH).

Unlike the ICH script, the per-method (alpha, lr) picks here are NOT hand-verified
against a notebook cell-by-cell -- they're selected automatically by the project's
standard rule (highest val_auroc among epochs with train_auroc > val_auroc, swept over
the full alpha x lr grid available in eharve06's checkpoints) since there was no
paired manual cross-check against rsna_pe.ipynb / rsna_at.ipynb this time.

Run from the repo root in the neuroimg_gpu env:
    python scripts/eval_precision_at_low_recall_pe_at.py
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
RECALLS = [0.05, 0.10, 0.25, 0.50]
ALPHAS = ["0.0", "1e-06", "1e-05", "0.0001", "0.001", "0.01", "0.1", "1.0"]
LRS = ["0.0001", "0.001", "0.01", "0.1"]

DATASET_CFG = {
    "Chest CT (RSNA PE)": {
        "data_dir": f"{DATA}/encoded_RSNA_PE/ViT_B_16",
        "l1_dir": f"{EH}/RSNA_PE_embedding_level=True",
        "ceil_dir": f"{EH}/RSNA_PE_best_possible_instance-level",
        "ng_dir": f"{EH}/RSNA_PE_beta=1.0_embedding_level=True",
        "ceil_kernel": 35,
        "out_csv": "/cluster/home/zmou01/pooling/experiments/precision_at_low_recall_rsna_pe.csv",
    },
    "Abdomen CT (RSNA AT)": {
        "data_dir": f"{DATA}/encoded_RSNA_AT/ViT_B_16",
        "l1_dir": f"{EH}/RSNA_AT_embedding_level=True",
        "ceil_dir": f"{EH}/RSNA_AT_best_possible_instance-level",
        "ng_dir": f"{EH}/RSNA_AT_beta=1.0_embedding_level=True",
        "ceil_kernel": 29,
        "out_csv": "/cluster/home/zmou01/pooling/experiments/precision_at_low_recall_rsna_at.csv",
    },
}

_MODEL_CACHE = {}


def _load(key, build, ckpt):
    if key not in _MODEL_CACHE:
        m = build()
        sd = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(sd.get("state_dict", sd))
        m.eval()
        _MODEL_CACHE[key] = m
    return _MODEL_CACHE[key]


def _select(dir_, name_fn, seed):
    """Best (alpha, lr) by val_auroc s.t. train_auroc > val_auroc, over the full grid."""
    best = None
    for alpha in ALPHAS:
        for lr in LRS:
            name = name_fn(alpha, lr, seed)
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
        raise FileNotFoundError(f"no valid checkpoint in {dir_} for seed {seed}")
    return best[1]


def score_centered_gaussian(h, S, seed, cfg):
    import utils
    x = torch.arange(1, S + 1, dtype=torch.float32)
    a = utils.normal_pdf(x, mu=S / 2.0, sigma=1.0)
    return (a / a.sum()).numpy()


def score_max(h, S, seed, cfg):
    idx = torch.argmax(h, dim=0)
    return np.array([(idx == j).sum().item() for j in range(S)], dtype=float) / h.shape[1]


def score_mean(h, S, seed, cfg):
    return np.full(S, 1.0 / S)


_SELECT_CACHE = {}


def _pool_scorer(pooling):
    def f(h, S, seed, cfg):
        key = ("pool", cfg["l1_dir"], pooling, seed)
        if key not in _SELECT_CACHE:
            _SELECT_CACHE[key] = _select(cfg["l1_dir"], lambda a, lr, s: f"alpha={a}_criterion=L1_lr={lr}_pooling={pooling}_seed={s}", seed)
        name = _SELECT_CACHE[key]
        m = _load((cfg["l1_dir"], pooling, seed), lambda: models.PoolClf(768, 1, pooling=pooling), f"{cfg['l1_dir']}/{name}.pt")
        with torch.no_grad():
            _, a = m(h, (S,))
        return torch.mean(a, dim=1).cpu().numpy().flatten()
    return f


def _ng_scorer(pooling):
    def f(h, S, seed, cfg):
        key = ("ng", cfg["ng_dir"], pooling, seed)
        if key not in _SELECT_CACHE:
            _SELECT_CACHE[key] = _select(cfg["ng_dir"], lambda a, lr, s: f"alpha={a}_criterion=GuidedL1_lr={lr}_pooling={pooling}_seed={s}", seed)
        name = _SELECT_CACHE[key]
        m = _load((cfg["ng_dir"], pooling, seed), lambda: models.PoolClf(768, 1, pooling=pooling), f"{cfg['ng_dir']}/{name}.pt")
        with torch.no_grad():
            _, a = m(h, (S,))
        return torch.mean(a, dim=1).cpu().numpy().flatten()
    return f


def _ceiling_scorer(h, S, seed, cfg):
    key = ("ceil", cfg["ceil_dir"], seed)
    if key not in _SELECT_CACHE:
        _SELECT_CACHE[key] = _select(cfg["ceil_dir"], lambda a, lr, s: f"alpha={a}_criterion=L1_lr={lr}_seed={s}", seed)
    name = _SELECT_CACHE[key]
    m = _load((cfg["ceil_dir"], seed), lambda: models.InstanceLevelClassifier(768, 1, kernel_size=cfg["ceil_kernel"]), f"{cfg['ceil_dir']}/{name}.pt")
    with torch.no_grad():
        out, _ = m(h, (S,))
    return torch.mean(out, dim=1).cpu().numpy().flatten()


METHODS = [
    ("Centered Gaussian", score_centered_gaussian),
    ("Max pooling", score_max),
    ("Mean pooling", score_mean),
    ("ABMIL (no guidance)", _pool_scorer("ABMIL")),
    ("Smooth Operator (SmAP)", _pool_scorer("SmAP")),
    ("TransMIL", _pool_scorer("TransMIL")),
    ("Smooth Operator TransMIL (SmTAP)", _pool_scorer("SmTAP")),
    ("TransMIL + NG", _ng_scorer("TransMIL")),
    ("ABMIL + NG (ours)", _ng_scorer("ABMIL")),
    ("Best-in-Class Ceiling", _ceiling_scorer),
]


def _prec_at_recall(y, s, r):
    p, rec, _ = precision_recall_curve(y, s)
    ok = rec >= r
    return float(np.max(p[ok])) if ok.any() else float("nan")


def run_dataset(tag, cfg):
    print(f"\n########## {tag} ##########")
    tests = {}
    for s in SEEDS:
        d = torch.load(f"{cfg['data_dir']}/seed={s}/test.pt", map_location="cpu", weights_only=False)
        tests[s] = (datasets.MILTensorDataset(d["X"], d["lengths"], d["y"]), d["lengths_y"])

    cols = [f"P@rec{int(r * 100)}" for r in RECALLS] + ["P@1", "AUPRC"]
    out_rows = []
    for name, scorer in METHODS:
        agg = {c: [] for c in cols}
        ok = True
        for seed in SEEDS:
            ds, ly = tests[seed]
            per_bag = {c: [] for c in cols}
            try:
                for i in range(len(ds)):
                    h_i, S_i, y_i = ds[i]
                    y_ij = np.asarray(ly[i], dtype=float)
                    if y_i != 1.0 or len(y_ij) != S_i or y_ij.sum() == 0 or y_ij.sum() == len(y_ij):
                        continue
                    a = np.asarray(scorer(h_i, int(S_i), seed, cfg), dtype=float).flatten()
                    for r in RECALLS:
                        per_bag[f"P@rec{int(r * 100)}"].append(_prec_at_recall(y_ij, a, r))
                    per_bag["P@1"].append(float(y_ij[int(np.argmax(a))]))
                    per_bag["AUPRC"].append(average_precision_score(y_ij, a))
            except FileNotFoundError as e:
                print(f"[skip] {name} seed={seed}: {e}")
                ok = False
                break
            for c in cols:
                agg[c].append(np.mean(per_bag[c]))
        if not ok or not agg[cols[0]]:
            continue
        row = {"method": name}
        row.update({c: f"{np.mean(agg[c]):.3f} +/- {np.std(agg[c]):.3f}" for c in cols})
        out_rows.append(row)
        print(f"{name:34s} | " + " | ".join(f"{row[c]:>15s}" for c in cols))

    pd.DataFrame(out_rows).to_csv(cfg["out_csv"], index=False)
    print(f"wrote {cfg['out_csv']}")


def main():
    for tag, cfg in DATASET_CFG.items():
        run_dataset(tag, cfg)


if __name__ == "__main__":
    main()
