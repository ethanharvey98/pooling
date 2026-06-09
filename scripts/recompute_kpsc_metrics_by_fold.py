"""Recompute KPSC zero-shot metrics on the 5 site-based test folds.

Joins per_subject_scores.csv with study_to_site.csv, restricts to each fold's
test sites, and reports AUROC / AUPRC / balanced accuracy per fold for each
samp_* aggregation, plus the 5-fold mean and std.

Usage:
    python scripts/recompute_kpsc_metrics_by_fold.py \\
        --eval_dir outputs/qoq_kpsc800_wmd_38793823
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, roc_auc_score,
)

FOLDS = {
    1: [9],
    2: [1, 4, 7, 10, 11],
    3: [2, 6],
    4: [3, 8],
    5: [5],
}

SCORE_COLS = [
    "samp_max_bag", "samp_mean_bag",
    "samp_T1_max",  "samp_T1_mean",
    "samp_T2_max",  "samp_T2_mean",
]


def metrics_for(labels: np.ndarray, scores: np.ndarray) -> dict:
    scores = np.nan_to_num(scores, nan=0.5)
    try:    auroc = float(roc_auc_score(labels, scores))
    except Exception: auroc = float("nan")
    try:    auprc = float(average_precision_score(labels, scores))
    except Exception: auprc = float("nan")
    try:    bal   = float(balanced_accuracy_score(labels, (scores >= 0.5).astype(int)))
    except Exception: bal   = float("nan")
    return {"auroc": auroc, "auprc": auprc, "bal_acc": bal, "n": int(len(labels))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=Path, required=True,
                    help="Directory containing per_subject_scores.csv and study_to_site.csv")
    ap.add_argument("--out_json", type=Path, default=None,
                    help="Optional output for per-fold + summary metrics (default: <eval_dir>/metrics_by_fold.json)")
    args = ap.parse_args()

    subj = pd.read_csv(args.eval_dir / "per_subject_scores.csv")
    sites = pd.read_csv(args.eval_dir / "study_to_site.csv")
    df = subj.merge(sites, on="study_id", how="left")
    missing = df["SiteID"].isna().sum()
    if missing:
        print(f"[warn] {missing} subjects had no SiteID after merge; dropping")
        df = df.dropna(subset=["SiteID"])
    df["SiteID"] = df["SiteID"].astype(int)

    print(f"merged: {len(df)} subjects, prevalence={df['label'].mean():.3f}")
    print(f"site counts:\n{df['SiteID'].value_counts().sort_index().to_string()}\n")

    out = {"folds": {}, "summary": {}}

    for col in SCORE_COLS:
        per_fold = []
        for fold, sites_in in FOLDS.items():
            sub = df[df["SiteID"].isin(sites_in)]
            m = metrics_for(sub["label"].to_numpy(), sub[col].to_numpy())
            per_fold.append({"fold": fold, "sites": sites_in, **m})
        out["folds"][col] = per_fold

        # Mean / std across the 5 folds (macro across folds).
        for k in ("auroc", "auprc", "bal_acc"):
            vals = np.array([f[k] for f in per_fold], dtype=float)
            vals = vals[~np.isnan(vals)]
            out["summary"].setdefault(col, {})[k] = {
                "mean": float(vals.mean()) if len(vals) else float("nan"),
                "std":  float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            }

    # --- print a per-column table ---
    for col in SCORE_COLS:
        print(f"\n=== {col} ===")
        rows = [{"fold": f["fold"], "sites": "{" + ",".join(map(str, f["sites"])) + "}",
                 "n": f["n"], "auroc": f["auroc"], "auprc": f["auprc"], "bal_acc": f["bal_acc"]}
                for f in out["folds"][col]]
        rows_df = pd.DataFrame(rows)
        s = out["summary"][col]
        print(rows_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"  mean: auroc={s['auroc']['mean']:.3f}±{s['auroc']['std']:.3f}  "
              f"auprc={s['auprc']['mean']:.3f}±{s['auprc']['std']:.3f}  "
              f"bal_acc={s['bal_acc']['mean']:.3f}±{s['bal_acc']['std']:.3f}")

    dst = args.out_json or (args.eval_dir / "metrics_by_fold.json")
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
