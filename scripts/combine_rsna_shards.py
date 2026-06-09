"""Concat RSNA eval shards. Recompute subject-level aggregations
(max / topk_mean / mean) from the per-slice probs so we match the repo's
qoq_eval_utils.aggregate() convention regardless of what each shard wrote.

Usage:
    python scripts/combine_rsna_shards.py \\
        --shards outputs/qoq_rsna_<arrayjobid>_shard0 \\
                 outputs/qoq_rsna_<arrayjobid>_shard1 ... \\
        --out_dir outputs/qoq_rsna_<arrayjobid>_combined
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TOP_K = 5


def metrics_for(labels, scores):
    scores = np.nan_to_num(scores, nan=0.5)
    try:    auroc = float(roc_auc_score(labels, scores))
    except Exception: auroc = float("nan")
    try:    auprc = float(average_precision_score(labels, scores))
    except Exception: auprc = float("nan")
    return {"auroc": auroc, "auprc": auprc, "n": int(len(labels))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, nargs="+", required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    slices = pd.concat([pd.read_csv(s / "per_slice_scores.csv") for s in args.shards],
                       ignore_index=True)
    # Per-subject labels come from any shard's per_subject_scores.csv.
    subj_labels = pd.concat([pd.read_csv(s / "per_subject_scores.csv")[["study_id", "label"]]
                             for s in args.shards], ignore_index=True).drop_duplicates("study_id")

    # Recompute subject-level (max, topk_mean, mean) directly from per-slice probs.
    def agg(group):
        arr = group["prob_yes"].to_numpy()
        k = min(TOP_K, arr.size)
        return pd.Series({
            "n_slices":  int(arr.size),
            "max":       float(arr.max()),
            "topk_mean": float(np.sort(arr)[-k:].mean()),
            "mean":      float(arr.mean()),
        })

    subjs = slices.groupby("study_id", sort=False).apply(agg).reset_index()
    subjs = subjs.merge(subj_labels, on="study_id", how="left")
    subjs["label"] = subjs["label"].astype(int)

    slices.to_csv(args.out_dir / "per_slice_scores.csv", index=False)
    subjs.to_csv(args.out_dir / "per_subject_scores.csv", index=False)

    labels = subjs["label"].to_numpy()
    metrics = {col: metrics_for(labels, subjs[col].to_numpy())
               for col in ("max", "topk_mean", "mean")}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"combined {len(subjs)} subjects, {len(slices)} slices -> {args.out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
