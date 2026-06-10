"""Recompute RSNA metrics on each of the 3 paper-comparable seeded test splits.

Mirrors the patient-level split in `src/encode_rsna_full.py` (seeds 1001/2001/3001,
test_size=1/6, then val_size=1/5 of train+val, stratified by per-patient majority
scan_label). Produces per-seed (AUROC, AUPRC) for `max` and `mean` pooling, plus
mean ± std across the 3 seeds, plus a LaTeX table fragment.

Usage (on the cluster, where the labels CSV and npz dir are available):

    python scripts/rsna_seed_metrics.py \\
        --combined outputs/qoq_rsna_<arrayjobid>_combined
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEEDS = (1001, 2001, 3001)
POOLS = ("max", "mean")


def metric_pair(labels, scores):
    scores = np.nan_to_num(scores, nan=0.5)
    try:    auroc = float(roc_auc_score(labels, scores))
    except Exception: auroc = float("nan")
    try:    auprc = float(average_precision_score(labels, scores))
    except Exception: auprc = float("nan")
    return auroc, auprc


def patient_test_ids(labels_df: pd.DataFrame, seed: int) -> set[str]:
    """Reproduce the patient-level test partition from src/encode_rsna_full.py."""
    grouped = labels_df.groupby("Patient ID")["scan_label"].agg(lambda x: x.mode()[0]).reset_index()
    ids, id_labels = grouped["Patient ID"], grouped["scan_label"]
    train_and_val_ids, test_ids, _, _ = train_test_split(
        ids, id_labels, test_size=1/6, random_state=seed, stratify=id_labels)
    return set(test_ids.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", type=Path, required=True)
    ap.add_argument("--labels_csv", type=Path,
                    default=Path("/cluster/tufts/hugheslab/datasets/RSNA_ICH/full_dataset_labels.csv"))
    ap.add_argument("--numpy_dir", type=Path,
                    default=Path("/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy"))
    args = ap.parse_args()

    # --- per-subject scores (one row per study_id) ---
    subj = pd.read_csv(args.combined / "per_subject_scores.csv")
    print(f"per_subject_scores: {len(subj)} studies")

    # --- replay encode_rsna_full.py's filtering + patient-level scan_label ---
    labels_df = pd.read_csv(args.labels_csv)
    labels_df["scan_label"] = labels_df["Any"].apply(
        lambda x: 1 if any(ast.literal_eval(x)) else 0)
    labels_df["path"] = labels_df["Study ID"].apply(lambda x: args.numpy_dir / f"{x}.npz")
    n_before = len(labels_df)
    labels_df = labels_df[labels_df["path"].apply(Path.exists)].reset_index(drop=True)
    print(f"labels_df after npz filter: {len(labels_df)}/{n_before}")

    # study_id -> patient_id map (for joining test patients to our results)
    s2p = dict(zip(labels_df["Study ID"], labels_df["Patient ID"]))
    subj["patient_id"] = subj["study_id"].map(s2p)
    n_unmapped = subj["patient_id"].isna().sum()
    if n_unmapped:
        print(f"[warn] {n_unmapped} per-subject rows had no Patient ID match — dropping")
        subj = subj.dropna(subset=["patient_id"]).reset_index(drop=True)

    # --- per-seed metrics ---
    per_seed = {pool: {"auroc": [], "auprc": [], "n": []} for pool in POOLS}
    print()
    for seed in SEEDS:
        test_pats = patient_test_ids(labels_df, seed)
        in_test = subj["patient_id"].isin(test_pats)
        sub = subj[in_test]
        if len(sub) == 0:
            print(f"seed {seed}: zero overlap, skipping")
            continue
        labels = sub["label"].to_numpy()
        print(f"seed {seed}: n_test_studies={len(sub)} (prevalence={labels.mean():.3f})")
        for pool in POOLS:
            auroc, auprc = metric_pair(labels, sub[pool].to_numpy())
            per_seed[pool]["auroc"].append(auroc)
            per_seed[pool]["auprc"].append(auprc)
            per_seed[pool]["n"].append(int(len(sub)))
            print(f"    {pool:5s}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}")

    # --- mean ± std ---
    summary = {}
    for pool in POOLS:
        a = np.array(per_seed[pool]["auroc"], dtype=float)
        p = np.array(per_seed[pool]["auprc"], dtype=float)
        summary[pool] = {
            "auroc_mean": float(a.mean()), "auroc_std": float(a.std(ddof=1)),
            "auprc_mean": float(p.mean()), "auprc_std": float(p.std(ddof=1)),
            "per_seed_n":     per_seed[pool]["n"],
            "per_seed_auroc": per_seed[pool]["auroc"],
            "per_seed_auprc": per_seed[pool]["auprc"],
        }

    print("\n=== 3-seed mean ± std ===")
    for pool, s in summary.items():
        print(f"{pool:5s}  AUROC = {s['auroc_mean']:.3f} \u00b1 {s['auroc_std']:.3f}"
              f"   AUPRC = {s['auprc_mean']:.3f} \u00b1 {s['auprc_std']:.3f}")

    out_path = args.combined / "metrics_by_seed.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")

    # --- LaTeX fragment ---
    tex = []
    tex.append(r"\begin{tabular}{l c c}")
    tex.append(r"\toprule")
    tex.append(r"Pooling & AUROC & AUPRC \\")
    tex.append(r"\midrule")
    for pool in POOLS:
        s = summary[pool]
        tex.append(
            f"{pool} & ${s['auroc_mean']:.3f} \\pm {s['auroc_std']:.3f}$"
            f" & ${s['auprc_mean']:.3f} \\pm {s['auprc_std']:.3f}$ \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex_path = args.combined / "metrics_by_seed_table.tex"
    tex_path.write_text("\n".join(tex) + "\n")
    print(f"wrote {tex_path}")
    print("\n" + "\n".join(tex))


if __name__ == "__main__":
    main()
