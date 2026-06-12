"""Recompute RSNA metrics on each of the 3 paper-comparable seeded test splits.

Two metric levels are reported:

  - Subject-level: per-study aggregated score (max / mean across slices)
    vs. study label (Any positive slice -> 1). Same as src/layers.py::Max / Mean.
  - Instance-level: per-slice prob_yes vs. per-slice ground truth from
    `Any[slice_idx]`. Mirrors the sentinel logic in src/encode_rsna_full.py
    (if len(Any) != n_slices for a study, skip that study's instances).

Patient-level test partitions reproduced from src/encode_rsna_full.py for seeds
1001 / 2001 / 3001.

Usage (on the cluster):
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


def patient_test_ids(labels_df: pd.DataFrame, seed: int) -> set:
    """Patient-level test partition matching src/encode_rsna_full.py."""
    grouped = labels_df.groupby("Patient ID")["scan_label"].agg(lambda x: x.mode()[0]).reset_index()
    ids, id_labels = grouped["Patient ID"], grouped["scan_label"]
    _, test_ids, _, _ = train_test_split(
        ids, id_labels, test_size=1/6, random_state=seed, stratify=id_labels)
    return set(test_ids.tolist())


def attach_instance_labels(slices_df: pd.DataFrame, any_lists: dict) -> pd.DataFrame:
    """Add inst_label column. Sentinel -1 if `Any` length != n_slices for that study."""
    n_per_study = slices_df.groupby("study_id").size().to_dict()
    out = []
    for study, n in n_per_study.items():
        any_list = any_lists.get(study)
        if any_list is None or len(any_list) != n:
            for _ in range(n):
                out.append(-1)
        else:
            for s in range(n):
                out.append(int(any_list[s]))
    # The groupby preserves first-seen order. Reorder slices_df the same way for safety.
    df = (slices_df
          .sort_values(["study_id", "slice_idx"], kind="mergesort")
          .reset_index(drop=True)
          .copy())
    df["inst_label"] = out
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", type=Path, required=True)
    ap.add_argument("--labels_csv", type=Path,
                    default=Path("/cluster/tufts/hugheslab/datasets/RSNA_ICH/full_dataset_labels.csv"))
    ap.add_argument("--numpy_dir", type=Path,
                    default=Path("/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy"))
    args = ap.parse_args()

    # --- our results ---
    subj = pd.read_csv(args.combined / "per_subject_scores.csv")
    slices = pd.read_csv(args.combined / "per_slice_scores.csv")
    print(f"per_subject_scores: {len(subj):,} studies   per_slice_scores: {len(slices):,} slices")

    # --- replay encode_rsna_full.py filter + Any parsing ---
    labels_df = pd.read_csv(args.labels_csv)
    labels_df["Any"] = labels_df["Any"].apply(ast.literal_eval)
    labels_df["scan_label"] = labels_df["Any"].apply(lambda xs: 1 if any(xs) else 0)
    labels_df["path"] = labels_df["Study ID"].apply(lambda x: args.numpy_dir / f"{x}.npz")
    n_before = len(labels_df)
    labels_df = labels_df[labels_df["path"].apply(Path.exists)].reset_index(drop=True)
    print(f"labels_df after npz filter: {len(labels_df):,}/{n_before:,}")

    s2p = dict(zip(labels_df["Study ID"], labels_df["Patient ID"]))
    any_lists = dict(zip(labels_df["Study ID"], labels_df["Any"]))

    subj["patient_id"] = subj["study_id"].map(s2p)
    slices["patient_id"] = slices["study_id"].map(s2p)
    n_unmapped_subj = subj["patient_id"].isna().sum()
    n_unmapped_sl = slices["patient_id"].isna().sum()
    if n_unmapped_subj or n_unmapped_sl:
        print(f"[warn] {n_unmapped_subj} per-subject rows and {n_unmapped_sl} per-slice rows "
              f"had no Patient ID — dropping")
        subj = subj.dropna(subset=["patient_id"]).reset_index(drop=True)
        slices = slices.dropna(subset=["patient_id"]).reset_index(drop=True)

    # --- instance labels: -1 sentinel for length mismatch ---
    slices = attach_instance_labels(slices, any_lists)
    n_valid = (slices["inst_label"] != -1).sum()
    n_mismatch_studies = slices[slices["inst_label"] == -1]["study_id"].nunique()
    print(f"valid instance labels: {n_valid:,}/{len(slices):,}   "
          f"({n_mismatch_studies:,} studies dropped for length mismatch)")

    # --- per-seed metrics ---
    per_seed_subj = {pool: {"auroc": [], "auprc": [], "n": []} for pool in POOLS}
    per_seed_inst = {"auroc": [], "auprc": [], "n": []}
    print()

    for seed in SEEDS:
        test_pats = patient_test_ids(labels_df, seed)

        # Subject-level
        sub = subj[subj["patient_id"].isin(test_pats)]
        if len(sub):
            labels = sub["label"].to_numpy()
            print(f"seed {seed}: subjects n={len(sub):,} (prev={labels.mean():.3f})")
            for pool in POOLS:
                auroc, auprc = metric_pair(labels, sub[pool].to_numpy())
                per_seed_subj[pool]["auroc"].append(auroc)
                per_seed_subj[pool]["auprc"].append(auprc)
                per_seed_subj[pool]["n"].append(int(len(sub)))
                print(f"    subj {pool:5s} AUROC={auroc:.3f} AUPRC={auprc:.3f}")

        # Instance-level (drop sentinel and non-test slices)
        slc = slices[(slices["patient_id"].isin(test_pats)) & (slices["inst_label"] != -1)]
        if len(slc):
            il = slc["inst_label"].to_numpy()
            pp = slc["prob_yes"].to_numpy()
            auroc, auprc = metric_pair(il, pp)
            per_seed_inst["auroc"].append(auroc)
            per_seed_inst["auprc"].append(auprc)
            per_seed_inst["n"].append(int(len(slc)))
            print(f"    inst        AUROC={auroc:.3f} AUPRC={auprc:.3f}   "
                  f"n_slices={len(slc):,} (pos rate {il.mean():.3f})")

    # --- mean ± std ---
    def summarize(d):
        a = np.array(d["auroc"], dtype=float)
        p = np.array(d["auprc"], dtype=float)
        return {
            "auroc_mean": float(a.mean()), "auroc_std": float(a.std(ddof=1)),
            "auprc_mean": float(p.mean()), "auprc_std": float(p.std(ddof=1)),
            "per_seed_n": d["n"],
            "per_seed_auroc": d["auroc"], "per_seed_auprc": d["auprc"],
        }

    summary = {"subject": {pool: summarize(per_seed_subj[pool]) for pool in POOLS},
               "instance": summarize(per_seed_inst)}

    print("\n=== 3-seed mean ± std ===")
    for pool in POOLS:
        s = summary["subject"][pool]
        print(f"subj {pool:5s} AUROC = {s['auroc_mean']:.3f} ± {s['auroc_std']:.3f}   "
              f"AUPRC = {s['auprc_mean']:.3f} ± {s['auprc_std']:.3f}")
    si = summary["instance"]
    print(f"inst        AUROC = {si['auroc_mean']:.3f} ± {si['auroc_std']:.3f}   "
          f"AUPRC = {si['auprc_mean']:.3f} ± {si['auprc_std']:.3f}")

    (args.combined / "metrics_by_seed.json").write_text(json.dumps(summary, indent=2))

    # --- LaTeX fragment ---
    rows = []
    for pool in POOLS:
        s = summary["subject"][pool]
        rows.append((f"Subject ({pool}-pool)",
                     s["auroc_mean"], s["auroc_std"], s["auprc_mean"], s["auprc_std"]))
    rows.append(("Instance (slice)",
                 si["auroc_mean"], si["auroc_std"], si["auprc_mean"], si["auprc_std"]))

    tex = [
        r"\begin{tabular}{l c c}",
        r"\toprule",
        r"Level & AUROC & AUPRC \\",
        r"\midrule",
    ]
    for name, am, as_, pm, ps in rows:
        tex.append(f"{name} & ${am:.3f} \\pm {as_:.3f}$ & ${pm:.3f} \\pm {ps:.3f}$ \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (args.combined / "metrics_by_seed_table.tex").write_text("\n".join(tex) + "\n")
    print("\n" + "\n".join(tex))
    print(f"\nwrote {args.combined / 'metrics_by_seed.json'}")
    print(f"wrote {args.combined / 'metrics_by_seed_table.tex'}")


if __name__ == "__main__":
    main()
