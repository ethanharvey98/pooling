"""Same entropy + entropy-matched smoothing control as eval_bandwidth_matched.py, but the
ABMIL / ABMIL + NG checkpoints are the per-seed (alpha, lr) picks hard-coded in the notebooks
(rsna_ich / rsna_pe / rsna_at / synthetic_data) instead of the automatic val_auroc selection.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import eval_bandwidth_matched as bw  # noqa: E402

EH, HOME_EXP, DATA, SEEDS = bw.EH, bw.HOME_EXP, bw.DATA, bw.SEEDS


def from_picks(dir_fn, criterion, picks):
    def f(seed):
        a, lr = picks[seed]
        p = f"{dir_fn(seed)}/alpha={a}_criterion={criterion}_lr={lr}_pooling=ABMIL_seed={seed}.pt"
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p
    return f


def fixed(d):
    return lambda seed: d


CONFIGS = {
    "Semi-Synthetic": dict(
        kind="semi",
        abmil=from_picks(lambda s: bw.semi_dir("varying_n_embedding_level=True", s), "L1",
                         {1001: ("0.01", "0.1"), 2001: ("0.01", "0.1"), 3001: ("0.001", "0.001")}),
        ng=from_picks(lambda s: bw.semi_dir("varying_n_beta=1.0_embedding_level=True", s), "GuidedL1",
                      {1001: ("0.01", "0.1"), 2001: ("0.1", "0.0001"), 3001: ("0.001", "0.01")}),
    ),
    "Head CT": dict(
        kind="std", data_dir=f"{DATA}/encoded_RSNA_ICH_full_dataset/ViT_B_16",
        abmil=bw.head_abmil,
        ng=from_picks(fixed(f"{EH}/RSNA_ICH_full_dataset_beta=1.0_embedding_level=True"), "GuidedL1",
                      {1001: ("0.0001", "0.01"), 2001: ("0.0001", "0.01"), 3001: ("0.0001", "0.001")}),
    ),
    "Chest CT": dict(
        kind="std", data_dir=f"{DATA}/encoded_RSNA_PE/ViT_B_16",
        abmil=from_picks(fixed(f"{EH}/RSNA_PE_embedding_level=True"), "L1",
                         {1001: ("1e-05", "0.01"), 2001: ("0.001", "0.01"), 3001: ("0.001", "0.01")}),
        ng=from_picks(fixed(f"{EH}/RSNA_PE_beta=1.0_embedding_level=True"), "GuidedL1",
                      {1001: ("0.001", "0.01"), 2001: ("1e-05", "0.01"), 3001: ("0.001", "0.01")}),
    ),
    "Abdomen CT": dict(
        kind="std", data_dir=f"{DATA}/encoded_RSNA_AT/ViT_B_16",
        abmil=from_picks(fixed(f"{EH}/RSNA_AT_embedding_level=True"), "L1",
                         {1001: ("0.001", "0.1"), 2001: ("0.001", "0.1"), 3001: ("0.001", "0.01")}),
        ng=from_picks(fixed(f"{EH}/RSNA_AT_beta=1.0_embedding_level=True"), "GuidedL1",
                      {1001: ("0.001", "0.01"), 2001: ("0.0001", "0.1"), 3001: ("0.0001", "0.1")}),
    ),
}

LABELS = ["ABMIL", "ABMIL + NG", "ABMIL smoothed (entropy-matched)"]


def main():
    print("device:", bw.device, flush=True)
    rows = []
    for ds_name, cfg in CONFIGS.items():
        agg = {k: {"H_norm": [], "P1": [], "Prec10": []} for k in LABELS}
        sigmas = []
        for seed in SEEDS:
            val_ds, val_ly = bw.load_split(cfg, "val", seed)
            test_ds, test_ly = bw.load_split(cfg, "test", seed)
            ab_path, ng_path = cfg["abmil"](seed), cfg["ng"](seed)
            print(f"[{ds_name}] seed {seed}: ABMIL={os.path.basename(ab_path)}  NG={os.path.basename(ng_path)}", flush=True)
            m_ab, m_ng = bw.load_model(ab_path), bw.load_model(ng_path)

            val_ab = bw.collect(m_ab, val_ds, val_ly)
            target = float(np.mean([bw.h_norm(a) for a, _ in bw.collect(m_ng, val_ds, val_ly)]))
            sigma = bw.match_sigma(val_ab, target)
            sigmas.append(sigma)

            test_ab = bw.collect(m_ab, test_ds, test_ly)
            res = {
                "ABMIL": bw.evaluate(test_ab),
                "ABMIL + NG": bw.evaluate(bw.collect(m_ng, test_ds, test_ly)),
                "ABMIL smoothed (entropy-matched)": bw.evaluate(test_ab, lambda a: bw.smooth(a, sigma)),
            }
            print(f"    matched sigma={sigma:.2f} (target val H_norm={target:.3f})", flush=True)
            for k, r in res.items():
                print(f"    {k:34s} H_norm={r['H_norm']:.3f}  P@1={r['P1']:.3f}  P@rec10={r['Prec10']:.3f}", flush=True)
                for kk in agg[k]:
                    agg[k][kk].append(r[kk])

        print(f"=== {ds_name} ===", flush=True)
        for k, v in agg.items():
            row = {
                "dataset": ds_name, "method": k,
                "H_norm": f"{np.mean(v['H_norm']):.3f} +/- {np.std(v['H_norm']):.3f}",
                "P@1": f"{np.mean(v['P1']):.3f} +/- {np.std(v['P1']):.3f}",
                "P@rec10": f"{np.mean(v['Prec10']):.3f} +/- {np.std(v['Prec10']):.3f}",
                "sigma_per_seed": ",".join(f"{s:.2f}" for s in sigmas) if "smoothed" in k else "",
            }
            rows.append(row)
            print(f"  {k:34s} H_norm={row['H_norm']} | P@1={row['P@1']} | P@rec10={row['P@rec10']} {row['sigma_per_seed']}", flush=True)
    pd.DataFrame(rows).to_csv(f"{HOME_EXP}/_bandwidth_matched_notebook_picks.csv", index=False)


if __name__ == "__main__":
    main()
