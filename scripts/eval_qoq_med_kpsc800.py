"""Zero-shot eval of ddvd233/QoQ-Med-VL-7B on KPSC 800 MRI.

Two binary tasks: CBI (covert brain infarct) and WMD (white matter disease).
Per slice we collect BOTH the deterministic Yes/No logit-softmax P(Yes) and
N sampled generations at temperature > 0 (saved verbatim).

Smoke test:
    python scripts/eval_qoq_med_kpsc800.py --task cbi --max_subjects 4 \
        --n_samples 2 --central_fraction 0.2 --batch_size 4 \
        --output_dir outputs/smoke_kpsc_cbi
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import qoq_eval_utils as qu  # noqa: E402

MOD_IDX = {"T1": 0, "T2": 1}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["cbi", "wmd"], required=True)
    p.add_argument("--numpy_dir", type=Path,
                   default=Path("/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy"))
    p.add_argument("--labels_csv", type=Path, default=None)
    p.add_argument("--model_id", default="ddvd233/QoQ-Med-VL-7B")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--central_fraction", type=float, default=0.4)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_subjects", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--modalities", default="T1,T2")
    p.add_argument("--n_samples", type=int, default=5,
                   help="Number of sampled generations per slice (in addition to "
                        "the deterministic logit pass). 0 disables sampling.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_cache_assert", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not args.skip_cache_assert:
        qu.assert_caches_in_repo(REPO)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    for m in modalities:
        if m not in MOD_IDX:
            raise ValueError(f"unknown modality {m!r}")

    labels_df = qu.load_kpsc_labels(args.numpy_dir, args.labels_csv, args.task)
    if args.max_subjects > 0:
        labels_df = labels_df.head(args.max_subjects).reset_index(drop=True)
    print(f"[eval] task={args.task} n={len(labels_df)} subjects "
          f"(prevalence={labels_df['label'].mean():.3f})", flush=True)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_id)
    yes_id, no_id, yn_info = qu.resolve_yes_no_ids(processor.tokenizer)
    print(f"[eval] Yes/No tokens: {yn_info}", flush=True)

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelCls

    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model = ModelCls.from_pretrained(
        args.model_id, torch_dtype=dtypes[args.dtype], device_map="cuda"
    ).eval()

    prompt = qu.PROMPTS[args.task]

    slice_rows, answer_rows, subj_rows = [], [], []
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="subjects"):
        vol = qu.load_volume(Path(row["npz_path"]))  # (D, C, H, W)
        D = vol.shape[0]
        idxs = qu.central_slice_indices(D, args.central_fraction)
        bounds = {m: qu.percentile_bounds(vol[:, MOD_IDX[m]]) for m in modalities}

        tasks = [(s, m, qu.slice_to_pil(vol[s, MOD_IDX[m]], *bounds[m]))
                 for s in idxs for m in modalities]

        logit_probs, sampled_probs = [], []
        mod_logit = {m: [] for m in MOD_IDX}
        mod_sampled = {m: [] for m in MOD_IDX}

        for i in range(0, len(tasks), args.batch_size):
            chunk = tasks[i:i + args.batch_size]
            batch_pm = [(pil, m) for _, m, pil in chunk]

            lp = qu.score_batch(model, processor, yes_id, no_id, batch_pm, prompt)
            sa = qu.sample_answers(model, processor, batch_pm, args.n_samples,
                                   args.temperature, prompt)

            for (s, m, _), p_logit, answers in zip(chunk, lp, sa):
                n_yes = qu.yes_count_from_answers(answers)
                p_sampled = (n_yes / args.n_samples) if args.n_samples > 0 else float("nan")
                slice_rows.append({
                    "study_id": row["study_id"], "slice_idx": int(s), "modality": m,
                    "prob_yes_logit": float(p_logit),
                    "prob_yes_sampled": p_sampled,
                    "n_sampled_yes": n_yes,
                    "n_samples": args.n_samples,
                })
                for k, txt in enumerate(answers):
                    answer_rows.append({
                        "study_id": row["study_id"], "slice_idx": int(s), "modality": m,
                        "sample_idx": k, "raw_text": txt,
                    })
                logit_probs.append(float(p_logit))
                mod_logit[m].append(float(p_logit))
                if args.n_samples > 0:
                    sampled_probs.append(p_sampled)
                    mod_sampled[m].append(p_sampled)

        agg_logit = qu.aggregate(logit_probs, args.top_k)
        t1_logit = qu.aggregate(mod_logit["T1"], args.top_k)
        t2_logit = qu.aggregate(mod_logit["T2"], args.top_k)
        agg_samp = qu.aggregate(sampled_probs, args.top_k)
        t1_samp = qu.aggregate(mod_sampled["T1"], args.top_k)
        t2_samp = qu.aggregate(mod_sampled["T2"], args.top_k)
        subj_rows.append({
            "study_id": row["study_id"], "label": int(row["label"]),
            "n_slices": len(logit_probs),
            "logit_max_bag": agg_logit["max"], "logit_topk_mean_bag": agg_logit["topk_mean"],
            "logit_mean_bag": agg_logit["mean"],
            "logit_T1_max": t1_logit["max"], "logit_T1_topk_mean": t1_logit["topk_mean"],
            "logit_T1_mean": t1_logit["mean"],
            "logit_T2_max": t2_logit["max"], "logit_T2_topk_mean": t2_logit["topk_mean"],
            "logit_T2_mean": t2_logit["mean"],
            "samp_max_bag": agg_samp["max"], "samp_topk_mean_bag": agg_samp["topk_mean"],
            "samp_mean_bag": agg_samp["mean"],
            "samp_T1_max": t1_samp["max"], "samp_T1_topk_mean": t1_samp["topk_mean"],
            "samp_T1_mean": t1_samp["mean"],
            "samp_T2_max": t2_samp["max"], "samp_T2_topk_mean": t2_samp["topk_mean"],
            "samp_T2_mean": t2_samp["mean"],
        })

    pd.DataFrame(slice_rows).to_csv(args.output_dir / "per_slice_scores.csv", index=False)
    pd.DataFrame(answer_rows).to_csv(args.output_dir / "per_slice_answers.csv", index=False)
    subj_df = pd.DataFrame(subj_rows)
    subj_df.to_csv(args.output_dir / "per_subject_scores.csv", index=False)

    agg_cols = [
        "logit_max_bag", "logit_topk_mean_bag", "logit_mean_bag",
        "logit_T1_max", "logit_T1_topk_mean", "logit_T1_mean",
        "logit_T2_max", "logit_T2_topk_mean", "logit_T2_mean",
        "samp_max_bag", "samp_topk_mean_bag", "samp_mean_bag",
        "samp_T1_max", "samp_T1_topk_mean", "samp_T1_mean",
        "samp_T2_max", "samp_T2_topk_mean", "samp_T2_mean",
    ]
    labels = subj_df["label"].to_numpy()
    metrics = {}
    for col in agg_cols:
        scores = subj_df[col].to_numpy()
        if np.isnan(scores).all():
            continue
        metrics[col] = qu.metrics_for(labels, np.nan_to_num(scores, nan=0.5))
    metrics["_yn"] = yn_info
    metrics["_task"] = args.task
    metrics["_prompt"] = prompt
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output_dir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if not k.startswith("_")}, indent=2))


if __name__ == "__main__":
    main()
