"""Zero-shot eval of ddvd233/QoQ-Med-VL-7B on OASIS-3 Alzheimer's classification.

Smoke test:
    python scripts/eval_qoq_med_oasis3.py --max_subjects 4 \
        --central_fraction 0.2 --batch_size 4 --output_dir outputs/smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import qoq_eval_utils as qu  # noqa: E402

MOD_IDX = {"T1": 0, "T2": 1}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--numpy_dir", type=Path,
                   default=Path("/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy"))
    p.add_argument("--labels_csv", type=Path, default=None)
    p.add_argument("--model_id", default="ddvd233/QoQ-Med-VL-7B")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--central_fraction", type=float, default=0.4)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_subjects", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--modalities", default="T1,T2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_cache_assert", action="store_true")
    return p.parse_args()


def score_batch(model, processor, yes_id, no_id, batch):
    """batch: list of (PIL, modality). Returns list of P(Yes)."""
    device = next(model.parameters()).device
    texts = [processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": qu.PROMPT_TEMPLATE.format(modality=m)}]}],
        tokenize=False, add_generation_prompt=True)
        for img, m in batch]
    inputs = processor(text=texts, images=[img for img, _ in batch],
                       padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits
    last = logits[torch.arange(logits.size(0), device=logits.device),
                  inputs["attention_mask"].sum(1) - 1]
    yn = torch.stack([last[:, yes_id], last[:, no_id]], dim=-1).float()
    return F.softmax(yn, dim=-1)[:, 0].cpu().tolist()


def metrics_for(labels, scores):
    try:
        auroc = float(roc_auc_score(labels, scores))
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(labels, scores))
    except Exception:
        auprc = float("nan")
    try:
        bal = float(balanced_accuracy_score(labels, (np.asarray(scores) >= 0.5).astype(int)))
    except Exception:
        bal = float("nan")
    return {"auroc": auroc, "auprc": auprc, "bal_acc": bal, "n": int(len(labels))}


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

    labels_df = qu.load_labels(args.numpy_dir, args.labels_csv)
    if args.max_subjects > 0:
        labels_df = labels_df.head(args.max_subjects).reset_index(drop=True)
    print(f"[eval] {len(labels_df)} subjects", flush=True)

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

    slice_rows, subj_rows = [], []
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="subjects"):
        vol = qu.load_volume(Path(row["npz_path"]))  # (D, C, H, W)
        D = vol.shape[0]
        idxs = qu.central_slice_indices(D, args.central_fraction)
        bounds = {m: qu.percentile_bounds(vol[:, MOD_IDX[m]]) for m in modalities}

        tasks = [(s, m, qu.slice_to_pil(vol[s, MOD_IDX[m]], *bounds[m]))
                 for s in idxs for m in modalities]

        probs = []
        for i in range(0, len(tasks), args.batch_size):
            chunk = tasks[i:i + args.batch_size]
            probs.extend(score_batch(model, processor, yes_id, no_id,
                                     [(pil, m) for _, m, pil in chunk]))

        mod_probs = {m: [] for m in ["T1", "T2"]}
        for (s, m, _), p in zip(tasks, probs):
            slice_rows.append({"subject": row["Subject"], "mr_id": row["MR ID"],
                               "slice_idx": int(s), "modality": m, "prob_yes": float(p)})
            mod_probs[m].append(float(p))

        agg = qu.aggregate(probs, args.top_k)
        t1 = qu.aggregate(mod_probs["T1"], args.top_k)
        t2 = qu.aggregate(mod_probs["T2"], args.top_k)
        subj_rows.append({
            "subject": row["Subject"], "mr_id": row["MR ID"],
            "label": int(row["Alzheimer's"]), "n_slices": len(probs),
            "max_bag": agg["max"], "topk_mean_bag": agg["topk_mean"], "mean_bag": agg["mean"],
            "T1_max": t1["max"], "T1_topk_mean": t1["topk_mean"], "T1_mean": t1["mean"],
            "T2_max": t2["max"], "T2_topk_mean": t2["topk_mean"], "T2_mean": t2["mean"],
        })

    pd.DataFrame(slice_rows).to_csv(args.output_dir / "per_slice_scores.csv", index=False)
    subj_df = pd.DataFrame(subj_rows)
    subj_df.to_csv(args.output_dir / "per_subject_scores.csv", index=False)

    labels = subj_df["label"].to_numpy()
    metrics = {agg: metrics_for(labels, subj_df[agg].to_numpy())
               for agg in ("max_bag", "topk_mean_bag", "mean_bag",
                           "T1_max", "T1_topk_mean", "T1_mean",
                           "T2_max", "T2_topk_mean", "T2_mean")}
    metrics["_yn"] = yn_info
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output_dir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "_yn"}, indent=2))


if __name__ == "__main__":
    main()
