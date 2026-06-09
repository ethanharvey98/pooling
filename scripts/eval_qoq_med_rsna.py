"""Zero-shot eval of QoQ-Med-VL-7B on RSNA Intracranial Hemorrhage (CT, study-level).

Per slice:
  1) Generate a reasoning trace ending in `\\boxed{...}`.
  2) Take the response up to and including `\\boxed{`, run a single forward
     pass on prompt + image + that prefix, and read the next-token logits.
     P(Yes) is the 2-way softmax over (Yes_logit, No_logit).
  3) If the response has no `\\boxed{`, fall back to 0.5.

Study-level scores: max and mean of per-slice P(Yes).
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

MODEL_ID       = "ddvd233/QoQ-Med-VL-7B"
NUMPY_DIR      = Path("/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy")
LABELS_CSV     = Path("/cluster/tufts/hugheslab/datasets/RSNA_ICH/full_dataset_labels.csv")
BATCH_SIZE     = 8
MAX_NEW_TOKENS = 512
BOXED_OPEN     = "\\boxed{"

PROMPT = (
    "Is there evidence of intracranial hemorrhage in this CT slice? "
    "You FIRST think about the reasoning process as an internal monologue "
    "and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in \\boxed{} and must be exactly Yes or No."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--max_subjects", type=int, default=-1)
    p.add_argument("--shard_id", type=int, default=0,
                   help="0-indexed shard id; takes subjects [shard_id::num_shards]")
    p.add_argument("--num_shards", type=int, default=1)
    return p.parse_args()


def resolve_yes_no_ids(tokenizer):
    def pick(variants):
        for v in variants:
            ids = tokenizer(v, add_special_tokens=False).input_ids
            if len(ids) == 1:
                return ids[0]
        raise RuntimeError(f"none of {variants} tokenize to a single token")
    return pick(["Yes", " Yes"]), pick(["No", " No"])


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV)
    df["Any"] = df["Any"].apply(ast.literal_eval)
    df["label"] = df["Any"].apply(lambda xs: int(max(xs)))
    df["study_id"] = df["Study ID"]
    df["npz_path"] = df["study_id"].apply(lambda s: NUMPY_DIR / f"{s}.npz")
    df = df[df["npz_path"].apply(Path.exists)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No labels rows have a matching .npz in {NUMPY_DIR}")
    return df[["study_id", "label", "npz_path"]]


def slice_to_pil(slice_2d: np.ndarray) -> Image.Image:
    lo, hi = float(slice_2d.min()), float(slice_2d.max())
    img = (slice_2d - lo) / max(hi - lo, 1e-6)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([img] * 3, axis=-1), mode="RGB")


def chat_text(processor, prompt_text):
    return processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": None},
            {"type": "text",  "text":  prompt_text},
        ]}], tokenize=False, add_generation_prompt=True)


def generate_batch(model, processor, pils):
    device = next(model.parameters()).device
    texts = [chat_text(processor, PROMPT) for _ in pils]
    inputs = processor(text=texts, images=pils, padding=True,
                       return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )
    new = out[:, prompt_len:]
    return processor.tokenizer.batch_decode(new, skip_special_tokens=True)


def probe_batch(model, processor, yes_id, no_id, pils, prefixes):
    """One forward pass with prompt + prefix (ending in `\\boxed{`).
    Returns restricted-softmax P(Yes) per item.
    """
    device = next(model.parameters()).device
    texts = [chat_text(processor, PROMPT) + pre for pre in prefixes]
    inputs = processor(text=texts, images=pils, padding=True,
                       return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits
    last = logits[torch.arange(logits.size(0), device=device),
                  inputs["attention_mask"].sum(1) - 1]
    probs = F.softmax(last.float(), dim=-1)
    py, pn = probs[:, yes_id], probs[:, no_id]
    return (py / (py + pn)).cpu().tolist()


def metrics_for(labels, scores):
    scores = np.nan_to_num(scores, nan=0.5)
    try:    auroc = float(roc_auc_score(labels, scores))
    except Exception: auroc = float("nan")
    try:    auprc = float(average_precision_score(labels, scores))
    except Exception: auprc = float("nan")
    return {"auroc": auroc, "auprc": auprc, "n": int(len(labels))}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels_df = load_labels()
    if args.max_subjects > 0:
        labels_df = labels_df.head(args.max_subjects).reset_index(drop=True)
    if args.num_shards > 1:
        n_total = len(labels_df)
        labels_df = labels_df.iloc[args.shard_id::args.num_shards].reset_index(drop=True)
        print(f"[eval] shard {args.shard_id}/{args.num_shards}: "
              f"{len(labels_df)}/{n_total} studies", flush=True)
    print(f"[eval] n={len(labels_df)} studies (prevalence={labels_df['label'].mean():.3f})",
          flush=True)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    yes_id, no_id = resolve_yes_no_ids(processor.tokenizer)
    print(f"[eval] Yes/No token ids: {yes_id}/{no_id}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    slice_rows, subj_rows = [], []
    for subj_idx, (_, row) in enumerate(
        tqdm(labels_df.iterrows(), total=len(labels_df), desc="studies")
    ):
        vol = np.load(row["npz_path"])["arr_0"].transpose(3, 0, 1, 2).astype(np.float32)
        pils = [slice_to_pil(vol[s, 0]) for s in range(vol.shape[0])]

        if subj_idx == 0:
            dump = args.output_dir / "sample_images"
            dump.mkdir(parents=True, exist_ok=True)
            for s, pil in enumerate(pils):
                pil.save(dump / f"{row['study_id']}_slice{s:03d}.png")

        all_probs, all_responses = [], []
        for i in range(0, len(pils), BATCH_SIZE):
            chunk_pils = pils[i:i + BATCH_SIZE]
            responses = generate_batch(model, processor, chunk_pils)

            # Identify which items have `\boxed{` so we only probe those.
            probe_pils, probe_prefixes, probe_idx = [], [], []
            for j, r in enumerate(responses):
                idx = r.find(BOXED_OPEN)
                if idx >= 0:
                    probe_pils.append(chunk_pils[j])
                    probe_prefixes.append(r[:idx + len(BOXED_OPEN)])
                    probe_idx.append(j)

            chunk_probs = [0.5] * len(chunk_pils)
            if probe_idx:
                sub_probs = probe_batch(model, processor, yes_id, no_id,
                                        probe_pils, probe_prefixes)
                for j, p in zip(probe_idx, sub_probs):
                    chunk_probs[j] = p

            all_probs.extend(chunk_probs)
            all_responses.extend(responses)

        for s, (txt, p) in enumerate(zip(all_responses, all_probs)):
            slice_rows.append({"study_id": row["study_id"], "slice_idx": s,
                               "prob_yes": float(p), "response": txt})
        subj_rows.append({
            "study_id": row["study_id"], "label": int(row["label"]),
            "n_slices": len(all_probs),
            "max":  float(np.max(all_probs)),
            "mean": float(np.mean(all_probs)),
        })

    pd.DataFrame(slice_rows).to_csv(args.output_dir / "per_slice_scores.csv", index=False)
    subj_df = pd.DataFrame(subj_rows)
    subj_df.to_csv(args.output_dir / "per_subject_scores.csv", index=False)

    labels = subj_df["label"].to_numpy()
    metrics = {col: metrics_for(labels, subj_df[col].to_numpy()) for col in ("max", "mean")}
    metrics["_prompt"] = PROMPT
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if not k.startswith("_")}, indent=2))


if __name__ == "__main__":
    main()
