"""Helpers for QoQ-Med-VL-7B zero-shot evals (OASIS-3, KPSC, ...)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


PROMPTS = {
    "alzheimers": (
        "You are a radiologist. This is a 2D axial MRI slice of a human brain "
        "({modality}-weighted). Is there evidence of Alzheimer's disease in this "
        "slice? Answer with a single word: Yes or No."
    ),
    "cbi": (
        "You are a radiologist. This is a 2D axial MRI slice of a human brain "
        "({modality}-weighted). Is there evidence of a covert brain infarct in "
        "this slice? Answer with a single word: Yes or No."
    ),
    "wmd": (
        "You are a radiologist. This is a 2D axial MRI slice of a human brain "
        "({modality}-weighted). Is there evidence of white matter disease in "
        "this slice? Answer with a single word: Yes or No."
    ),
}

PROMPT_TEMPLATE = PROMPTS["alzheimers"]


def assert_caches_in_repo(repo: Path) -> None:
    repo = repo.resolve()
    for k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"):
        v = os.environ.get(k, "")
        if not v or not Path(v).resolve().is_relative_to(repo):
            raise RuntimeError(f"{k}={v!r} not inside {repo}. Source scripts/activate.sh.")


def load_volume(npz_path: Path) -> np.ndarray:
    """Returns (D, C, H, W) float32 — channels are T1, T2."""
    arr = np.load(npz_path)["arr_0"]  # (C, H, W, D)
    return arr.transpose(3, 0, 1, 2).astype(np.float32)


def central_slice_indices(n: int, fraction: float) -> list[int]:
    half = int(round(n * max(min(fraction, 1.0), 0.0) / 2))
    mid = n // 2
    return list(range(max(0, mid - half), min(n, mid + half))) or [mid]


def slice_to_pil(slice_2d: np.ndarray, clip_lo: float, clip_hi: float) -> Image.Image:
    img = np.clip(slice_2d, clip_lo, clip_hi)
    img = (img - clip_lo) / max(clip_hi - clip_lo, 1e-6)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = np.rot90(img, k=1)
    return Image.fromarray(np.stack([img] * 3, axis=-1), mode="RGB")


def percentile_bounds(vol_mod: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(vol_mod, [1.0, 99.0])
    return float(lo), float(max(hi, lo + 1e-6))


def resolve_yes_no_ids(tokenizer) -> tuple[int, int, dict]:
    def find(variants):
        for v in variants:
            ids = tokenizer(v, add_special_tokens=False).input_ids
            if len(ids) == 1:
                return v, ids[0]
        raise RuntimeError(f"No single-token variant in {variants}")
    y_tok, y_id = find(["Yes", " Yes"])
    n_tok, n_id = find(["No", " No"])
    return y_id, n_id, {"yes": y_tok, "no": n_tok, "yes_id": y_id, "no_id": n_id}


def aggregate(probs: list[float], top_k: int) -> dict:
    if not probs:
        return {"max": float("nan"), "topk_mean": float("nan"), "mean": float("nan")}
    arr = np.asarray(probs)
    k = min(top_k, arr.size)
    return {"max": float(arr.max()),
            "topk_mean": float(np.sort(arr)[-k:].mean()),
            "mean": float(arr.mean())}


def load_labels(numpy_dir: Path, labels_csv: Path | None) -> pd.DataFrame:
    """OASIS-3 schema: Subject, MR ID, Alzheimer's. Filter to rows with NPZ on disk."""
    csv = Path(labels_csv) if labels_csv else Path(numpy_dir) / "labels.csv"
    if not csv.exists():
        raise FileNotFoundError(f"labels CSV not found: {csv}")
    df = pd.read_csv(csv)
    need = {"Subject", "MR ID", "Alzheimer's"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"labels CSV missing columns: {missing}")
    df = df.copy()
    df["npz_path"] = df["MR ID"].apply(lambda m: Path(numpy_dir) / f"{m}.npz")
    df = df[df["npz_path"].apply(Path.exists)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No subjects in {csv} have a matching .npz in {numpy_dir}")
    return df


_KPSC_TASK_COL = {"cbi": "idCBI", "wmd": "idWMD"}


def load_kpsc_labels(numpy_dir: Path, labels_csv: Path | None, task: str) -> pd.DataFrame:
    """KPSC schema: path, SiteID, idCBI, idWMD. Returns df with columns
    (study_id, label, npz_path) for the requested task, filtered to existing NPZ files."""
    if task not in _KPSC_TASK_COL:
        raise ValueError(f"task must be one of {list(_KPSC_TASK_COL)}, got {task!r}")
    csv = Path(labels_csv) if labels_csv else Path(numpy_dir) / "labels.csv"
    if not csv.exists():
        raise FileNotFoundError(f"labels CSV not found: {csv}")
    df = pd.read_csv(csv)
    label_col = _KPSC_TASK_COL[task]
    need = {"path", label_col}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"labels CSV missing columns: {missing}")
    df = df.copy()
    df["npz_path"] = df["path"].apply(Path)
    df["study_id"] = df["npz_path"].apply(lambda p: p.stem)
    df["label"] = df[label_col].astype(int)
    df = df[df["npz_path"].apply(Path.exists)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No KPSC rows in {csv} have a matching .npz on disk")
    keep = ["study_id", "label", "npz_path"]
    if "SiteID" in df.columns:
        keep.append("SiteID")
    return df[keep].copy()


def build_chat_inputs(processor, batch: list[tuple], prompt_template: str):
    """batch: list of (PIL, modality). Returns processor(text, images) dict on CPU."""
    texts = [processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt_template.format(modality=m)}]}],
        tokenize=False, add_generation_prompt=True)
        for img, m in batch]
    return processor(text=texts, images=[img for img, _ in batch],
                     padding=True, return_tensors="pt")


def score_batch(model, processor, yes_id, no_id, batch, prompt_template=PROMPT_TEMPLATE):
    """Deterministic Yes/No logit softmax. Returns list of P(Yes)."""
    device = next(model.parameters()).device
    inputs = build_chat_inputs(processor, batch, prompt_template).to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits
    last = logits[torch.arange(logits.size(0), device=logits.device),
                  inputs["attention_mask"].sum(1) - 1]
    yn = torch.stack([last[:, yes_id], last[:, no_id]], dim=-1).float()
    return F.softmax(yn, dim=-1)[:, 0].cpu().tolist()


def sample_answers(model, processor, batch, n_samples, temperature,
                   prompt_template=PROMPT_TEMPLATE, max_new_tokens=1):
    """Run model.generate n_samples times per batch element; return list[list[str]]
    (outer: per batch item, inner: n_samples decoded strings)."""
    if n_samples <= 0:
        return [[] for _ in batch]
    device = next(model.parameters()).device
    inputs = build_chat_inputs(processor, batch, prompt_template).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    rows = [[] for _ in batch]
    with torch.inference_mode():
        for _ in range(n_samples):
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            )
            new_ids = out[:, prompt_len:]
            decoded = processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            for i, txt in enumerate(decoded):
                rows[i].append(txt.strip())
    return rows


def yes_count_from_answers(answers: list[str]) -> int:
    """Count answers whose first non-empty token starts with 'y' (case-insensitive)."""
    n = 0
    for a in answers:
        t = (a or "").strip()
        if t and t[0].lower() == "y":
            n += 1
    return n


def metrics_for(labels, scores) -> dict:
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
