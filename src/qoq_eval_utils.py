"""Helpers for QoQ-Med-VL-7B zero-shot eval on OASIS-3."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

PROMPT_TEMPLATE = (
    "You are a radiologist. This is a 2D axial MRI slice of a human brain "
    "({modality}-weighted). Is there evidence of Alzheimer's disease in this "
    "slice? Answer with a single word: Yes or No."
)


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
    """Load labels.csv; error if missing. Filter to rows with NPZ on disk."""
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
