#!/usr/bin/env bash
# One-shot HPC bootstrap: installs Miniforge, creates conda env, installs deps,
# pre-downloads QoQ-Med-VL-7B — all INSIDE the repo directory. Safe to re-run.
#
# Usage (on Tufts HPC login or compute node):
#   cd /path/to/pooling
#   bash scripts/setup_env_hpc.sh
#
# After this, source scripts/activate.sh and run the eval.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR
cd "$REPO_DIR"

echo "[setup] REPO_DIR=$REPO_DIR"

# Redirect ALL caches/downloads into the repo BEFORE any tool can default to ~/
export HF_HOME="$REPO_DIR/hf_cache"
export HF_HUB_CACHE="$REPO_DIR/hf_cache/hub"
export TRANSFORMERS_CACHE="$REPO_DIR/hf_cache"
export HF_DATASETS_CACHE="$REPO_DIR/hf_cache/datasets"
export PIP_CACHE_DIR="$REPO_DIR/pip_cache"
export TORCH_HOME="$REPO_DIR/torch_cache"
export XDG_CACHE_HOME="$REPO_DIR/.cache"
export CONDA_PKGS_DIRS="$REPO_DIR/conda_pkgs"
export CONDA_ENVS_PATH="$REPO_DIR/envs"
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
         "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" \
         "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" \
         "$REPO_DIR/logs" "$REPO_DIR/outputs"

# 1. Miniforge into repo dir
if [ ! -x "$REPO_DIR/miniforge3/bin/conda" ]; then
  echo "[setup] downloading Miniforge into $REPO_DIR/miniforge3"
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64)  INSTALLER="Miniforge3-Linux-x86_64.sh" ;;
    aarch64) INSTALLER="Miniforge3-Linux-aarch64.sh" ;;
    *) echo "unsupported arch: $ARCH" >&2; exit 1 ;;
  esac
  curl -L -o "$REPO_DIR/miniforge.sh" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/$INSTALLER"
  bash "$REPO_DIR/miniforge.sh" -b -p "$REPO_DIR/miniforge3"
  rm "$REPO_DIR/miniforge.sh"
else
  echo "[setup] Miniforge already present"
fi

# 2. Init conda for this shell
# shellcheck disable=SC1091
source "$REPO_DIR/miniforge3/etc/profile.d/conda.sh"

# 3. Env at repo/envs/qoq with latest Python
if [ ! -d "$REPO_DIR/envs/qoq" ]; then
  echo "[setup] creating env at $REPO_DIR/envs/qoq (python=3.12)"
  conda create -y -p "$REPO_DIR/envs/qoq" python=3.12 -c conda-forge
else
  echo "[setup] env already exists"
fi
conda activate "$REPO_DIR/envs/qoq"

# 4. Pip installs
pip install --upgrade pip
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install \
  "transformers>=4.49" \
  accelerate \
  qwen-vl-utils \
  pillow \
  "numpy==1.26.*" \
  pandas \
  scikit-learn \
  tqdm \
  "huggingface_hub>=0.24" \
  nibabel

# 5. Pre-pull VLM weights into repo HF cache (confirms auth, forces download location).
#    If HF_TOKEN is in the environment, use it; otherwise assume `huggingface-cli login`
#    was already run and a token exists in $HF_HOME/token.
python - <<'PY'
import os, sys
from pathlib import Path
repo = Path(os.environ["REPO_DIR"] if "REPO_DIR" in os.environ else os.getcwd()).resolve()
for k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"):
    v = Path(os.environ[k]).resolve()
    assert str(v).startswith(str(repo)), f"{k}={v} leaks outside {repo}"
    print(f"[setup] {k} -> {v}")
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download("ddvd233/QoQ-Med-VL-7B")
    print(f"[setup] model cached at: {path}")
except Exception as e:
    print(f"[setup] WARNING: could not pre-pull model ({e.__class__.__name__}: {e})")
    print("[setup]   Run `huggingface-cli login` then re-run this script.")
    sys.exit(0)
PY

echo "[setup] DONE. Next:  source $REPO_DIR/scripts/activate.sh"
