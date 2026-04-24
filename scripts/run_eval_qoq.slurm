#!/bin/bash
#SBATCH --job-name=qoq_oasis3
#SBATCH --partition=gpu            # TODO: confirm Tufts partition (preempt / hugheslab / gpu)
#SBATCH --gres=gpu:a100:1          # TODO: confirm GPU SKU available (a100 / h100 / rtx6000)
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage:
#   cd <repo>/pooling
#   sbatch scripts/run_eval_qoq.slurm
#
# Before submitting the first time, run once on the login node:
#   bash scripts/setup_env_hpc.sh
#   source scripts/activate.sh
#   huggingface-cli login            # if not already logged in
#   bash scripts/setup_env_hpc.sh    # re-run to pre-pull model into repo cache

set -euo pipefail

# Resolve repo dir relative to this script (so the .slurm file is portable)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs outputs

# shellcheck disable=SC1091
source "$REPO_DIR/scripts/activate.sh"

echo "[slurm] host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
nvidia-smi
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="outputs/qoq_oasis3_${SLURM_JOB_ID:-$STAMP}"

python scripts/eval_qoq_med_oasis3.py \
  --output_dir "$OUT" \
  --batch_size 8 \
  --dtype bfloat16 \
  --central_fraction 0.4 \
  --top_k 5

echo "[slurm] DONE -> $OUT"
