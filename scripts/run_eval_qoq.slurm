#!/bin/bash
#SBATCH --job-name=qoq_oasis3
#SBATCH --partition=hugheslab
#SBATCH --gres=gpu:rtx_a6000:1
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

# SLURM copies this script to a spool dir on the compute node, so we cannot
# resolve paths via BASH_SOURCE. Use SLURM_SUBMIT_DIR instead, set by sbatch
# to the directory from which the job was submitted (expected: repo root).
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
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
