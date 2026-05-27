#!/bin/bash
#SBATCH --job-name=qoq_kpsc800
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage:
#   cd <repo>/pooling
#   sbatch scripts/run_eval_qoq_kpsc800.slurm cbi
#   sbatch scripts/run_eval_qoq_kpsc800.slurm wmd
#
# Optional second positional arg overrides --n_samples (default 5). Useful for
# a first timed pass at --n_samples 1 to estimate wall-clock before scaling up:
#   sbatch scripts/run_eval_qoq_kpsc800.slurm cbi 1
#
# Before submitting the first time, run once on the login node:
#   bash scripts/setup_env_hpc.sh
#   source scripts/activate.sh
#   huggingface-cli login            # if not already logged in
#   bash scripts/setup_env_hpc.sh    # re-run to pre-pull model into repo cache

set -euo pipefail

TASK="${1:?Usage: sbatch $0 {cbi|wmd} [n_samples]}"
N_SAMPLES="${2:-5}"
case "$TASK" in
  cbi|wmd) ;;
  *) echo "TASK must be 'cbi' or 'wmd', got '$TASK'" >&2; exit 2 ;;
esac

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
OUT="outputs/qoq_kpsc800_${TASK}_${SLURM_JOB_ID:-$STAMP}"

python scripts/eval_qoq_med_kpsc800.py \
  --task "$TASK" \
  --output_dir "$OUT" \
  --batch_size 8 \
  --dtype bfloat16 \
  --central_fraction 0.4 \
  --top_k 5 \
  --n_samples "$N_SAMPLES" \
  --temperature 0.7

echo "[slurm] DONE -> $OUT"
