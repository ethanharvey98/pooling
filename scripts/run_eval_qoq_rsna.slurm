#!/bin/bash
#SBATCH --job-name=qoq_rsna
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# 5-way array job. Each task processes 1/5 of the RSNA studies.
# Usage (from repo root):
#   sbatch scripts/run_eval_qoq_rsna.slurm
# After all 5 finish, combine with:
#   python scripts/combine_rsna_shards.py \
#       --shards outputs/qoq_rsna_<arrayjobid>_shard{0,1,2,3,4} \
#       --out_dir outputs/qoq_rsna_<arrayjobid>_combined

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

mkdir -p logs outputs

# shellcheck disable=SC1091
source "$REPO_DIR/scripts/activate.sh"

echo "[slurm] host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "[slurm] array_job=${SLURM_ARRAY_JOB_ID}  task=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
nvidia-smi
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-5}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="outputs/qoq_rsna_${SLURM_ARRAY_JOB_ID:-$STAMP}_shard${SHARD_ID}"

python scripts/eval_qoq_med_rsna.py \
  --output_dir "$OUT" \
  --shard_id   "$SHARD_ID" \
  --num_shards "$NUM_SHARDS"

echo "[slurm] DONE -> $OUT"
