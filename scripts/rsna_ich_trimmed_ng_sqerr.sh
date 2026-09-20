#!/bin/bash
#SBATCH --job-name=ich_ng_sqerr
#SBATCH --array=0-5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --time=12:00:00
#SBATCH --error=/cluster/home/zmou01/slurmlog/err/log_%A_%a.err
#SBATCH --output=/cluster/home/zmou01/slurmlog/out/log_%A_%a.out

source ~/.bashrc
conda activate /cluster/tufts/hugheslab/zmou01/conda_envs/neuroimg_gpu

cd /cluster/home/zmou01/pooling

# Same as rsna_ich_trimmed_ng_fkl.sh but divergence=squared error (the
# default) instead of forward kl -- fills out beta in {0.1, 10} x 3 seeds
# (beta=1.0 x 3 seeds already exists: normal_guidance_abmil_alpha=0.0001_
# beta=1.0_lr=0.01[_seed{2001,3001}].pth, the original established
# trimmed squared-error baseline), matching the same beta grid as the
# forward-kl sweep for a fair side-by-side comparison of both divergences.
EXPDIR="/cluster/tufts/hugheslab/zmou01/models/stage2_experiments"
DATA="/cluster/tufts/hugheslab/zmou01/datasets/encoded_RSNA_ICH_trimmed/ViT_B_16"

BETAS=(0.1 0.1 0.1 10 10 10)
SEEDS=(1001 2001 3001 1001 2001 3001)

BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python -u src/oasis-3.py \
    --alpha=0.0001 \
    --batch_size=64 \
    --criterion="GuidedL1" \
    --divergence="squared error" \
    --beta=${BETA} \
    --dataset_dir="${DATA}/seed=${SEED}" \
    --epochs=1000 \
    --embedding_level \
    --experiments_dir="${EXPDIR}" \
    --lr=0.01 \
    --model_name="normal_guidance_abmil_alpha=0.0001_beta=${BETA}_lr=0.01_seed=${SEED}" \
    --pooling="ABMIL" \
    --save \
    --seed=${SEED} \
    --weight_decay=0.0
