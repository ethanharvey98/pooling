#!/bin/bash
#SBATCH --job-name=ich_ng_fkl
#SBATCH --array=0-8
#SBATCH --partition=hugheslab
#SBATCH --qos=preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --time=12:00:00
#SBATCH --error=/cluster/home/zmou01/slurmlog/err/log_%A_%a.err
#SBATCH --output=/cluster/home/zmou01/slurmlog/out/log_%A_%a.out

source ~/.bashrc
conda activate /cluster/tufts/hugheslab/zmou01/conda_envs/neuroimg_gpu

cd /cluster/home/zmou01/pooling

# Plain ABMIL+NG (criterion=GuidedL1, not GP-guided at all -- applies the
# self-referential penalty to ALL bags uniformly, positive and negative
# alike, same as the original NG baseline) on the TRIMMED RSNA-ICH dataset
# (not the full/untrimmed one pooling/experiments' own runs used), with
# divergence="forward kl" instead of the default squared error. alpha/lr
# fixed at the already-established "best params" values (0.0001 / 0.01);
# beta (the guidance weight, called lambda_0/lambda in our own reporting
# convention) swept over {0.1, 1, 10} x 3 seeds.
EXPDIR="/cluster/tufts/hugheslab/zmou01/models/stage2_experiments"
DATA="/cluster/tufts/hugheslab/zmou01/datasets/encoded_RSNA_ICH_trimmed/ViT_B_16"

BETAS=(0.1 0.1 0.1 1 1 1 10 10 10)
SEEDS=(1001 2001 3001 1001 2001 3001 1001 2001 3001)

BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python -u src/oasis-3.py \
    --alpha=0.0001 \
    --batch_size=64 \
    --criterion="GuidedL1" \
    --divergence="forward kl" \
    --beta=${BETA} \
    --dataset_dir="${DATA}/seed=${SEED}" \
    --epochs=1000 \
    --embedding_level \
    --experiments_dir="${EXPDIR}" \
    --lr=0.01 \
    --model_name="normal_guidance_abmil_trimmed_fkl_alpha=0.0001_beta=${BETA}_lr=0.01_seed=${SEED}" \
    --pooling="ABMIL" \
    --save \
    --seed=${SEED} \
    --weight_decay=0.0
