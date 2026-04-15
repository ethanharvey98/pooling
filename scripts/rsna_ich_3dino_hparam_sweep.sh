#!/bin/bash
#SBATCH --array=0-95%8
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate jupyter-env

# Hyperparameter sweep for RSNA ICH 3DINO experiments (embedding-level only)
# Learning rates: 0.1, 0.01, 0.001, 0.0001 (4 values)
# Alpha: 1.0, 0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06, 0.0 (8 values)
# Seeds: 1001, 2001, 3001 (3 values)
# Only Mean pooling (single embedding per volume)
# Total: 4 * 8 * 3 = 96 experiments

lrs=(0.1 0.01 0.001 0.0001)
alphas=(1.0 0.1 0.01 0.001 0.0001 1e-05 1e-06 0.0)
seeds=(1001 2001 3001)

task_id=$SLURM_ARRAY_TASK_ID

# idx = lr_idx * 24 + alpha_idx * 3 + seed_idx
lr_idx=$((task_id / 24))
remainder=$((task_id % 24))
alpha_idx=$((remainder / 3))
seed_idx=$((remainder % 3))

lr=${lrs[$lr_idx]}
alpha=${alphas[$alpha_idx]}
seed=${seeds[$seed_idx]}

dataset_dir="/cluster/tufts/hugheslab/dloevl01/encoded_RSNA_ICH/3DINO_ViT_concat/seed=${seed}"
experiments_dir="/cluster/tufts/hugheslab/dloevl01/DINO3_Experiments/pooling/experiments/RSNA_ICH_3DINO_embedding_level=True"

model_name="alpha=${alpha}_criterion=L1_lr=${lr}_pooling=Mean_seed=${seed}"

echo "Running: lr=${lr}, alpha=${alpha}, seed=${seed}"

python ../src/oasis-3.py \
    --alpha=${alpha} \
    --batch_size=64 \
    --criterion='L1' \
    --dataset_dir="${dataset_dir}" \
    --embedding_level \
    --epochs=1000 \
    --experiments_dir="${experiments_dir}" \
    --lr=${lr} \
    --model_name="${model_name}" \
    --pooling='Mean' \
    --save \
    --seed=${seed} \
    --weight_decay=0.0

conda deactivate
