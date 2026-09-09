#!/bin/bash
#SBATCH --job-name=ich_gp_guid
#SBATCH --array=1-6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --time=08:00:00
#SBATCH --error=/cluster/tufts/hugheslab/zmou01/slurmlog/err/log_%A_%a.err
#SBATCH --output=/cluster/tufts/hugheslab/zmou01/slurmlog/out/log_%A_%a.out

source ~/.bashrc
conda activate /cluster/tufts/hugheslab/zmou01/conda_envs/neuroimg_gpu

cd /cluster/home/zmou01/pooling

EXPDIR="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_GP_guidance_embedding_level=True"
DATA="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16"

# criterion=GuidedL1 (normal / GP guidance), alpha=1e-4, beta(lambda)=1.0, embedding-level
# Final selections from notebooks/rsna_ich.ipynb cell 23:
#   ABMIL    seeds 1001/2001/3001 -> lr 0.01 / 0.01 / 0.001
#   TransMIL seeds 1001/2001/3001 -> lr 0.001 / 0.001 / 0.001
experiments=(
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="GuidedL1" --beta=1.0 --dataset_dir="'"$DATA"'/seed=1001" --epochs=1000 --embedding_level --experiments_dir="'"$EXPDIR"'" --lr=0.01  --model_name="alpha=0.0001_criterion=GuidedL1_lr=0.01_pooling=ABMIL_seed=1001"  --pooling="ABMIL"    --save --seed=1001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="GuidedL1" --beta=1.0 --dataset_dir="'"$DATA"'/seed=2001" --epochs=1000 --embedding_level --experiments_dir="'"$EXPDIR"'" --lr=0.01  --model_name="alpha=0.0001_criterion=GuidedL1_lr=0.01_pooling=ABMIL_seed=2001"  --pooling="ABMIL"    --save --seed=2001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="GuidedL1" --beta=1.0 --dataset_dir="'"$DATA"'/seed=3001" --epochs=1000 --embedding_level --experiments_dir="'"$EXPDIR"'" --lr=0.001 --model_name="alpha=0.0001_criterion=GuidedL1_lr=0.001_pooling=ABMIL_seed=3001" --pooling="ABMIL"    --save --seed=3001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="GuidedL1" --beta=1.0 --dataset_dir="'"$DATA"'/seed=1001" --epochs=1000 --embedding_level --experiments_dir="'"$EXPDIR"'" --lr=0.001 --model_name="alpha=0.0001_criterion=GuidedL1_lr=0.001_pooling=TransMIL_seed=1001" --pooling="TransMIL" --save --seed=1001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="GuidedL1" --beta=1.0 --dataset_dir="'"$DATA"'/seed=2001" --epochs=1000 --embedding_level --experiments_dir="'"$EXPDIR"'" --lr=0.001 --model_name="alpha=0.0001_criterion=GuidedL1_lr=0.001_pooling=TransMIL_seed=2001" --pooling="TransMIL" --save --seed=2001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="GuidedL1" --beta=1.0 --dataset_dir="'"$DATA"'/seed=3001" --epochs=1000 --embedding_level --experiments_dir="'"$EXPDIR"'" --lr=0.001 --model_name="alpha=0.0001_criterion=GuidedL1_lr=0.001_pooling=TransMIL_seed=3001" --pooling="TransMIL" --save --seed=3001 --weight_decay=0.0'
)

eval "${experiments[$SLURM_ARRAY_TASK_ID - 1]}"
