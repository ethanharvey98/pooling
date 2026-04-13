#!/bin/bash
#SBATCH --array=0-2
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate /cluster/tufts/hugheslab/dloevl01/DINO3_Experiments/pooling/envs/dino3

seeds=(1001 2001 3001)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

WEIGHTS="/cluster/tufts/hugheslab/dloevl01/DINO3_Experiments/pooling/envs/dino3/3dino_vit_weights.pth"

python ../src/encode_rsna_3dino.py \
    --encoded_dir="/cluster/tufts/hugheslab/dloevl01/encoded_RSNA_ICH/3DINO_ViT_concat/seed=${seed}" \
    --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy' \
    --csv_path='/cluster/tufts/hugheslab/datasets/RSNA_ICH/subset_labels.csv' \
    --pretrained_weights="${WEIGHTS}" \
    --seed=${seed} \
    --n_last_blocks=4 \
    --avgpool

conda deactivate
