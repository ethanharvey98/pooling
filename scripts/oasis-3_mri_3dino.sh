#!/bin/bash
#SBATCH --array=0-2%3
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate /cluster/tufts/hugheslab/dloevl01/DINO3_Experiments/pooling/envs/dino3

experiments=(
    "python ../src/oasis-3.py --batch_size=64 --criterion='ERM' --dataset_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/3DINO_ViT/seed=1001' --embedding_level --epochs=1000 --experiments_dir='/cluster/tufts/hugheslab/dloevl01/DINO3_Experiments/pooling/experiments/OASIS-3_MRI_3DINO' --lr=0.01 --model_name='pooling=Mean_lr=0.01_seed=1001' --pooling='Mean' --save --seed=1001"
    "python ../src/oasis-3.py --batch_size=64 --criterion='ERM' --dataset_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/3DINO_ViT/seed=2001' --embedding_level --epochs=1000 --experiments_dir='/cluster/tufts/hugheslab/dloevl01/DINO3_Experiments/pooling/experiments/OASIS-3_MRI_3DINO' --lr=0.01 --model_name='pooling=Mean_lr=0.01_seed=2001' --pooling='Mean' --save --seed=2001"
    "python ../src/oasis-3.py --batch_size=64 --criterion='ERM' --dataset_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/3DINO_ViT/seed=3001' --embedding_level --epochs=1000 --experiments_dir='/cluster/tufts/hugheslab/dloevl01/DINO3_Experiments/pooling/experiments/OASIS-3_MRI_3DINO' --lr=0.01 --model_name='pooling=Mean_lr=0.01_seed=3001' --pooling='Mean' --save --seed=3001"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
