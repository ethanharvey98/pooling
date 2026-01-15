#!/bin/bash
#SBATCH --array=0-2%3
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=64g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=120:00:00

source ~/.bashrc
conda activate jupyter-env

# Define an array of commands
experiments=(
    "python ../src/encode_rsna_full.py --encoded_dir='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16/seed=1001' --encoder='ViT-B/16' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --seed=1001"
    "python ../src/encode_rsna_full.py --encoded_dir='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16/seed=2001' --encoder='ViT-B/16' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --seed=2001"
    "python ../src/encode_rsna_full.py --encoded_dir='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16/seed=3001' --encoder='ViT-B/16' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --seed=3001"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
