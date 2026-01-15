#!/bin/bash
#SBATCH --array=0-191%8
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=120:00:00

source ~/.bashrc
conda activate jupyter-env

# Define an array of commands
# Embedding-level only: 8 alphas x 4 lrs x 2 poolings (mean, attention) x 3 seeds = 192 experiments
experiments=(
    # Paste output from notebook cell here
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
