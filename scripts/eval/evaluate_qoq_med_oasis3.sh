#!/bin/bash
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --mem=32g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/log_%j.out
#SBATCH --partition=gpu
#SBATCH --time=72:00:00

source ~/.bashrc
conda activate /cluster/tufts/hugheslab/dloevl01/DINO3_Experiments/pooling/envs/dino3

python ../scripts/eval/evaluate_qoq_med_oasis3.py \
    --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' \
    --output_dir='/cluster/tufts/hugheslab/dloevl01/DINO3_Experiments/pooling/experiments/qoq_med_oasis3' \
    --resume

conda deactivate
