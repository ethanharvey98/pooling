#!/bin/bash
#SBATCH --job-name=entropy_head
#SBATCH --partition=gpu
#SBATCH --qos=preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/tufts/hugheslab/zmou01/slurmlog/out/entropy_head_%j.out
#SBATCH --error=/cluster/tufts/hugheslab/zmou01/slurmlog/err/entropy_head_%j.err

source ~/.bashrc
conda activate /cluster/tufts/hugheslab/zmou01/conda_envs/neuroimg_gpu
cd /cluster/home/zmou01/pooling
python scripts/eval_entropy_head.py
