#!/bin/bash
#SBATCH --job-name=ng_underflow_check
#SBATCH --partition=gpu,hugheslab
#SBATCH --qos=preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/tufts/hugheslab/zmou01/slurmlog/out/ng_underflow_%j.out
#SBATCH --error=/cluster/tufts/hugheslab/zmou01/slurmlog/err/ng_underflow_%j.err

source ~/.bashrc
conda activate /cluster/tufts/hugheslab/zmou01/conda_envs/neuroimg_gpu
cd /cluster/home/zmou01/pooling
python scripts/check_ng_underflow_training.py
