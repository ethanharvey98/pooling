#!/bin/bash
#SBATCH --job-name=ceil_semi_speedtest
#SBATCH --partition=gpu,hugheslab
#SBATCH --qos=preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --time=00:15:00
#SBATCH --output=/cluster/tufts/hugheslab/zmou01/slurmlog/out/ceil_speedtest_%j.out
#SBATCH --error=/cluster/tufts/hugheslab/zmou01/slurmlog/err/ceil_speedtest_%j.err

source ~/.bashrc
conda activate /cluster/tufts/hugheslab/zmou01/conda_envs/neuroimg_gpu
cd /cluster/home/zmou01/pooling
python scripts/train_ceiling_semi.py --alpha=0.0001 --lr=0.01 --seed=1001 --epochs=30 --model_name=gpu_speedtest
