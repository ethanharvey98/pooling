#!/bin/bash
#SBATCH --array=0-2
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/log_%j.err
#SBATCH --mem=16g
#SBATCH --ntasks=1
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=1:00:00

source ~/.bashrc
conda activate jupyter-env

seeds=(1001 2001 3001)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

python ../src/convert_thumos14.py \
    --data_dir='/cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14' \
    --output_dir='/cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14_encoded' \
    --mode=binary \
    --seed=${seed}

conda deactivate
