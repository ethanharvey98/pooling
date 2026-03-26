#!/bin/bash
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/log_%j.err
#SBATCH --mem=8g
#SBATCH --ntasks=1
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=4:00:00

source ~/.bashrc
conda activate jupyter-env

pip install gdown

TARGET_DIR="/cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14"
mkdir -p "${TARGET_DIR}"

# Download P-MIL THUMOS14 features and annotations from Google Drive
gdown --folder 1RFNa0bEz9adEDB4xmVXPYfCGB-D_Y1wR -O "${TARGET_DIR}"

echo "Downloaded to ${TARGET_DIR}:"
ls -lh "${TARGET_DIR}"

conda deactivate
