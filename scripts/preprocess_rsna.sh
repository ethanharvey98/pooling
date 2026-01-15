#!/bin/bash
#SBATCH --array=0-9%10
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/log_%j.err
#SBATCH --mem=64g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=120:00:00

source ~/.bashrc
conda activate jupyter-env

# Define an array of commands
experiments=(
    "python ../src/preprocess_rsna_full.py --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --start=0 --stop=2174"
    "python ../src/preprocess_rsna_full.py --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --start=2174 --stop=4348"
    "python ../src/preprocess_rsna_full.py --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --start=4348 --stop=6522"
    "python ../src/preprocess_rsna_full.py --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --start=6522 --stop=8696"
    "python ../src/preprocess_rsna_full.py --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --start=8696 --stop=10870"
    "python ../src/preprocess_rsna_full.py --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --start=10870 --stop=13044"
    "python ../src/preprocess_rsna_full.py --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --start=13044 --stop=15218"
    "python ../src/preprocess_rsna_full.py --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --start=15218 --stop=17392"
    "python ../src/preprocess_rsna_full.py --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --start=17392 --stop=19566"
    "python ../src/preprocess_rsna_full.py --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --start=19566 --stop=21744"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
