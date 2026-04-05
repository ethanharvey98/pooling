#!/bin/bash
#SBATCH --array=0-5%6
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=64g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=120:00:00

source ~/.bashrc
conda activate jupyter-env

LABELS_CSV='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv'
SUBSET_CSV='/home/denny-loevlie/RSNA_Investigation/rsna_ich_subset_1149_complete.csv'
SUBSET_NUMPY='/cluster/tufts/hugheslab/dloevl01/datasets/RSNA/RSNA_ICH_numpy'
FULL_NUMPY='/cluster/tufts/hugheslab/datasets/RSNA_numpy'

experiments=(
    # Full dataset (3 seeds)
    "python ../src/encode_rsna_full.py --encoded_dir='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA_april5_official/ViT_B_16/seed=1001' --encoder='ViT-B/16' --labels_csv=${LABELS_CSV} --numpy_dir=${FULL_NUMPY} --seed=1001"
    "python ../src/encode_rsna_full.py --encoded_dir='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA_april5_official/ViT_B_16/seed=2001' --encoder='ViT-B/16' --labels_csv=${LABELS_CSV} --numpy_dir=${FULL_NUMPY} --seed=2001"
    "python ../src/encode_rsna_full.py --encoded_dir='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA_april5_official/ViT_B_16/seed=3001' --encoder='ViT-B/16' --labels_csv=${LABELS_CSV} --numpy_dir=${FULL_NUMPY} --seed=3001"
    # Subset (3 seeds)
    "python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA_ICH_april5_official/ViT_B_16/seed=1001' --encoder='ViT-B/16' --numpy_dir=${SUBSET_NUMPY} --csv_path=${SUBSET_CSV} --labels_csv=${LABELS_CSV} --seed=1001"
    "python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA_ICH_april5_official/ViT_B_16/seed=2001' --encoder='ViT-B/16' --numpy_dir=${SUBSET_NUMPY} --csv_path=${SUBSET_CSV} --labels_csv=${LABELS_CSV} --seed=2001"
    "python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA_ICH_april5_official/ViT_B_16/seed=3001' --encoder='ViT-B/16' --numpy_dir=${SUBSET_NUMPY} --csv_path=${SUBSET_CSV} --labels_csv=${LABELS_CSV} --seed=3001"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
