#!/bin/bash
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/log_%j.err
#SBATCH --mem=8g
#SBATCH --ntasks=1
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=00:30:00

source ~/.bashrc
conda activate jupyter-env

echo "========================================="
echo "FULL DATASET (21744 scans)"
echo "========================================="
for seed in 1001 2001 3001; do
    echo ""
    echo "--- seed=$seed ---"
    python ../src/check_patient_overlap.py \
        --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' \
        --seed=$seed
done

echo ""
echo "========================================="
echo "SUBSET (1149 scans)"
echo "========================================="
for seed in 1001 2001 3001; do
    echo ""
    echo "--- seed=$seed ---"
    python ../src/check_patient_overlap.py \
        --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' \
        --subset_csv='/home/denny-loevlie/RSNA_Investigation/rsna_ich_subset_1149_complete.csv' \
        --seed=$seed
done

conda deactivate
