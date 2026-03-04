#!/bin/bash
#SBATCH --array=0-8%4
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=32g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=168:00:00

source ~/.bashrc
conda activate l3d_2024f_cuda12_1

# Define an array of commands
experiments=(
    "python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_RSNA/Qwen2.5-VL/seed=1001' --encoder='Qwen2.5-VL' --model_name='Qwen/Qwen2.5-VL-7B-Instruct' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --seed=1001"
    "python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_RSNA/Qwen2.5-VL/seed=2001' --encoder='Qwen2.5-VL' --model_name='Qwen/Qwen2.5-VL-7B-Instruct' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --seed=2001"
    "python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_RSNA/Qwen2.5-VL/seed=3001' --encoder='Qwen2.5-VL' --model_name='Qwen/Qwen2.5-VL-7B-Instruct' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --seed=3001"
    "python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_RSNA/QoQ-Med-VL/seed=1001' --encoder='Qwen2.5-VL' --model_name='ddvd233/QoQ-Med-VL-7B' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --seed=1001"
    "python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_RSNA/QoQ-Med-VL/seed=2001' --encoder='Qwen2.5-VL' --model_name='ddvd233/QoQ-Med-VL-7B' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --seed=2001"
    "python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_RSNA/QoQ-Med-VL/seed=3001' --encoder='Qwen2.5-VL' --model_name='ddvd233/QoQ-Med-VL-7B' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --seed=3001"
    "python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_RSNA/ViT_B_16/seed=1001' --encoder='ViT-B/16' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --seed=1001"
    "python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_RSNA/ViT_B_16/seed=2001' --encoder='ViT-B/16' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --seed=2001"
    "python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_RSNA/ViT_B_16/seed=3001' --encoder='ViT-B/16' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --seed=3001"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
