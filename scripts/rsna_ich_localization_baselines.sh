#!/bin/bash
#SBATCH --job-name=ich_baselines
#SBATCH --array=1-12%10
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --time=10:00:00
#SBATCH --error=/cluster/tufts/hugheslab/zmou01/slurmlog/err/log_%A_%a.err
#SBATCH --output=/cluster/tufts/hugheslab/zmou01/slurmlog/out/log_%A_%a.out

source ~/.bashrc
conda activate /cluster/tufts/hugheslab/zmou01/conda_envs/neuroimg_gpu
cd /cluster/home/zmou01/pooling

# RSNA ICH localization baselines, retrained with this pipeline, notebook-selected params.
# SmAP / TransMIL / SmTAP (L1) + best-in-class ceiling (instance-level clf, kernel_size=12). 600 epochs.
experiments=(
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="L1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=1001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_baselines_embedding_level=True" --lr=0.01 --model_name="alpha=0.0001_criterion=L1_lr=0.01_pooling=SmAP_seed=1001" --pooling="SmAP" --save --seed=1001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="L1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=2001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_baselines_embedding_level=True" --lr=0.01 --model_name="alpha=0.0001_criterion=L1_lr=0.01_pooling=SmAP_seed=2001" --pooling="SmAP" --save --seed=2001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="L1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=3001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_baselines_embedding_level=True" --lr=0.01 --model_name="alpha=0.0001_criterion=L1_lr=0.01_pooling=SmAP_seed=3001" --pooling="SmAP" --save --seed=3001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0 --batch_size=64 --criterion="L1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=1001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_baselines_embedding_level=True" --lr=0.01 --model_name="alpha=0.0_criterion=L1_lr=0.01_pooling=TransMIL_seed=1001" --pooling="TransMIL" --save --seed=1001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="L1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=2001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_baselines_embedding_level=True" --lr=0.001 --model_name="alpha=0.0001_criterion=L1_lr=0.001_pooling=TransMIL_seed=2001" --pooling="TransMIL" --save --seed=2001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="L1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=3001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_baselines_embedding_level=True" --lr=0.001 --model_name="alpha=0.0001_criterion=L1_lr=0.001_pooling=TransMIL_seed=3001" --pooling="TransMIL" --save --seed=3001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="L1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=1001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_baselines_embedding_level=True" --lr=0.01 --model_name="alpha=0.0001_criterion=L1_lr=0.01_pooling=SmTAP_seed=1001" --pooling="SmTAP" --save --seed=1001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="L1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=2001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_baselines_embedding_level=True" --lr=0.001 --model_name="alpha=0.0001_criterion=L1_lr=0.001_pooling=SmTAP_seed=2001" --pooling="SmTAP" --save --seed=2001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="L1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=3001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_baselines_embedding_level=True" --lr=0.001 --model_name="alpha=0.0001_criterion=L1_lr=0.001_pooling=SmTAP_seed=3001" --pooling="SmTAP" --save --seed=3001 --weight_decay=0.0'
    'python src/best_possible_instance-level.py --alpha=1e-05 --batch_size=64 --criterion="L1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=1001" --epochs=600 --kernel_size=12 --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_best_possible_instance-level" --lr=0.01 --model_name="alpha=1e-05_criterion=L1_lr=0.01_seed=1001" --save --seed=1001 --weight_decay=0.0'
    'python src/best_possible_instance-level.py --alpha=1e-05 --batch_size=64 --criterion="L1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=2001" --epochs=600 --kernel_size=12 --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_best_possible_instance-level" --lr=0.01 --model_name="alpha=1e-05_criterion=L1_lr=0.01_seed=2001" --save --seed=2001 --weight_decay=0.0'
    'python src/best_possible_instance-level.py --alpha=1e-05 --batch_size=64 --criterion="L1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=3001" --epochs=600 --kernel_size=12 --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_best_possible_instance-level" --lr=0.01 --model_name="alpha=1e-05_criterion=L1_lr=0.01_seed=3001" --save --seed=3001 --weight_decay=0.0'
)

eval "${experiments[$SLURM_ARRAY_TASK_ID - 1]}"
