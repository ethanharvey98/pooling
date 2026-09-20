#!/bin/bash
#SBATCH --job-name=ich_cia
#SBATCH --array=1-12%12
#SBATCH --partition=gpu,hugheslab
#SBATCH --qos=preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --time=10:00:00
#SBATCH --error=/cluster/tufts/hugheslab/zmou01/slurmlog/err/log_%A_%a.err
#SBATCH --output=/cluster/tufts/hugheslab/zmou01/slurmlog/out/log_%A_%a.out

source ~/.bashrc
conda activate /cluster/tufts/hugheslab/zmou01/conda_envs/neuroimg_gpu
cd /cluster/home/zmou01/pooling

# CIA-MIL (counterfactual intervention, Chraki et al. MIDL 2026) on RSNA ICH, ABMIL.
# alpha=1e-4 (L1), beta_effect in {0.2,1.0}, lr in {0.01,0.001}, 3 seeds, 600 epochs.
experiments=(
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="CIA" --beta_effect=0.2 --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=1001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_CIA_embedding_level=True" --lr=0.01 --model_name="alpha=0.0001_criterion=CIA_beta_effect=0.2_lr=0.01_pooling=ABMIL_seed=1001" --pooling="ABMIL" --save --seed=1001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="CIA" --beta_effect=0.2 --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=2001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_CIA_embedding_level=True" --lr=0.01 --model_name="alpha=0.0001_criterion=CIA_beta_effect=0.2_lr=0.01_pooling=ABMIL_seed=2001" --pooling="ABMIL" --save --seed=2001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="CIA" --beta_effect=0.2 --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=3001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_CIA_embedding_level=True" --lr=0.01 --model_name="alpha=0.0001_criterion=CIA_beta_effect=0.2_lr=0.01_pooling=ABMIL_seed=3001" --pooling="ABMIL" --save --seed=3001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="CIA" --beta_effect=0.2 --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=1001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_CIA_embedding_level=True" --lr=0.001 --model_name="alpha=0.0001_criterion=CIA_beta_effect=0.2_lr=0.001_pooling=ABMIL_seed=1001" --pooling="ABMIL" --save --seed=1001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="CIA" --beta_effect=0.2 --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=2001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_CIA_embedding_level=True" --lr=0.001 --model_name="alpha=0.0001_criterion=CIA_beta_effect=0.2_lr=0.001_pooling=ABMIL_seed=2001" --pooling="ABMIL" --save --seed=2001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="CIA" --beta_effect=0.2 --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=3001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_CIA_embedding_level=True" --lr=0.001 --model_name="alpha=0.0001_criterion=CIA_beta_effect=0.2_lr=0.001_pooling=ABMIL_seed=3001" --pooling="ABMIL" --save --seed=3001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="CIA" --beta_effect=1.0 --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=1001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_CIA_embedding_level=True" --lr=0.01 --model_name="alpha=0.0001_criterion=CIA_beta_effect=1.0_lr=0.01_pooling=ABMIL_seed=1001" --pooling="ABMIL" --save --seed=1001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="CIA" --beta_effect=1.0 --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=2001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_CIA_embedding_level=True" --lr=0.01 --model_name="alpha=0.0001_criterion=CIA_beta_effect=1.0_lr=0.01_pooling=ABMIL_seed=2001" --pooling="ABMIL" --save --seed=2001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="CIA" --beta_effect=1.0 --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=3001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_CIA_embedding_level=True" --lr=0.01 --model_name="alpha=0.0001_criterion=CIA_beta_effect=1.0_lr=0.01_pooling=ABMIL_seed=3001" --pooling="ABMIL" --save --seed=3001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="CIA" --beta_effect=1.0 --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=1001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_CIA_embedding_level=True" --lr=0.001 --model_name="alpha=0.0001_criterion=CIA_beta_effect=1.0_lr=0.001_pooling=ABMIL_seed=1001" --pooling="ABMIL" --save --seed=1001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="CIA" --beta_effect=1.0 --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=2001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_CIA_embedding_level=True" --lr=0.001 --model_name="alpha=0.0001_criterion=CIA_beta_effect=1.0_lr=0.001_pooling=ABMIL_seed=2001" --pooling="ABMIL" --save --seed=2001 --weight_decay=0.0'
    'python src/oasis-3.py --alpha=0.0001 --batch_size=64 --criterion="CIA" --beta_effect=1.0 --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=3001" --epochs=600 --embedding_level --experiments_dir="/cluster/home/zmou01/pooling/experiments/RSNA_ICH_full_dataset_CIA_embedding_level=True" --lr=0.001 --model_name="alpha=0.0001_criterion=CIA_beta_effect=1.0_lr=0.001_pooling=ABMIL_seed=3001" --pooling="ABMIL" --save --seed=3001 --weight_decay=0.0'
)

eval "${experiments[$SLURM_ARRAY_TASK_ID - 1]}"
