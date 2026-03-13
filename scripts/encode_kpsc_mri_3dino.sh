#!/bin/bash
#SBATCH --array=0-4
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate /cluster/tufts/hugheslab/dloevl01/DINO3_Experiments/pooling/envs/dino3

WEIGHTS="/cluster/tufts/hugheslab/dloevl01/DINO3_Experiments/pooling/envs/dino3/3dino_vit_weights.pth"
NUMPY_DIR="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy"
ENC_BASE="/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/3DINO_ViT_concat_T1T2"

# 5 site-based splits
experiments=(
    "python ../src/encode_kpsc_3dino.py --encoded_dir=${ENC_BASE}/test_site_ids=9_train_site_ids=1_2_3_4_6_7_8_10_11_val_site_ids=5 --numpy_dir=${NUMPY_DIR} --pretrained_weights=${WEIGHTS} --n_last_blocks=4 --avgpool --test_site_ids 9 --train_site_ids 1 2 3 4 6 7 8 10 11 --val_site_ids 5"
    "python ../src/encode_kpsc_3dino.py --encoded_dir=${ENC_BASE}/test_site_ids=1_4_7_10_11_train_site_ids=2_3_5_6_8_val_site_ids=9 --numpy_dir=${NUMPY_DIR} --pretrained_weights=${WEIGHTS} --n_last_blocks=4 --avgpool --test_site_ids 1 4 7 10 11 --train_site_ids 2 3 5 6 8 --val_site_ids 9"
    "python ../src/encode_kpsc_3dino.py --encoded_dir=${ENC_BASE}/test_site_ids=2_6_train_site_ids=3_5_8_9_val_site_ids=1_4_7_10_11 --numpy_dir=${NUMPY_DIR} --pretrained_weights=${WEIGHTS} --n_last_blocks=4 --avgpool --test_site_ids 2 6 --train_site_ids 3 5 8 9 --val_site_ids 1 4 7 10 11"
    "python ../src/encode_kpsc_3dino.py --encoded_dir=${ENC_BASE}/test_site_ids=3_8_train_site_ids=1_4_5_7_9_10_11_val_site_ids=2_6 --numpy_dir=${NUMPY_DIR} --pretrained_weights=${WEIGHTS} --n_last_blocks=4 --avgpool --test_site_ids 3 8 --train_site_ids 1 4 5 7 9 10 11 --val_site_ids 2 6"
    "python ../src/encode_kpsc_3dino.py --encoded_dir=${ENC_BASE}/test_site_ids=5_train_site_ids=1_2_4_6_7_9_10_11_val_site_ids=3_8 --numpy_dir=${NUMPY_DIR} --pretrained_weights=${WEIGHTS} --n_last_blocks=4 --avgpool --test_site_ids 5 --train_site_ids 1 2 4 6 7 9 10 11 --val_site_ids 3 8"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
