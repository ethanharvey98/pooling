#!/bin/bash
#SBATCH --array=0-10%4
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=168:00:00

source ~/.bashrc
conda activate brain-scan-classifiers

# Define an array of commands
experiments=(
    "python ../src/encode_oasis-3.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_CT/MedSAM/seed=1001' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_CT_numpy' --seed=1001"
    "python ../src/encode_oasis-3.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_CT/MedSAM/seed=2001' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_CT_numpy' --seed=2001"
    "python ../src/encode_oasis-3.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_CT/MedSAM/seed=3001' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_CT_numpy' --seed=3001"
    "python ../src/encode_oasis-3.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/MedSAM/seed=1001' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --seed=1001"
    "python ../src/encode_oasis-3.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/MedSAM/seed=2001' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --seed=2001"
    "python ../src/encode_oasis-3.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/MedSAM/seed=3001' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --seed=3001"
    "python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/MedSAM/test_site_ids=9_train_site_ids=1_2_3_4_6_7_8_10_11_val_site_ids=5' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 9 --train_site_ids 1 2 3 4 6 7 8 10 11 --val_site_ids 5"
    "python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/MedSAM/test_site_ids=1_4_7_10_11_train_site_ids=2_3_5_6_8_val_site_ids=9' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 1 4 7 10 11 --train_site_ids 2 3 5 6 8 --val_site_ids 9"
    "python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/MedSAM/test_site_ids=2_6_train_site_ids=3_5_8_9_val_site_ids=1_4_7_10_11' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 2 6 --train_site_ids 3 5 8 9 --val_site_ids 1 4 7 10 11"
    "python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/MedSAM/test_site_ids=3_8_train_site_ids=1_4_5_7_9_10_11_val_site_ids=2_6' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 3 8 --train_site_ids 1 4 5 7 9 10 11 --val_site_ids 2 6"
    "python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/MedSAM/test_site_ids=5_train_site_ids=1_2_4_6_7_9_10_11_val_site_ids=3_8' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 5 --train_site_ids 1 2 4 6 7 9 10 11 --val_site_ids 3 8"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate