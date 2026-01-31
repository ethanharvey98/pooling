#!/usr/bin/env python3
"""
Generate experiment commands for timing sweep.

Usage:
    python scripts/generate_timing_commands.py
"""

methods = ['Max', 'Mean', 'ABMIL', 'TransMIL', 'SmAP']
seeds = [1001, 2001, 3001]
alphas = [0.0, 1.0]
lrs = [0.01, 0.0001]

experiments = []

for method in methods:
    for seed in seeds:
        for alpha in alphas:
            for lr in lrs:
                cmd = f'    "python ../src/oasis-3-timing.py --alpha={alpha} --batch_size=64 --criterion=\'L1\' --dataset_dir=\'/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16/seed={seed}\' --embedding_level --epochs=1000 --experiments_dir=\'/cluster/tufts/hugheslab/dloevl01/pooling/experiments/RSNA/timing\' --lr={lr} --model_name=\'alpha={alpha}_lr={lr}_pooling={method}_seed={seed}_timing\' --pooling=\'{method}\' --save --seed={seed} --weight_decay=0.0"'
                experiments.append(cmd)

print("experiments=(")
for exp in experiments:
    print(exp)
print(")")

print(f"\n# Total experiments: {len(experiments)}")
