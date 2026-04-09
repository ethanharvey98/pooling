#!/bin/bash
cd "$(dirname "$0")/../src"

PATHS=(
    '/cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=1003_data_seed_train=1001_data_seed_val=1002_n_test=1000_n_train=10000_n_val=2500/alpha=0.01_criterion=L1_lr=0.1_pooling=ABMIL_seed=1001.pt'
    '/cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=2003_data_seed_train=2001_data_seed_val=2002_n_test=1000_n_train=10000_n_val=2500/alpha=0.01_criterion=L1_lr=0.1_pooling=ABMIL_seed=2001.pt'
    '/cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=3003_data_seed_train=3001_data_seed_val=3002_n_test=1000_n_train=10000_n_val=2500/alpha=0.001_criterion=L1_lr=0.001_pooling=ABMIL_seed=3001.pt'
    '/cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=1003_data_seed_train=1001_data_seed_val=1002_n_test=1000_n_train=10000_n_val=2500/alpha=0.001_criterion=L1_lr=0.0001_pooling=TransMIL_seed=1001.pt'
    '/cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=2003_data_seed_train=2001_data_seed_val=2002_n_test=1000_n_train=10000_n_val=2500/alpha=0.001_criterion=L1_lr=0.001_pooling=TransMIL_seed=2001.pt'
    '/cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=3003_data_seed_train=3001_data_seed_val=3002_n_test=1000_n_train=10000_n_val=2500/alpha=0.001_criterion=L1_lr=0.0001_pooling=TransMIL_seed=3001.pt'
    '/cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=1003_data_seed_train=1001_data_seed_val=1002_n_test=1000_n_train=10000_n_val=2500/alpha=0.01_criterion=L1_lr=0.01_pooling=SmAP_seed=1001.pt'
    '/cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=2003_data_seed_train=2001_data_seed_val=2002_n_test=1000_n_train=10000_n_val=2500/alpha=0.01_criterion=L1_lr=0.01_pooling=SmAP_seed=2001.pt'
    '/cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=3003_data_seed_train=3001_data_seed_val=3002_n_test=1000_n_train=10000_n_val=2500/alpha=0.01_criterion=L1_lr=0.001_pooling=SmAP_seed=3001.pt'
)

echo "=== Model evaluations ==="
python instance_metrics.py "${PATHS[@]}"

echo ""
echo "=== Center Gaussian baseline ==="
python instance_metrics.py --center-gaussian "${PATHS[@]}"
