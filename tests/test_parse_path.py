import pytest
from instance_metrics import parse_path

PATHS_AND_EXPECTED = [
    ('cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=1003_data_seed_train=1001_data_seed_val=1002_n_test=1000_n_train=10000_n_val=2500/alpha=0.01_criterion=L1_lr=0.1_pooling=ABMIL_seed=1001.pt',
     'ABMIL', 1003, 1000, 0.5, 12, 20, 60, 1001),
    ('cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=2003_data_seed_train=2001_data_seed_val=2002_n_test=1000_n_train=10000_n_val=2500/alpha=0.01_criterion=L1_lr=0.1_pooling=ABMIL_seed=2001.pt',
     'ABMIL', 2003, 1000, 0.5, 12, 20, 60, 2001),
    ('cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=3003_data_seed_train=3001_data_seed_val=3002_n_test=1000_n_train=10000_n_val=2500/alpha=0.001_criterion=L1_lr=0.001_pooling=ABMIL_seed=3001.pt',
     'ABMIL', 3003, 1000, 0.5, 12, 20, 60, 3001),
    ('cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=1003_data_seed_train=1001_data_seed_val=1002_n_test=1000_n_train=10000_n_val=2500/alpha=0.001_criterion=L1_lr=0.0001_pooling=TransMIL_seed=1001.pt',
     'TransMIL', 1003, 1000, 0.5, 12, 20, 60, 1001),
    ('cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=2003_data_seed_train=2001_data_seed_val=2002_n_test=1000_n_train=10000_n_val=2500/alpha=0.001_criterion=L1_lr=0.001_pooling=TransMIL_seed=2001.pt',
     'TransMIL', 2003, 1000, 0.5, 12, 20, 60, 2001),
    ('cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=3003_data_seed_train=3001_data_seed_val=3002_n_test=1000_n_train=10000_n_val=2500/alpha=0.001_criterion=L1_lr=0.0001_pooling=TransMIL_seed=3001.pt',
     'TransMIL', 3003, 1000, 0.5, 12, 20, 60, 3001),
    ('cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=1003_data_seed_train=1001_data_seed_val=1002_n_test=1000_n_train=10000_n_val=2500/alpha=0.01_criterion=L1_lr=0.01_pooling=SmAP_seed=1001.pt',
     'SmAP', 1003, 1000, 0.5, 12, 20, 60, 1001),
    ('cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=2003_data_seed_train=2001_data_seed_val=2002_n_test=1000_n_train=10000_n_val=2500/alpha=0.01_criterion=L1_lr=0.01_pooling=SmAP_seed=2001.pt',
     'SmAP', 2003, 1000, 0.5, 12, 20, 60, 2001),
    ('cluster/tufts/hugheslab/eharve06/pooling/experiments/varying_n_embedding_level=True/delta=0.5_r=12_s_low=20_s_high=60/data_seed_test=3003_data_seed_train=3001_data_seed_val=3002_n_test=1000_n_train=10000_n_val=2500/alpha=0.01_criterion=L1_lr=0.001_pooling=SmAP_seed=3001.pt',
     'SmAP', 3003, 1000, 0.5, 12, 20, 60, 3001),
]


@pytest.mark.parametrize(
    "path,exp_pooling,exp_dseed,exp_ntest,exp_delta,exp_r,exp_slow,exp_shigh,exp_seed",
    PATHS_AND_EXPECTED,
)
def test_parse_path(path, exp_pooling, exp_dseed, exp_ntest, exp_delta, exp_r, exp_slow, exp_shigh, exp_seed):
    params, pooling = parse_path(path)
    assert pooling == exp_pooling
    assert params['data_seed_test'] == exp_dseed
    assert params['n_test'] == exp_ntest
    assert params['delta'] == exp_delta
    assert params['r'] == exp_r
    assert params['s_low'] == exp_slow
    assert params['s_high'] == exp_shigh
    assert params['seed'] == exp_seed
