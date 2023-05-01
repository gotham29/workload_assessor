"""
Simple script to test:
    - htm.core (through htm_streamer)
    - darts (through ts_forecaster)
"""

import os
import sys
import pandas as pd

# Add repo dirs to path -- ASSUMES 'ts_forecaster' and 'htm_streamer' kept in same dir as 'workload_assessor'
_TS_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ts_forecaster')
_HTM_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'htm_streamer')
sys.path.append(_TS_SOURCE_DIR)
sys.path.append(_HTM_SOURCE_DIR)

from htm_source.pipeline.htm_batch_runner import run_batch
from pyod.models.knn import KNN
from ts_source.pipeline.pipeline import run_pipeline


# Build Data
df_dict = {
    'steering angle': [
        107.326262103429,
        107.252477277373,
        107.091684379027,
        106.771488525831,
        106.485175533561,
        105.835721873688,
        105.365618736137,
        104.817764140168,
        104.279441073186,
        103.594956226834,
        103.047507173348,
        102.459724999038,
        101.795290172995,
        101.085130645553,
        100.138451058664,
        99.1748573620684,
        98.1204153193227,
        97.0332916751723,
        95.9723844314904,
        94.9369520858654,
        93.7556439750418,
        92.9600791218878,
        91.7983657565759,
        90.8669303515012,
        89.9100908382111,
        88.9900064804953,
        88.1238644288739,
        87.3082967264052,
        86.3790721148266,
        85.6629208039352,
        84.8996019141831,
        84.0286943115117,
        83.3613686079021,
        82.7528499792728,
        82.0313011113898,
        81.4266654453036,
        80.7933131026257,
        80.1669847557356,
        79.5628289104385,
        78.9994936907161,
        78.3957792463942,
        77.8588213634727,
        77.2756692036471,
        76.6807810712802,
        76.1339467348188,
        75.5265183751358,
        74.9704453540483,
        74.4181596728557,
        73.8742529262273,
        73.3494485111602,
        72.8576339097843,
        72.424772378682,
        72.0033087257572,
        71.6506216626799,
        71.3067387134427,
        71.0449614682182,
        70.7871901289419,
        70.591896237551,
        70.4807007578632,
        70.3583054754839,
        70.4009702521343,
        70.4928311738291,
        70.5008481083674,
        70.4769612292926,
        70.4758735216673,
        70.5670480093505,
        70.57038455676,
        70.6569264686387,
        70.6173967488172,
        70.6553777232444,
        70.6543173363758,
        70.7078438213538,
        70.7044731229985,
        70.752012947169,
        70.7415695879281,
        70.7697868069367,
        70.7889267045384,
        70.7928233274598,
        70.8076841115465,
        70.88317989999,
        70.9448650459316,
        70.9707053591091,
        71.0589035918615,
        71.2720242769861,
        71.3918121346498,
        71.5491831082214,
        71.7090146574416,
        71.8685798292842,
        72.0450753325319,
        72.2242431472924,
        72.4145031892649,
        72.6327396860779,
        72.8365542384584,
        73.08450888835,
        73.2993132151051,
        73.5271956465734,
        73.7531161062026,
        73.9697799519591,
        74.2174391961692,
        74.4655646007902
    ],
    'timestamp': [_ for _ in range(100)]
}
df_train = pd.DataFrame(df_dict)

# Build configs
htm_config_user = {
    'features': {
        'steering angle': {'type': float, 'min': None, 'max': None, 'weight': 1.0}
    },
    'models_state': {
        'model_for_each_feature': False,
        'return_pred_count': True,
        'use_sp': True
    },
    'timesteps_stop': {
        'learning': None,
        'running': None,
        'sampling': None
    }
}

htm_config_model = {
    'models_encoders': {
        'n': 400,
        'w': 21,
        'n_buckets': 130,
        'p_padding': 10,
        'seed': 0
    },
    'models_params': {
        'anomaly_likelihood': {
            'probationaryPeriod': 500,
            'reestimationPeriod': 100,
        },
        'sp': {
            'potentialPct': 0.8,
            'columnCount': 2048,
            'globalInhibition': True,
            'boostStrength': 0.0,
            'localAreaDensity': 0.0,
            'stimulusThreshold': 0.0,
            'numActiveColumnsPerInhArea': 40,
            'synPermActiveInc': 0.003,
            'synPermConnected': 0.2,
            'synPermInactiveDec': 0.0005,
            'wrapAround': True,
            'minPctOverlapDutyCycle': 0.001,
            'dutyCyclePeriod': 1000,
            'seed': 0
        },
        'tm': {
            'activationThreshold': 20,
            'cellsPerColumn': 32,
            'columnDimensions': 2048,
            'initialPerm': 0.21,
            'maxSegmentsPerCell': 128,
            'maxSynapsesPerSegment': 128,
            'minThreshold': 13,
            'newSynapseCount': 31,
            'permanenceDec': 0.0,
            'permanenceInc': 0.10,
            'permanenceConnected': 0.5,
            'predictedSegmentDecrement': 0.001,
            'seed': 0
        }
    },
    'models_predictor': {
        'enable': False,
        'resolution': 1,
        'sdrc_alpha': 0.1,
        'steps_ahead': [1, 2]
    },
    'spatial_anomaly': {
        'enable': False,
        'tolerance': 0.05,
        'perc_min': 0,
        'perc_max': 100,
        'anom_prop': 0.3,
        'window': 100000
    }
}

config_ts = {
    'mode': 'post-hoc',
    'alg': 'LightGBMModel',
    'data_cap': 10000,
    'test_prop': 0.3,
    'eval_metric': 'rmse',
    'forecast_horizon': 1,
    'time_col': 'timestamp',
    'scale': False,
    'train_models': True,
    'do_gridsearch': True,
    'dirs': {
        'input': os.getcwd(),
        'output': os.getcwd(),
        'results': os.getcwd()
    },
    'features': {
        'in': ['steering angle'],
        'pred': ['steering angle']
    },
    'modnames_grids': {
        'LightGBMModel': {
            'lags': [1, 5, 10]
        }
    }
}

# Test DARTS
print('\nTesting DARTS...')
output_dirs = {'data': os.path.join(config_ts['dirs']['output'], 'data_files'),
               'results': os.path.join(config_ts['dirs']['output'], 'anomaly'),
               'models': os.path.join(config_ts['dirs']['output'], 'models'),
               'scalers': os.path.join(config_ts['dirs']['output'], 'scalers')}
for outtype, outdir in output_dirs.items():
    os.makedirs(outdir, exist_ok=True)
modnames_models, modname_best, modnames_preds = run_pipeline(config=config_ts,
                                                             data=df_train,
                                                             data_path=False,
                                                             output_dir=False,
                                                             output_dirs=output_dirs)
print("  DONE")

# Test HTM
print('Testing HTM...')
features_models, features_outputs = run_batch(cfg_user=htm_config_user,
                                              cfg_model=htm_config_model,
                                              config_path_user=None,
                                              config_path_model=None,
                                              learn=True,
                                              data=df_train,
                                              iter_print=1000,
                                              features_models={})
print("  DONE")

# Test PYOD
print('\nTesting PYOD...')
model = KNN()
model.fit(df_train)
print("  DONE")
