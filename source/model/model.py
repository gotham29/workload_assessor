import os
import sys

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_TS_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ts_forecaster')
_HTM_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'htm_streamer')

sys.path.append(_SOURCE_DIR)
sys.path.append(_TS_SOURCE_DIR)
sys.path.append(_HTM_SOURCE_DIR)

from htm_source.utils.fs import save_models
from htm_source.pipeline.htm_batch_runner import run_batch
from ts_source.pipeline.pipeline import run_pipeline


def train_save_models(df_train, alg, dir_output: str, config: dict, htm_config: dict):
    """
    Purpose:
        Train models (using 'htm_source' or 'ts_source' modules)
    Inputs:
        df_train:
            type: pd.DataFrame
            meaning: data to train on
        alg:
            type: str
            meaning: which ML algorithm to use
        dir_output:
            type: str
            meaning: dir to save outputs
        htm_config:
            type: dict
            meaning: config for 'htm_streamer.run_batch'
    Outputs:
        features_models:
            type: dict
            meaning: keys are pred features (or 'megamodel_features={featurecount}'), values are models
    """
    dir_output_models = os.path.join(dir_output, 'models')
    if alg == 'HTM':
        # htm_source
        features_models, features_outputs = run_batch(cfg=htm_config,
                                                    config_path=None,
                                                    learn=True,
                                                    data=df_train[htm_config['features'].keys()],
                                                    iter_print=1000,
                                                    features_models={})
        save_models(features_models, dir_output_models)
    else:
        # ts_source
        config_ts = {k:v for k,v in config.items()}
        config_ts['train_models'] = True
        config_ts['modnames_grids'] = {k:v for k,v in config_ts['modnames_grids'].items() if k == alg}
        output_dirs = {'data': os.path.join(dir_output, 'data_files'),
                        'results': os.path.join(dir_output, 'anomaly'),
                        'models': dir_output_models,
                        'scalers': os.path.join(dir_output, 'scalers')}
        modnames_models, modname_best, modnames_preds = run_pipeline(config=config_ts,
                                                                     data=df_train,
                                                                     data_path=False,
                                                                     output_dir=False,
                                                                     output_dirs=output_dirs)
        features_models = {modname_best: modnames_models[modname_best]}
    return features_models

