import os
import sys
import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_TS_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ts_forecaster')
_HTM_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'htm_streamer')

sys.path.append(_SOURCE_DIR)
sys.path.append(_TS_SOURCE_DIR)
sys.path.append(_HTM_SOURCE_DIR)

from htm_source.utils.fs import save_models
from htm_source.pipeline.htm_batch_runner import run_batch
from ts_source.pipeline.pipeline import run_pipeline

from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
# from pyod.models.lof import LOF
# from pyod.models.auto_encoder import AutoEncoder
# from pyod.models.vae import VAE
from pyod.models.kde import KDE

PYOD_MODNAMES_MODELS = {
    'IForest': IForest(),
    'OCSVM': OCSVM(),
    'KNN': KNN(),
    # 'LOF': LOF(),
    # 'AE': AutoEncoder(),
    # 'VAE': VAE(),
    'KDE': KDE()
}


class NaiveModel:
    def __init__(self):
        pass
    def predict(self, lag1):
        return lag1


class SteeringEntropy:
    def __init__(self):
        pass
    def predict(self, lag1, lag2, lag3):
        return lag1 + (lag1 - lag2) + 0.5 * ((lag1 - lag2) - (lag2 - lag3))


def train_save_models(df_train: pd.DataFrame, alg: str, dir_output: str, config: dict,
                      htm_config_user: dict, htm_config_model: dict):
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
    # dir_output_models = os.path.join(dir_output, 'models')
    features_model = list(htm_config_user['features'].keys())
    if alg == 'HTM':
        # htm_source
        features_models, features_outputs = run_batch(cfg_user=htm_config_user,
                                                      cfg_model=htm_config_model,
                                                      config_path_user=None,
                                                      config_path_model=None,
                                                      learn=True,
                                                      data=df_train[features_model],
                                                      iter_print=1000,
                                                      features_models={})
    elif alg == 'Naive':
        features_models = {feat: NaiveModel() for feat in features_model}
    elif alg == 'SteeringEntropy':
        features_models = {feat: SteeringEntropy() for feat in features_model}
    elif alg in PYOD_MODNAMES_MODELS:
        model = PYOD_MODNAMES_MODELS[alg]
        features_models = {feat: model.fit(df_train[features_model]) for feat in features_model}
    else:
        # ts_source
        config_ts = {k: v for k, v in config.items()}
        config_ts['train_models'] = True
        config_ts['modnames_grids'] = {k: v for k, v in config_ts['modnames_grids'].items() if k == alg}
        output_dirs = {'data': os.path.join(dir_output, 'data_files'),
                       'results': os.path.join(dir_output, 'anomaly'),
                       'models': dir_output}
        modnames_models, modname_best, modnames_preds = run_pipeline(config=config_ts,
                                                                     data=df_train,
                                                                     data_path=False,
                                                                     output_dir=False,
                                                                     output_dirs=output_dirs)
        features_models = {modname_best: modnames_models[modname_best]}
    # save_models(features_models, dir_output)

    return features_models
