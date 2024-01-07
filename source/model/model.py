import os
import sys
import pandas as pd
import numpy as np

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_TS_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ts_forecaster')
_HTM_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'htm_streamer')

sys.path.append(_SOURCE_DIR)
sys.path.append(_TS_SOURCE_DIR)
sys.path.append(_HTM_SOURCE_DIR)

from htm_source.utils.fs import save_models
from htm_source.pipeline.htm_batch_runner import run_batch
from ts_source.pipeline.pipeline import run_pipeline
from source.analyze.anomaly import get_ascore_entropy

from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.lof import LOF
# from pyod.models.auto_encoder import AutoEncoder
# from pyod.models.vae import VAE
from pyod.models.kde import KDE

PYOD_MODNAMES_MODELS = {
    'IForest': IForest(),
    'OCSVM': OCSVM(),
    'KNN': KNN(),
    'LOF': LOF(),
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

    df_train.reset_index(drop=True, inplace=True)

    features_model = list(htm_config_user['features'].keys())
    features_alphas = {}
    if alg == 'HTM':
        # drop timestamp feature if megamodel
        if not htm_config_user['models_state']['model_for_each_feature']:
            htm_features = {k: v for k, v in htm_config_user['features'].items() if v['type'] != 'timestamp'}
            htm_config_user['features'] = htm_features
            config['htm_config_user'] = htm_config_user
        # htm_source
        features_models, features_outputs = run_batch(cfg_user=htm_config_user,
                                                      cfg_model=htm_config_model,
                                                      config_path_user=None,
                                                      config_path_model=None,
                                                      learn=True,
                                                      data=df_train[features_model],
                                                      iter_print=1000,
                                                      features_models={})
    elif alg in ['IPSD', 'FPSD']:
        features_models = {feat: None for feat in features_model if feat != config['time_col']}
    elif alg == 'Naive':
        features_models = {feat: NaiveModel() for feat in features_model if feat != config['time_col']}
    elif alg in ['SteeringEntropy', 'Fessonia']:
        features_models = {feat: SteeringEntropy() for feat in features_model if feat != config['time_col']}
        features_errors = {f:[] for f in features_models}
        for feat, model in features_models.items():
            pred_prev = None
            errors = []
            print(f"  df_train = {df_train.shape}")
            for _,row in df_train.iterrows():
                print(f"    ind = {_}")
                aScore, pred_prev = get_ascore_entropy(_, row, feat, model, df_train, pred_prev, LAG=3)
                errors.append(aScore)
            features_errors[feat] = errors
        features_alphas = {f: np.percentile(errors, 90) for f,errors in features_errors.items()}
        print(f"\nfeatures_alphas = {features_alphas}")

    elif alg in PYOD_MODNAMES_MODELS:
        model = PYOD_MODNAMES_MODELS[alg]
        features_model = [f for f in features_model if f != config['time_col']]
        features_models = {feat: model.fit(df_train[features_model]) for feat in features_model if
                           feat != config['time_col']}
    else:
        # ts_source
        config_ts = {k: v for k, v in config.items()}
        config_ts['train_models'] = True
        config_ts['modnames_grids'] = {k: v for k, v in config_ts['modnames_grids'].items() if k == alg}
        output_dirs = {'data': os.path.join(dir_output, 'data_files'),
                       'results': os.path.join(dir_output, 'anomaly'),
                       'models': dir_output}
        features_model = [f for f in features_model if f != config['time_col']]
        features_models = {}
        for feat in features_model:
            modnames_models, modname_best, modnames_preds = run_pipeline(config=config_ts,
                                                                         data=df_train[[feat, config['time_col']]],
                                                                         data_path=False,
                                                                         output_dir=False,
                                                                         output_dirs=output_dirs)
            features_models[feat] = modnames_models[alg]

    # dir_output_models = os.path.join(dir_output, 'models')
    # save_models(features_models, dir_output)
    return config, features_models, features_alphas
