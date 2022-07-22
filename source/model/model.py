import os
import sys

from htm_source.pipeline.htm_batch_runner import run_batch

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.utils.utils import save_models


def train_models(df_train, dir_output: str, htm_config: dict):
    features_models, features_outputs = run_batch(cfg=htm_config,
                                                  config_path=None,
                                                  learn=True,
                                                  data=df_train,
                                                  iter_print=1000,
                                                  features_models={})
    save_models(features_models, dir_output)
    return features_models
