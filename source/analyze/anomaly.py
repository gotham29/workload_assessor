import os
import sys

import numpy as np
import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_TS_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ts_forecaster')
_HTM_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'htm_streamer')

sys.path.append(_SOURCE_DIR)
sys.path.append(_TS_SOURCE_DIR)
sys.path.append(_HTM_SOURCE_DIR)

from source.preprocess.preprocess import get_testtypes_alldata
from htm_source.pipeline.htm_batch_runner import run_batch
from ts_source.utils.utils import make_dir, add_timecol
from ts_source.model.model import get_preds_rolling, LAG_MIN, get_model_lag, get_modname, MODNAMES_LAGPARAMS, \
    MODNAMES_MODELS


def get_testtypes_outputs(alg, htm_config, features_models, columns_model: list,
                          filenames_data: dict, testtypes_filenames: dict, dir_output: str,
                          time_col: str = 'timestamp', forecast_horizon: int = 1) -> dict:
    testtypes_anomscores = {ttype: [] for ttype in testtypes_filenames if ttype != 'training'}
    testtypes_predcounts = {ttype: [] for ttype in testtypes_filenames if ttype != 'training'}
    # split test data into testtpyes
    testtypes_alldata = get_testtypes_alldata(testtypes_anomscores=testtypes_anomscores,
                                              testtypes_filenames=testtypes_filenames,
                                              filenames_data=filenames_data,
                                              columns_model=columns_model,
                                              dir_output=dir_output)
    # get & save anomscores by testtype
    dir_out = os.path.join(dir_output, "anomaly")
    testtypes_features_anomscores = {}
    for ttype, data in testtypes_alldata.items():
        data = add_timecol(data, time_col)
        if alg == 'HTM':
            feats_models, features_outputs = run_batch(cfg=htm_config,
                                                       config_path=None,
                                                       learn=False,
                                                       data=data,
                                                       iter_print=1000,
                                                       features_models=features_models)
            testtypes_features_anomscores[ttype] = {f: [] for f in features_outputs}
            for f, outs in features_outputs.items():
                testtypes_features_anomscores[ttype][f] = outs['anomaly_score']
                testtypes_anomscores[ttype] += outs['anomaly_score']
                testtypes_predcounts[ttype] += outs['pred_count']
        else:  # ts_source alg
            testtypes_features_anomscores[ttype] = {f: [] for f in columns_model}
            for feat, model in features_models.items():  # Assumes single model
                break
            mod_name = get_modname(model)
            features = model.training_series.components
            preds = get_preds_rolling(model=model,
                                      df=data,
                                      features=features,
                                      LAG=max(LAG_MIN, get_model_lag(mod_name, model)),
                                      time_col=time_col,
                                      forecast_horizon=forecast_horizon)
            preds_df = pd.DataFrame(preds, columns=list(features))
            data_ = data.tail(preds_df.shape[0])
            for f in features:
                ascores = list(abs(data_[f].values - preds_df[f].values))
                testtypes_features_anomscores[ttype][f] = ascores
                testtypes_anomscores[ttype] += ascores
        # Save anomaly score by testtype
        path_out = os.path.join(dir_out, f"{ttype}.csv")
        ttype_anom = pd.DataFrame(testtypes_features_anomscores[ttype])
        ttype_anom.to_csv(path_out)
    if testtypes_predcounts == {ttype: [] for ttype in testtypes_filenames if ttype != 'training'}:
        testtypes_predcounts = {}
    return testtypes_anomscores, testtypes_predcounts, testtypes_alldata


def get_testtypes_diffs(testtypes_anomscores):
    # Get difference in median anomaly score between testtypes
    ttypesdiffs = {}
    ttype_combos_done = []
    for ttype, ascores in testtypes_anomscores.items():
        ttypesdiffs[ttype] = {}
        testtypes_anomscores_other = {k: v for k, v in testtypes_anomscores.items() if k != ttype}
        for ttype_other, ascores_other in testtypes_anomscores_other.items():
            combo = '--'.join(sorted([ttype, ttype_other]))
            if combo in ttype_combos_done:
                continue
            ttype_combos_done.append(combo)
            ttypesdiffs[ttype][ttype_other] = abs(np.median(ascores) - np.median(ascores_other))
    return ttypesdiffs
