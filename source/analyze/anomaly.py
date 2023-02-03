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

def get_ascores_entropy(data):
    preds = []
    for _ in range(len(data)):
        if _ < 3:
            continue
        lag1, lag2, lag3 = data[_-3], data[_-2], data[_-1]
        pred = lag1 + (lag1-lag2) + 0.5*((lag1-lag2) - (lag2-lag3))
        # θ (n-1)   +   (θ(n-1) - θ(n-2))   +   1/2 ( (θ(n-1)-θ(n-2)) - (θ(n-2)-θ(n-3)) )
        preds.append(pred)
    ascores = []
    for _, p in enumerate(preds):
        ascore = abs(p - data[_+3])
        ascores.append(ascore)
    return ascores


def get_testtypes_outputs(alg, htm_config_user, htm_config_model, features_models, columns_model: list,
                          filenames_data: dict, testtypes_filenames: dict, dir_output: str, learn_in_testing: bool,
                          time_col: str = 'timestamp', forecast_horizon: int = 1, save_results: bool = True) -> dict:
    testtypes_anomscores = {ttype: [] for ttype in testtypes_filenames if ttype != 'training'}
    testtypes_predcounts = {ttype: [] for ttype in testtypes_filenames if ttype != 'training'}
    # split test data into testtpyes
    testtypes_alldata = get_testtypes_alldata(testtypes_anomscores=testtypes_anomscores,
                                              testtypes_filenames=testtypes_filenames,
                                              filenames_data=filenames_data,
                                              columns_model=columns_model,
                                              dir_output=dir_output,
                                              save_results=save_results)
    # get & save anomscores by testtype
    dir_out = os.path.join(dir_output, "anomaly")
    testtypes_features_anomscores = {}
    for ttype, data in testtypes_alldata.items():
        data = add_timecol(data, time_col)
        if alg == 'HTM':
            feats_models, features_outputs = run_batch(cfg_user=htm_config_user,
                                                       cfg_model=htm_config_model,
                                                       config_path_user=None,
                                                       config_path_model=None,
                                                       learn=learn_in_testing,
                                                       data=data,
                                                       iter_print=1000,
                                                       features_models=features_models)
            testtypes_features_anomscores[ttype] = {f: [] for f in features_outputs}
            for f, outs in features_outputs.items():
                testtypes_features_anomscores[ttype][f] = outs['anomaly_score']
                testtypes_anomscores[ttype] += outs['anomaly_score']
                testtypes_predcounts[ttype] += outs['pred_count']

        elif 'Entropy' in alg:
            testtypes_features_anomscores[ttype] = {f: [] for f in columns_model}
            for f in columns_model:
                ascores = get_ascores_entropy(data[f].values)
                testtypes_features_anomscores[ttype][f] = ascores
                testtypes_anomscores[ttype] += ascores

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
                ascores_ = list(abs(data_[f].values - preds_df[f].values))
                # CLIP and/or NORMALIZE
                ascores = []
                clip_val = np.percentile(ascores_, 95)
                for ascore in ascores_:
                    ascore = ascore if ascore <= clip_val else clip_val
                    ascores.append(ascore)
                testtypes_features_anomscores[ttype][f] = ascores
                testtypes_anomscores[ttype] += ascores
        # Save anomaly score by testtype
        if save_results:
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
            pct_change = (np.mean(ascores_other) - np.mean(ascores)) / np.mean(ascores_other)
            ttypesdiffs[ttype][ttype_other] = round(pct_change*100, 1)
    return ttypesdiffs
