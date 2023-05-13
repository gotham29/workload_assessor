import os
import sys
import numpy as np
from darts import TimeSeries

_TS_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ts_forecaster')
sys.path.append(_TS_SOURCE_DIR)

from ts_source.preprocess.preprocess import reshape_datats
from ts_source.model.model import get_model_lag, LAG_MIN


def get_ascore_entropy(_, row, feat, model, data_test, pred_prev, LAG=3):
    aScore, do_pred = 0, True
    if _ < LAG:
        pred_prev, do_pred = None, False
    if pred_prev:
        aScore = abs(pred_prev - row[feat])
    if do_pred:
        lag1, lag2, lag3 = data_test[feat][_ - 3], data_test[feat][_ - 2], data_test[feat][_ - 1]
        pred_prev = model.predict(lag1, lag2, lag3)
    return aScore, pred_prev


def get_ascore_pyod(_, data, model):
    aScore = [0]
    if _ > 0:
        aScore = model.decision_function(data[(_ - 1):_])  # outlier scores
    return abs(aScore[0])


def get_entropy_ts(_, model, row, data_test, config, pred_prev, LAG_MIN):
    aScore, do_pred = 0, True
    features_model = list(model.training_series.components)
    features = features_model + [config['time_col']]
    LAG = max(LAG_MIN, get_model_lag(config['alg'], model))
    if _ < LAG:
        pred_prev, do_pred = None, False
    if pred_prev:
        aScore = abs(pred_prev - row[features_model])
    if do_pred:
        df_lag = data_test[features][_ - LAG:_]
        ts = TimeSeries.from_dataframe(df_lag, time_col=config['time_col'])
        pred = model.predict(n=config['forecast_horizon'], series=ts)
        pred_prev = reshape_datats(ts=pred, shape=(len(features_model)))
    return aScore, pred_prev


def get_ascores_entropy(data):
    preds = []
    for _ in range(len(data)):
        if _ < 3:
            continue
        lag1, lag2, lag3 = data[_-3], data[_-2], data[_-1]
        pred = lag1 + (lag1-lag2) + 0.5*((lag1-lag2) - (lag2-lag3))
        preds.append(pred)
    ascores = []
    for _, p in enumerate(preds):
        ascore = abs(p - data[_+3])
        ascores.append(ascore)
    return ascores


def get_ascores_naive(data):
    preds = []
    for _ in range(len(data)):
        if _ == 0:
            continue
        pred = data[_-1]
        preds.append(pred)
    ascores = []
    for _, p in enumerate(preds):
        ascore = abs(p - data[_+1])
        ascores.append(ascore)
    return ascores


def get_ascore_pyod(_, data, model):
    aScore = [0]
    if _ > 0:
        aScore = model.decision_function(data[(_-1):_])  # outlier scores
    return abs(aScore[0])


def get_ascores_pyod(data, model):
    aScores = list()
    for _ in range(data.shape[0]):
        aScores.append(get_ascore_pyod(_, data, model))
    return aScores


def get_f1score(tp, fp, fn):
    # get precision
    denom_precision = (tp + fp)
    if denom_precision == 0:
        precision = 0
    else:
        precision = tp / denom_precision
    # get recall
    denom_recall = (tp + fn)
    if denom_recall == 0:
        recall = 0
    else:
        recall = tp / denom_recall
    # get f1
    denom_f1 = (precision + recall)
    if denom_f1 == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / denom_f1
    return round(f1, 3)


def get_wllevels_diffs(wllevels_anomscores):
    # Get difference in median anomaly score between wllevels
    wllevels_diffs = {}
    wllevel_combos_done = []
    for wllevel, ascores in wllevels_anomscores.items():
        wllevels_diffs[wllevel] = {}
        wllevels_anomscores_other = {k: v for k, v in wllevels_anomscores.items() if k != wllevel}
        for wllevel_other, ascores_other in wllevels_anomscores_other.items():
            combo = '--'.join(sorted([wllevel, wllevel_other]))
            if combo in wllevel_combos_done:
                continue
            wllevel_combos_done.append(combo)
            ascores_total = max(np.sum(ascores), 0.025)
            ascores_total_other = max(np.sum(ascores_other), 0.025)
            pct_change = (ascores_total_other - ascores_total) / ascores_total
            wllevels_diffs[wllevel][wllevel_other] = round(pct_change*100, 2)
    return wllevels_diffs


def get_subjects_wldiffs(subjects_wllevelsascores):
    subjects_wldiffs = {}
    subjects_levels_wldiffs = {}
    for subj, wllevels_anomscores in subjects_wllevelsascores.items():
        wl_t0 = max(np.sum(wllevels_anomscores['baseline']), 0.025)
        wl_t1t0_diffs = {}
        for wllevel, ascores in wllevels_anomscores.items():
            if wllevel == 'baseline':
                continue
            wl_t1 = max(np.sum(ascores), 0.025)
            percent_diff = ((wl_t1 - wl_t0) / wl_t0) * 100
            wl_t1t0_diffs[wllevel] = round(percent_diff, 2)
        subjects_levels_wldiffs[subj] = wl_t1t0_diffs
        subjects_wldiffs[subj] = round(sum(list(wl_t1t0_diffs.values())), 2)
    return subjects_wldiffs, subjects_levels_wldiffs

