import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)


def subtract_median(filenames_data, wllevels_filenames, time_col='timestamp'):
    filenames_wllevels = {}
    wllevels_medians = {}
    for wl, filenames in wllevels_filenames.items():
        dfs = [filenames_data[f] for f in filenames]
        df = pd.concat(dfs, axis=0)
        feats_medians = {feat: m for feat, m in dict(df.median()).items() if feat != time_col}
        wllevels_medians[wl] = feats_medians
        for f in filenames:
            filenames_wllevels[f] = wl
    for fn, data in filenames_data.items():
        if fn not in filenames_wllevels:
            continue
        wl = filenames_wllevels[fn]
        feats_medians = wllevels_medians[wl]
        for feat, median in feats_medians.items():
            data[feat] = data[feat] - median
    return filenames_data


def get_autocorr(data_t1, data_t0):
    diff = data_t1-data_t0
    if data_t0 == 0:
        data_t0 = 0.001
    diff_pct = (diff / data_t0)*100
    return diff_pct


def get_autocorrs(steering_angles):
    diff_pcts = [0]
    for _, v in enumerate(steering_angles):
        if _ == 0:
            continue
        diff = get_autocorr(v, steering_angles[_-1])
        diff_pcts.append(diff)
    return diff_pcts


def select_by_autocorr(steering_angles, diff_pcts, diff_thresh=5):
    data_selected = []
    for _, v in enumerate(steering_angles):
        if diff_pcts[_] > diff_thresh:
            data_selected.append(v)
    return data_selected


def update_colnames(filenames_data:dict, colnames:list):
    for fname, data in filenames_data.items():
        data.columns = colnames
    return filenames_data


def agg_data(filenames_data:dict, hz_baseline:int, hz_convertto:int):
    agg = int(hz_baseline / hz_convertto)
    for fname, data in filenames_data.items():
        data = data.groupby(data.index // agg).mean()
        filenames_data[fname] = data
    return filenames_data


def get_dftrain(wllevels_filenames, filenames_data, columns_model, dir_data):
    path_out = os.path.join(dir_data, 'train.csv')
    dfs_train = []
    for fname in wllevels_filenames['training']:
        dfs_train.append(filenames_data[fname])
    df_train = pd.concat(dfs_train, axis=0)[columns_model]
    df_train.to_csv(path_out)
    return df_train


def get_wllevelsdf(subj, subj_wllevels_diffs):
    subj_df = pd.DataFrame(subj_wllevels_diffs)
    col_vals = [subj]+['' for _ in range(subj_df.shape[0]-1)]
    subj_df.insert(len(subj_df.columns), 'subject', col_vals)
    return subj_df


def clip_data(filenames_data:dict, clip_percents:dict):
    for fname, data in filenames_data.items():
        clip_count_start = int(data.shape[0] * (clip_percents['start']/100))
        clip_count_end = data.shape[0] - int(data.shape[0] * (clip_percents['end']/100))
        filenames_data[fname] = data[clip_count_start:clip_count_end]
    return filenames_data


def clip_start(filenames_data, cfg):
    filenames_data_ = {}
    for fn, data in filenames_data.items():
        d_array = data[cfg['columns_model'][0]].values
        start_positive = True if d_array[0]>0 else False
        cross_zero_ind = None
        for _,val in enumerate(d_array):
            if cross_zero_ind is not None:
                break
            if start_positive:
                if val < 0:
                    cross_zero_ind = _
            else:
                if val > 0:
                    cross_zero_ind = _
        data = data[cross_zero_ind:]
        filenames_data_[fn] = data[_:]
    return filenames_data_


def get_wllevels_alldata(wllevels_anomscores:dict, wllevels_filenames:dict, filenames_data:dict, columns_model:list, dir_output:str, save_results=False):
    dir_out = os.path.join(dir_output, 'data_files')
    wllevels_alldata = {}
    print('  test data...')
    for wllevel in wllevels_anomscores:
        wllevel_filenames = wllevels_filenames[wllevel]
        wllevel_datas = []
        print(f"    wllevel = {wllevel}")
        for fn in wllevel_filenames:
            wllevel_datas.append(filenames_data[fn])
            print(f"      {fn}")
        wllevel_data = pd.concat(wllevel_datas, axis=0)[columns_model]
        wllevels_alldata[wllevel] = wllevel_data
        # save data
        if save_results:
            path_out = os.path.join(dir_out, f"{wllevel}.csv")
            wllevel_data.to_csv(path_out)
    return wllevels_alldata


def prep_data(data, cfg_prep):
    if cfg_prep['difference']:
        data = diff_data(data, cfg_prep['difference'])
    if cfg_prep['standardize']:
        data = standardize_data(data)
    if cfg_prep['movingaverage']:
        data = movingavg_data(data, cfg_prep['movingaverage'])
    return data


def preprocess_data(filenames_data, cfg_prep, columns_model):
    filenames_data2 = {}
    percents_data_dropped = []
    for fn, data in filenames_data.items():
        data = prep_data(data, cfg_prep)
        diff_pcts = get_autocorrs(data[ columns_model[0] ].values)  #data['steering angle'].values
        data_selected = select_by_autocorr(data[ columns_model[0] ].values, diff_pcts, diff_thresh=cfg_prep['autocorr_thresh'])  #data['steering angle'].values
        percent_data_dropped = 1 - (len(data_selected) / float(len(data)))
        percents_data_dropped.append(percent_data_dropped)
        data = pd.DataFrame({ columns_model[0] : data_selected})  #pd.DataFrame({'steering angle': data_selected})
        filenames_data2[fn] = data.astype('float32')
    percent_data_dropped = np.sum(percents_data_dropped) / len(filenames_data)
    print(f"  % data DROPPED = {round(percent_data_dropped * 100, 1)}")
    return filenames_data2


def diff_data(data, diff_lag):
    df_dict = {}
    for c in data:
        c_diff = [data[c][i] - data[c][i - diff_lag] for i in range(diff_lag, len(data[c]))]
        df_dict[c] = c_diff
    return pd.DataFrame(df_dict)


def standardize_data(data):
    # fit transform
    transformer = StandardScaler()
    transformer.fit(data)
    # difference transform
    transformed = transformer.transform(data)
    df = pd.DataFrame(transformed)
    df.columns = data.columns
    return df


def movingavg_data(data, window):
    df_dict = {}
    for c in data:
        rolling = data[c].rolling(window=window)
        df_dict[c] = rolling.mean()
    df = pd.DataFrame(df_dict)
    return df.dropna(axis=0, how='all')