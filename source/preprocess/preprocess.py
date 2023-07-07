import copy
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_TS_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ts_forecaster')

sys.path.append(_SOURCE_DIR)
sys.path.append(_TS_SOURCE_DIR)

from ts_source.utils.utils import add_timecol


def preprocess_data(subj, cfg, filenames_data, subjects_spacesadd):
    # Agg
    filenames_data = agg_data(filenames_data=filenames_data, hz_baseline=cfg['hzs']['baseline'],
                              hz_convertto=cfg['hzs']['convertto'])
    # Clip both ends
    filenames_data = clip_data(filenames_data=filenames_data, clip_percents=cfg['clip_percents'])
    # Subtract median
    filenames_data = subtract_median(filenames_data=filenames_data)
    # Differece/Standardize/MA
    filenames_data = {fn: diff_standard_ma(data, cfg['preprocess']) for fn, data in filenames_data.items()}
    # Transform
    filenames_data = filter_by_autocorr(subj, subjects_spacesadd[subj], filenames_data, cfg['preprocess'], filestrs_doautocorr=['training','static'])
    # Train models
    df_train = get_dftrain(wllevels_filenames=cfg['wllevels_filenames'], filenames_data=filenames_data,
                           columns_model=cfg['columns_model'])
    # Add timecol
    df_train = add_timecol(df_train, cfg['time_col'])
    for fn, data in filenames_data.items():
        data.drop(columns=[cfg['time_col']], inplace=True)
        filenames_data[fn] = add_timecol(data, cfg['time_col'])

    return filenames_data, df_train


def subtract_median(filenames_data):
    for fn, data in filenames_data.items():
        feats_medians = {feat: m for feat, m in dict(data.median()).items()}
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


def select_by_autocorr(data, diff_pcts, diff_thresh):
    inds_keep = []
    for _ in range(data.shape[0]):
        if diff_pcts[_] > diff_thresh:
            inds_keep.append(_)
    return data[data.index.isin(inds_keep)]


def update_colnames(filenames_data:dict, colnames:list):
    for fname, data in filenames_data.items():
        data.columns = colnames
        # HACK - ADD COLUMN
        # data.columns = [c for c in colnames if c != 'steering angle 2']
        # data.insert(loc=2, column='steering angle 2', value=data['steering angle'].values)
    return filenames_data


def agg_data(filenames_data:dict, hz_baseline:int, hz_convertto:int):
    agg = int(hz_baseline / hz_convertto)
    for fname, data in filenames_data.items():
        data = data.groupby(data.index // agg).mean()
        filenames_data[fname] = data
    return filenames_data


def get_dftrain(wllevels_filenames, filenames_data, columns_model):
    dfs_train = []
    for fname in wllevels_filenames['training']:
        dfs_train.append(filenames_data[fname])
    df_train = pd.concat(dfs_train, axis=0)
    df_train = df_train[columns_model]
    return df_train


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


def get_wllevels_totaldfs(wllevels_filenames:dict, filenames_data:dict, columns_model:list, out_dir_files:str):
    levels_order = [v for v in list(wllevels_filenames.keys()) if v != 'training']
    wllevels_totaldfs = {}
    print('  test data...')
    for wllevel, wllevel_filenames in wllevels_filenames.items():
        if wllevel == 'training':
            continue
        wllevel_datas = []
        print(f"    wllevel = {wllevel}")
        for fn in wllevel_filenames:
            wllevel_datas.append(filenames_data[fn])
            print(f"      {fn}")
        wllevel_data = pd.concat(wllevel_datas, axis=0)[columns_model]
        wllevels_totaldfs[wllevel] = wllevel_data
        # save data
        path_out = os.path.join(out_dir_files, f"{wllevel}.csv")
        wllevel_data.to_csv(path_out)
    # reorder wllevels
    wllevels_totaldfs_ = {}
    for k in levels_order:
        wllevels_totaldfs_[k] = wllevels_totaldfs[k]

    return wllevels_totaldfs_


def get_wllevels_indsend(wllevels_totaldfs_):
    wllevels_indsend, wllevel_i = {}, 0
    for wllevel, dftotal in wllevels_totaldfs_.items():
        wllevel_i += len(dftotal)
        wllevels_indsend[wllevel] = wllevel_i
    return wllevels_indsend


def diff_standard_ma(data, cfg_prep):
    if cfg_prep['difference']:
        data = diff_data(data, cfg_prep['difference'])
    if cfg_prep['standardize']:
        data = standardize_data(data)
    if cfg_prep['movingaverage']:
        data = movingavg_data(data, cfg_prep['movingaverage'])
    return data


def filter_by_autocorr(subj, spaces_add, filenames_data, cfg_prep, filestrs_doautocorr=['training', 'static']):
    filenames_data2 = {}
    percents_data_dropped = []
    for fn, data in filenames_data.items():
        # do autocorr if file in filestrs_doautocorr
        if any(filestr in fn for filestr in filestrs_doautocorr):
            diff_pcts = get_autocorrs(data[cfg_prep['autocorr_column']].values)
            data_ = select_by_autocorr(data, diff_pcts, diff_thresh=cfg_prep['autocorr_thresh'])
            percent_data_dropped = 1 - (len(data_) / float(len(data)))
            percents_data_dropped.append(percent_data_dropped)
        else:
            data_ = copy.deepcopy(data)
        filenames_data2[fn] = data_.astype('float32')
    percent_data_dropped = np.sum(percents_data_dropped) / len(filenames_data)
    print(f"  {subj}{spaces_add} --> %data DROPPED: {round(percent_data_dropped * 100, 1)}")
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