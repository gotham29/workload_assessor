import os
import sys
import pandas as pd

from sklearn.preprocessing import StandardScaler

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)


def get_autocorr(data_t1, data_t0):
    diff = data_t1-data_t0
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


def get_ttypesdf(subj, subj_ttypesdiffs):
    subj_df = pd.DataFrame(subj_ttypesdiffs)
    col_vals = [subj]+['' for _ in range(subj_df.shape[0]-1)]
    subj_df.insert(len(subj_df.columns), 'subject', col_vals)
    return subj_df


def clip_data(filenames_data:dict, clip_percents:dict):
    for fname, data in filenames_data.items():
        clip_count_start = int(data.shape[0] * (clip_percents['start']/100))
        clip_count_end = data.shape[0] - int(data.shape[0] * (clip_percents['end']/100))
        filenames_data[fname] = data[clip_count_start:clip_count_end]
    return filenames_data


def get_testtypes_alldata(testtypes_anomscores:dict, testtypes_filenames:dict, filenames_data:dict, columns_model:list, dir_output:str, save_results=False):
    dir_out = os.path.join(dir_output, 'data_files')
    testtypes_alldata = {}
    # print('  test data...')
    for ttype in testtypes_anomscores:
        ttype_filenames = testtypes_filenames[ttype]
        ttype_datas = []
        # print(f"    type = {ttype}")
        for fn in ttype_filenames:
            ttype_datas.append( filenames_data[fn] )
            # print(f"      {fn}")
        ttype_data = pd.concat(ttype_datas, axis=0)[columns_model]
        testtypes_alldata[ttype] = ttype_data
        # save data
        if save_results:
            path_out = os.path.join(dir_out, f"{ttype}.csv")
            ttype_data.to_csv(path_out)
    return testtypes_alldata


def prep_data(data, cfg_prep):
    if cfg_prep['difference']:
        data = diff_data(data, cfg_prep['difference'])
    if cfg_prep['standardize']:
        data = standardize_data(data)
    if cfg_prep['movingaverage']:
        data = movingavg_data(data, cfg_prep['movingaverage'])
    return data


def preprocess_data(filenames_data, cfg_prep):
    filenames_data2 = {}
    for fn, data in filenames_data.items():
        data = prep_data(data, cfg_prep)
        diff_pcts = get_autocorrs(data['steering angle'].values)
        data_selected = select_by_autocorr(data['steering angle'].values, diff_pcts, diff_thresh=cfg_prep['autocorr_thresh'])
        percent_data_dropped = 1 - (len(data_selected) / float(len(data)))
        print(f"        % data DROPPED = {round( percent_data_dropped*100 , 1)}")
        data = pd.DataFrame({'steering angle': data_selected})
        filenames_data2[fn] = data.astype('float32')
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