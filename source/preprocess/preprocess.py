import copy
import os
import sys
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_TS_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ts_forecaster')

sys.path.append(_SOURCE_DIR)
sys.path.append(_TS_SOURCE_DIR)

from ts_source.utils.utils import add_timecol
from source.utils.utils import load_files

FILETYPES_READFUNCS = {
    'xls': pd.read_excel,
    'csv': pd.read_csv
}


def preprocess_data(subj, cfg, filenames_data, subjects_spacesadd):
    diff_standard_ma_list = [cfg['preprocess']['movingaverage'], cfg['preprocess']['standardize'],
                             cfg['preprocess']['difference']]
    do_agg = True if cfg['hzs']['baseline'] != cfg['hzs']['convertto'] else False
    do_clip = True if not (cfg['clip_percents']['start'] == 0 and cfg['clip_percents']['end'] == 0) else False
    do_subtractmedian = True if cfg['preprocess']['subtract_median'] else False
    do_diff_standard_ma = True if any(diff_standard_ma_list) else False
    do_filter_by_autocorr = True if cfg['preprocess']['autocorr'] else False

    # Agg
    if do_agg:
        filenames_data = agg_data(filenames_data=filenames_data, hz_baseline=cfg['hzs']['baseline'],
                                  hz_convertto=cfg['hzs']['convertto'])
    # Clip both ends
    if do_clip:
        filenames_data = clip_data(filenames_data=filenames_data, clip_percents=cfg['clip_percents'])
    # Subtract median
    if do_subtractmedian:
        filenames_data = subtract_median(filenames_data=filenames_data)
    # Differece/Standardize/MA
    if do_diff_standard_ma:
        filenames_data = {fn: diff_standard_ma(data, cfg['preprocess']) for fn, data in filenames_data.items()}
    # Transform
    if do_filter_by_autocorr:
        filenames_data = filter_by_autocorr(subj, subjects_spacesadd[subj], filenames_data, cfg['preprocess'],
                                            filestrs_doautocorr=['training', 'static'])
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
    diff = data_t1 - data_t0
    if data_t0 == 0:
        data_t0 = 0.001
    diff_pct = (diff / data_t0) * 100
    return diff_pct


def get_autocorrs(steering_angles):
    diff_pcts = [0]
    for _, v in enumerate(steering_angles):
        if _ == 0:
            continue
        diff = get_autocorr(v, steering_angles[_ - 1])
        diff_pcts.append(diff)
    return diff_pcts


def get_psd_peak(data_1d, sample_rate):
    # Compute the power spectral density (PSD) using the FFT
    psd = np.abs(np.fft.fft(data_1d)) ** 2
    # Create a frequency array that corresponds to the PSD values
    # The sampling frequency (Fs) is assumed to be 1. If your signal has a different Fs, adjust accordingly.
    freqs = np.fft.fftfreq(n=len(data_1d), d=1 / sample_rate)
    # Find the index of the highest peak in the PSD
    highest_peak_index = np.argmax(psd)
    # Calculate the corresponding frequency of the highest peak
    highest_peak_frequency = np.abs(freqs[highest_peak_index])
    return highest_peak_frequency, freqs, psd


def get_pds_peaks(dir_input, dir_output, file_type, file_substr, column_names, column_behavior, sample_rate,
                  autocorr_thresh=5):
    subjs_psdpeaks = {}
    subjs = [f for f in os.listdir(dir_input) if '.' not in f and '--drop' not in f]
    for subj in subjs:
        dir_input_subj = os.path.join(dir_input, subj)
        filenames = [f for f in os.listdir(dir_input_subj) if file_substr in f]
        filenames_data = load_files(dir_input=dir_input_subj, file_type=file_type,
                                    read_func=FILETYPES_READFUNCS[file_type],
                                    filenames=filenames)
        filenames_data = update_colnames(filenames_data=filenames_data, colnames=column_names)
        subj_data = pd.concat(filenames_data.values(), axis=0)
        diff_pcts = get_autocorrs(subj_data[column_behavior].values)
        subj_data_ = select_by_autocorr(subj_data, diff_pcts, diff_thresh=autocorr_thresh)
        percent_data_dropped = 1 - (len(subj_data_) / float(len(subj_data)))
        print(f"{subj}\n  % dropped --> {percent_data_dropped}")
        psd_peak, freqs, psd = get_psd_peak(subj_data_[column_behavior].values, sample_rate)
        subjs_psdpeaks[subj] = psd_peak
        path_out = os.path.join(dir_output, f"PSD freqs by psd--{subj}.png")
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, psd)
        plt.xlim(0, 20)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title(f'Highest Peak Frequency: {round(psd_peak, 3)} Hz')
        plt.savefig(path_out)
    return subjs_psdpeaks


def select_by_autocorr(data, diff_pcts, diff_thresh):
    inds_keep = []
    for _ in range(data.shape[0]):
        if diff_pcts[_] > diff_thresh:
            inds_keep.append(_)
    return data[data.index.isin(inds_keep)]


def update_colnames(filenames_data: dict, colnames: list):
    for fname, data in filenames_data.items():
        data.columns = colnames
    return filenames_data


def agg_data(filenames_data: dict, hz_baseline: int, hz_convertto: int):
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


def clip_data(filenames_data: dict, clip_percents: dict):
    for fname, data in filenames_data.items():
        clip_count_start = int(data.shape[0] * (clip_percents['start'] / 100))
        clip_count_end = data.shape[0] - int(data.shape[0] * (clip_percents['end'] / 100))
        filenames_data[fname] = data[clip_count_start:clip_count_end]
    return filenames_data


def clip_start(filenames_data, cfg):
    filenames_data_ = {}
    for fn, data in filenames_data.items():
        d_array = data[cfg['columns_model'][0]].values
        start_positive = True if d_array[0] > 0 else False
        cross_zero_ind = None
        for _, val in enumerate(d_array):
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


def get_wllevels_totaldfs(wllevels_filenames: dict, filenames_data: dict, columns_model: list, out_dir_files: str):
    levels_order = [v for v in list(wllevels_filenames.keys()) if v != 'training']
    wllevels_totaldfs = {}
    for wllevel, wllevel_filenames in wllevels_filenames.items():
        if wllevel == 'training':
            continue
        wllevel_datas = []
        for fn in wllevel_filenames:
            wllevel_datas.append(filenames_data[fn])
        wllevel_data = pd.concat(wllevel_datas, axis=0)[columns_model]
        wllevels_totaldfs[wllevel] = wllevel_data
        # save data
        path_out = os.path.join(out_dir_files, f"{wllevel}.csv")
        wllevel_data.to_csv(path_out)
    # reorder wllevels
    index_map = {v: i for i, v in enumerate(levels_order)}
    wllevels_totaldfs = dict(sorted(wllevels_totaldfs.items(), key=lambda pair: index_map[pair[0]]))
    return wllevels_totaldfs


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


def get_run_number(f):
    return int(f.split('run_')[1].replace('.csv', '').replace('0', ''))


def prep_nasa(df):
    df.drop(index=df.index[0], axis=0, inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    altitude_base = float(df['ALTITUDE'].values[0])
    altitudes_normed = [v - altitude_base for v in df['ALTITUDE'].values]
    df['ALTITUDE'] = pd.Series(altitudes_normed)
    return df


def save_config(cfg: dict, yaml_path: str) -> dict:
    """
    Purpose:
        Save config to path
    Inputs:
        cfg
            type: dict
            meaning: config (yaml)
        yaml_path
            type: str
            meaning: .yaml path to save to
    Outputs:
        cfg
            type: dict
            meaning: config (yaml) -- saved
    """
    with open(yaml_path, 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)
    return cfg


def make_traintest_files_nasa(dir_nasa, features_model, hz_baseline=16, hz_convertto=6.67, features_meta=['RUN_rpt1', 'TIME', 'ALTITUDE']):
    path_runsdata = os.path.join(dir_nasa, 'runs_metadata.csv')
    runsdata = pd.read_csv(path_runsdata)
    # set type for each run
    ## drop --> those with 'Compensation'==0 if 'Delay (msec)'>0
    ## train --> those with 'Failure'==0 AND 'Compensation'==1 if 'Delay (msec)'>0
    ## test --> those with 'Failure'!=0 AND 'Compensation'==1 if 'Delay (msec)'>0
    runs_types = {}
    testruns_altitudes = {}
    for _, row in runsdata.iterrows():
        # check for drop
        if row['Delay (msec)'] > 0 and row['Compensation'] == 0:
            runs_types[row['Run']] = 'drop'
        # check for train or test
        elif row['Failure'] == 0:
            if row['Delay (msec)'] > 0 and row['Compensation'] == 0:
                continue
            else:
                runs_types[row['Run']] = 'train'
        else:
            if row['Delay (msec)'] > 0 and row['Compensation'] == 0:
                continue
            else:
                runs_types[row['Run']] = 'test'
    # get altitude of engine failure for all test runs
    for run, run_type in runs_types.items():
        if run_type == 'test':
            row = runsdata[runsdata['Run'] == run].iloc[0]
            testruns_altitudes[run] = row['Altitude']
    subjects_testfiles_wltogglepoints = {}
    dirs_subj = [os.path.join(dir_nasa, f) for f in os.listdir(dir_nasa) if os.path.isdir(os.path.join(dir_nasa, f))]
    runs_train = [r for r in runs_types if runs_types[r] == 'train']
    runs_test = [r for r in runs_types if runs_types[r] == 'test']
    features = features_meta + features_model
    # for each subject
    for dir_subj in dirs_subj:
        subj = dir_subj.split('/')[-1]
        subjects_testfiles_wltogglepoints[subj] = {}
        dir_subj_date = \
        [os.path.join(dir_subj, f) for f in os.listdir(dir_subj) if os.path.isdir(os.path.join(dir_subj, f))][0]
        paths_train = [os.path.join(dir_subj_date, f) for f in os.listdir(dir_subj_date) if '_run_' in f and get_run_number(f) in runs_train]
        paths_test = [os.path.join(dir_subj_date, f) for f in os.listdir(dir_subj_date) if '_run_' in f and get_run_number(f) in runs_test]
        # make 1 train.csv using all runs combined
        dfs_train = [pd.read_csv(p) for p in paths_train]
        df_train = pd.concat(dfs_train, axis=0)[features]
        df_train = prep_nasa(df_train)
        path_train = os.path.join(dir_subj, 'train.csv')
        df_train.to_csv(path_train, index=False)
        # make test_x.csv files for all runs
        for p in paths_test:
            df_test = pd.read_csv(p)[features]
            df_test = prep_nasa(df_test)
            # agg data to 6.67 Hz
            agg = int(hz_baseline / hz_convertto)
            df_test = df_test.groupby(df_test.index // agg).mean()
            altitude_enginefail = testruns_altitudes[get_run_number(p)]
            # get timestep_enginefail
            for ind,altitude in enumerate(df_test['ALTITUDE']):
                if altitude > altitude_enginefail:
                    timestep_enginefail = ind
                    break
            filename = p.replace(dir_subj_date,'')[1:]
            subjects_testfiles_wltogglepoints[subj][filename] = {}
            subjects_testfiles_wltogglepoints[subj][filename]['time_total'] = len(df_test)
            subjects_testfiles_wltogglepoints[subj][filename]['times_wltoggle'] = [timestep_enginefail]
            path_test = os.path.join(dir_subj, f'{filename}')
            df_test.to_csv(path_test, index=False)
    path_cfg = os.path.join(dir_nasa, 'subjects_testfiles_wltogglepoints.yaml')
    save_config(subjects_testfiles_wltogglepoints, path_cfg)


if __name__ == '__main__':
    dir_nasa = "/Users/samheiserman/Desktop/PhD/paper3 - driving sim (real-time)/NASA"
    features_meta = [
        'RUN_rpt1',
        'TIME',
        'ALTITUDE'
    ]
    features_model = [
        'ROLL_STICK',
        'ROLLSTK',
        'ROLL',
        'PITCH_STIC',
        'PITCHSTK',
        'PITCH',
        'RUDDER_PED',
        'X_MOTION',
        'Y_MOTION',
        'Z_MOTION',
    ]
    make_traintest_files_nasa(dir_nasa, features_model, hz_baseline=16, hz_convertto=6.67, features_meta=features_meta)