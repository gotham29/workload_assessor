import argparse
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


def get_args(
        args_add=[{'name_abbrev': '-cp',
                   'name': '--config_path',
                   'required': True,
                   'help': 'path to config'}]
):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for arg_add in args_add:
        parser.add_argument(arg_add['name_abbrev'], arg_add['name'],
                            required=arg_add['required'], help=arg_add['help'])
    args = parser.parse_args()
    return args


def combine_dicts(dicts):
    d1 = dicts[0]
    d_comb = {k: [] for k in d1}
    for d_ in dicts:
        for k, v in d_.items():
            d_comb[k] += v
    return d_comb


def load_files(dir_input: str, file_type: str, read_func: str):
    filenames_data = {}
    fdir_allfiles = [f for f in os.listdir(dir_input)]
    fdir_typefiles = [f for f in fdir_allfiles if f.split('.')[-1] == file_type]
    for f in fdir_typefiles:
        d_path = os.path.join(dir_input, f)
        data = read_func(d_path).replace(np.nan, 0, inplace=False)
        filenames_data[f] = data
    return filenames_data


def make_dirs_subj(dir_out, outputs=['anomaly', 'data_files', 'data_plots', 'models']):
    outnames_dirs = {}
    os.makedirs(dir_out, exist_ok=True)
    for out in outputs:
        dir_out_type = os.path.join(dir_out, out)
        os.makedirs(dir_out_type, exist_ok=True)
        outnames_dirs[out] = dir_out_type
    return outnames_dirs


def print_realtimewl_config(path_in, file_type='xls', runs=[1,2,3,4,5,6,7,8,9]):
    df = pd.read_csv(path_in)
    # loop over subjects
    for subj, df_subj in df.groupby('Subject'):
        # loop over runs
        print(f"\n  {subj}:")
        for run in runs:
            col_timetotal = f"Run {run}-Total"
            col_time1 = f"Run {run}-1"
            col_time2 = f"Run {run}-2"
            time_total = df_subj[col_timetotal].values[0]
            time_1 = df_subj[col_time1].values[0]
            time_2 = df_subj[col_time2].values[0]
            print(f"    realtime{run}.{file_type}:")
            print(f"      time_total: {time_total}")
            print("      times_wltoggle:")
            print(f"        - {time_1}")
            print(f"        - {time_2}")

