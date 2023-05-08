import argparse
import os
import sys

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
        data = read_func(d_path)
        filenames_data[f] = data
    return filenames_data


def make_dirs_subj(dir_out, outputs=['anomaly', 'data_files', 'data_plots', 'models']):
    os.makedirs(dir_out, exist_ok=True)
    for out in outputs:
        dir_out_type = os.path.join(dir_out, out)
        os.makedirs(dir_out_type, exist_ok=True)

