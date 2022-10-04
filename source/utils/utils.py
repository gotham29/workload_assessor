import argparse
import sys
import os

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_TS_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ts_forecaster')
_HTM_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'htm_streamer')

sys.path.append(_SOURCE_DIR)
sys.path.append(_TS_SOURCE_DIR)
sys.path.append(_HTM_SOURCE_DIR)

from ts_source.utils.utils import make_dir, load_config
from htm_source.utils.fs import load_models, load_pickle_object_as_data, save_data_as_pickle


def get_args(
    args_add = [{'name_abbrev': '-cp',
                'name': '--config_path',
                'required': True,
                'help': 'path to config'}]
    ):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for arg_add in args_add:
        parser.add_argument(arg_add['name_abbrev'], arg_add['name'],
                            required=arg_add['required'], help=arg_add['help'])
    # parser.add_argument('-cp', '--config_path', required=True,
    #                     help='path to config')
    args = parser.parse_args()
    return args


def load_files(dir_input: str, file_type: str, read_func: str):
    filenames_data = {}
    fdir_allfiles = [f for f in os.listdir(dir_input)]
    fdir_typefiles = [f for f in fdir_allfiles if f.split('.')[-1] == file_type]
    for f in fdir_typefiles:
        d_path = os.path.join(dir_input, f)
        data = read_func(d_path)
        filenames_data[f] = data
    return filenames_data


def make_dirs_subj(config_dirs, meta_data, subj, outputs=['anomaly','data_files','data_plots','models','scalers']):
    dir_subj_in = os.path.join(config_dirs['input'], subj)
    dir_subj_out = os.path.join(config_dirs['output'], subj)
    make_dir(dir_subj_out)
    dir_meta = os.path.join(dir_subj_out, meta_data)
    make_dir(dir_meta)
    for out in outputs:
        dir_out = os.path.join(dir_meta, out)
        make_dir( dir_out )
    return dir_subj_in, dir_meta


