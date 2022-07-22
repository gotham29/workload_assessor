import argparse
import os

import yaml
from htm_source.utils.utils import load_pickle_object_as_data, save_data_as_pickle


# from yaml import Loader


def get_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config_path', required=True,
                        help='path to config')
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


def save_models(features_models, dir_output):
    dir_out = os.path.join(dir_output, 'models')
    make_dir(dir_out)
    print('  save models...')
    for feat, mod in features_models.items():
        path_out = os.path.join(dir_out, f"{feat}.pkl")
        save_data_as_pickle(mod, path_out)
        print(f"    {feat}")


# def load_pickle_object_as_data(file_path):
#     """
#     Loads a pickle object from path
#     :param file_path: file path to load pickle object from
#     :return: Returns object
#     """
#     with open(file_path, 'rb') as f_handle:
#         data = pickle.load(f_handle)
#     return data


def load_models(dir_models):
    """
    Purpose:
        Load pkl models for each feature from dir
    Inputs:
        dir_models
            type: str
            meaning: path to dir where pkl models are loaded from
    Outputs:
        features_models
            type: dict
            meaning: model obj for each feature
    """
    pkl_files = [f for f in os.listdir(dir_models) if '.pkl' in f]
    print(f"\nLoading {len(pkl_files)} models...")
    features_models = {}
    for f in pkl_files:
        pkl_path = os.path.join(dir_models, f)
        model = load_pickle_object_as_data(pkl_path)
        features_models[f.replace('.pkl', '')] = model
    return features_models


def make_dir(mydir):
    if not os.path.exists(mydir):
        os.mkdir(mydir)


def load_config(yaml_path):
    """
    Purpose:
        Load config from path
    Inputs:
        yaml_path
            type: str
            meaning: .yaml path to load from
    Outputs:
        cfg
            type: dict
            meaning: config (yaml) -- loaded
    """
    with open(yaml_path, 'r') as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return cfg


def make_dirs(dir_input_subj, dir_output_subj, subj, meta_data):
    make_dir(dir_input_subj)
    make_dir(dir_output_subj)
    dir_output_subj = os.path.join(dir_output_subj, meta_data)
    make_dir(dir_output_subj)
    return dir_output_subj
