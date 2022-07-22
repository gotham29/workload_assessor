import os
import sys

import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.model.model import train_models
from source.utils.utils import get_args, load_config, load_files, make_dirs
from source.preprocess.preprocess import update_colnames, agg_data, get_dftrain, clip_data, get_ttypesdf
from source.analyze.plot import plot_data, plot_boxplot
from source.analyze.anomaly import get_testtypes_anomscores, get_ttypesdiffs

FILETYPES_READFUNCS = {
    'xls': pd.read_excel,
    'csv': pd.read_csv
}


def run_subject(subj, dir_input, dir_output, htm_config, colnames, columns_model, file_type, read_func, hz_baseline,
                hz_convertto, testtypes_filenames, clip_percents):
    # Load data
    filenames_data = load_files(dir_input=dir_input, file_type=file_type, read_func=read_func)
    # Update columns
    filenames_data = update_colnames(filenames_data=filenames_data, colnames=colnames)
    # Agg data
    filenames_data = agg_data(filenames_data=filenames_data, hz_baseline=hz_baseline, hz_convertto=hz_convertto)
    # Clip data
    filenames_data = clip_data(filenames_data=filenames_data, clip_percents=clip_percents)
    # Plot data
    plot_data(filenames_data=filenames_data, file_type=file_type, dir_output=dir_output)
    # Train models
    df_train = get_dftrain(testtypes_filenames=testtypes_filenames, filenames_data=filenames_data,
                           columns_model=columns_model, dir_output=dir_output)
    features_models = train_models(df_train, dir_output=dir_output, htm_config=htm_config)
    # Get WL results (levels 0-3)
    testtypes_anomscores = get_testtypes_anomscores(subj=subj, htm_config=htm_config, features_models=features_models,
                                                    columns_model=columns_model, filenames_data=filenames_data,
                                                    testtypes_filenames=testtypes_filenames, dir_output=dir_output)
    # Write outputs
    plot_boxplot(data_plot=testtypes_anomscores,
                 title=f'Anomscores by WL Level\nhz={hz_convertto},; features={columns_model}',
                 outpath=os.path.join(dir_output, 'anomaly', 'testtypes_boxplots.png'), xlabel='WL Levels', ylabel='WL')
    # Get WL diffs btwn test types
    ttypesdiffs = get_ttypesdiffs(testtypes_anomscores=testtypes_anomscores)
    return ttypesdiffs, filenames_data


def main(config):
    subjects = [f for f in os.listdir(config['dirs']['input']) if
                os.path.isdir(os.path.join(config['dirs']['input'], f))]
    meta_data = f"HZ={config['hzs']['convertto']}; FEATURES={config['columns_model']}"

    print(f"{meta_data}")
    print(f"Subjects Found = {len(subjects)}")

    subjects_ttypesdiffs = []
    subjects_filenamesdata = {}
    for subj in subjects:
        print(f"\n subj = {subj}")
        dir_input_subj = os.path.join(config['dirs']['input'], subj)
        dir_output_subj = os.path.join(config['dirs']['output'], subj)
        dir_output_subj = make_dirs(dir_input_subj, dir_output_subj, subj, meta_data)
        subj_ttypesdiffs, filenames_data = run_subject(subj=subj,
                                                       dir_input=dir_input_subj,
                                                       dir_output=dir_output_subj,
                                                       htm_config=config['htm_config'],
                                                       colnames=config['colnames'],
                                                       columns_model=config['columns_model'],
                                                       file_type=config['file_type'],
                                                       read_func=config['read_func'],
                                                       hz_baseline=config['hzs']['baseline'],
                                                       hz_convertto=config['hzs']['convertto'],
                                                       testtypes_filenames=config['testtypes_filenames'],
                                                       clip_percents=config['clip_percents'])
        subj_df = get_ttypesdf(subj, subj_ttypesdiffs)
        subjects_ttypesdiffs.append(subj_df)
        subjects_filenamesdata[subj] = filenames_data

    df_ttypesdiffs = pd.concat(subjects_ttypesdiffs, axis=0)
    path_out = os.path.join(config['dirs']['output'], f"subjects_ttypesdiffs--{','.join(subjects)}--{meta_data}.csv")
    df_ttypesdiffs.to_csv(path_out)


if __name__ == '__main__':
    config = load_config(get_args().config_path)
    config['read_func'] = FILETYPES_READFUNCS[config['file_type']]
    main(config)
