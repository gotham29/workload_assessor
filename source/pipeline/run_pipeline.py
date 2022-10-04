import os
import sys
import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_TS_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ts_forecaster')
_HTM_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'htm_streamer')

sys.path.append(_SOURCE_DIR)
sys.path.append(_TS_SOURCE_DIR)
sys.path.append(_HTM_SOURCE_DIR)

from source.model.model import train_save_models
from source.utils.utils import get_args, load_config, load_files, make_dirs_subj
from source.preprocess.preprocess import update_colnames, agg_data, get_dftrain, clip_data, get_ttypesdf
from source.analyze.plot import plot_data, plot_boxes, plot_lines, plot_bars
from source.analyze.anomaly import get_testtypes_outputs, get_testtypes_diffs
from ts_source.utils.utils import make_dir, add_timecol, load_models as load_models_darts
from htm_source.utils.fs import load_models as load_models_htm

FILETYPES_READFUNCS = {
    'xls': pd.read_excel,
    'csv': pd.read_csv
}


def norm_data(filenames_data, wllevels_filenames, time_col='timestamp'):
    filenames_wllevels = {}
    wllevels_means = {}
    for wl, filenames in wllevels_filenames.items():
        dfs = [filenames_data[f] for f in filenames]
        df = pd.concat(dfs, axis=0)
        feats_means = {feat: m for feat, m in dict(df.mean()).items() if feat != time_col}
        wllevels_means[wl] = feats_means
        for f in filenames:
            filenames_wllevels[f] = wl
    for fn, data in filenames_data.items():
        wl = filenames_wllevels[fn]
        feats_means = wllevels_means[wl]
        for feat, mean in feats_means.items():
            data[feat] = data[feat]-mean
    return filenames_data  #filenames_data_normed


def run_subject(config, dir_input, dir_output, wllevels_tlx, time_col='timestamp'):
    # Load data
    filenames_data = load_files(dir_input=dir_input, file_type=config['file_type'], read_func=config['read_func'])
    # Update columns
    filenames_data = update_colnames(filenames_data=filenames_data, colnames=config['colnames'])
    # Agg data
    filenames_data = agg_data(filenames_data=filenames_data, hz_baseline=config['hzs']['baseline'],
                              hz_convertto=config['hzs']['convertto'])
    # Clip data
    filenames_data = clip_data(filenames_data=filenames_data, clip_percents=config['clip_percents'])
    # Norm data
    filenames_data = norm_data(filenames_data=filenames_data, wllevels_filenames=config['testtypes_filenames'],
                               time_col=time_col)
    # Plot data
    plot_data(filenames_data=filenames_data, file_type=config['file_type'], dir_output=dir_output)
    # Train models
    df_train = get_dftrain(wllevels_filenames=config['testtypes_filenames'], filenames_data=filenames_data,
                           columns_model=config['columns_model'], dir_output=dir_output)
    # Add timecol
    df_train = add_timecol(df_train, time_col)
    # Train model(s)
    if config['train_models']:
        features_models = train_save_models(df_train, alg=config['alg'], dir_output=dir_output, config=config,
                                            htm_config=config['htm_config'])
    # Load model(s)
    dir_models = os.path.join(dir_output, 'models')
    if config['alg'] == 'HTM':
        features_models = load_models_htm(dir_models)
    else:
        features_models = load_models_darts(dir_models, alg=config['alg'])
    # Get WL results (levels 0-3)
    wllevels_anomscores, wllevels_predcounts, wllevels_alldata = get_testtypes_outputs(alg=config['alg'],
                                                                                       htm_config=config[
                                                                                           'htm_config'],
                                                                                       dir_output=dir_output,
                                                                                       columns_model=config[
                                                                                           'columns_model'],
                                                                                       filenames_data=filenames_data,
                                                                                       features_models=features_models,
                                                                                       testtypes_filenames=config[
                                                                                           'testtypes_filenames'])
    # Write outputs
    print(f"  Writing outputs to --> {dir_output}")
    print("    Boxplots...")
    plot_boxes(data_plot1=wllevels_anomscores,
                  data_plot2=wllevels_predcounts,
                  title_1=f"Anomaly Scores by WL Level\nhz={config['hzs']['convertto']},; features={config['columns_model']}",
                  title_2=f"Prediction Counts by WL Level\nhz={config['hzs']['convertto']},; features={config['columns_model']}",
                  out_dir=os.path.join(dir_output, 'anomaly'),
                  xlabel='MWL Levels',
                  ylabel='HTM Metric')
    print("    Lineplots...")
    plot_lines(wllevels_anomscores=wllevels_anomscores,
                   wllevels_predcounts=wllevels_predcounts,
                   wllevels_alldata=wllevels_alldata,
                   columns_model=config['columns_model'],
                   out_dir=os.path.join(dir_output, 'anomaly'))
    print("    Barplots...")
    plot_bars(wllevels_tlx=wllevels_tlx,
              title='NASA TLX vs Task WL',
              xlabel='Task WL',
              ylabel='NASA TLX',
              out_dir=os.path.join(dir_output, 'anomaly'))
    # Get WL diffs btwn testtypes
    wllevels_diffs = get_testtypes_diffs(testtypes_anomscores=wllevels_anomscores)
    return wllevels_anomscores, wllevels_diffs, filenames_data


def main(config):
    # Run WL assessor for all subjects found
    subjects = [f for f in os.listdir(config['dirs']['input']) if
                os.path.isdir(os.path.join(config['dirs']['input'], f))]
    meta_data = f"ALG={config['alg']}; HZ={config['hzs']['convertto']}; FEATURES={config['columns_model']}"
    print(f"{meta_data}")
    print(f"Subjects Found = {len(subjects)}")
    subjects_ttypesdiffs = []
    subjects_filenamesdata = {}
    subjects_ttypesascores = {}
    for subj in subjects:
        print(f"\n subj = {subj}")
        dir_input_subj, dir_output_subj = make_dirs_subj(config['dirs'], meta_data, subj)
        ttypes_ascores, ttypes_diffs, filenames_data = run_subject(config=config,
                                                                   dir_input=dir_input_subj,
                                                                   dir_output=dir_output_subj,
                                                                   wllevels_tlx=config['subjects_wllevels_tlx'][subj])
        df_ttypesdiffs = get_ttypesdf(subj, ttypes_diffs)
        subjects_ttypesdiffs.append(df_ttypesdiffs)
        subjects_filenamesdata[subj] = filenames_data
        subjects_ttypesascores[subj] = ttypes_ascores
    df_ttypesdiffs = pd.concat(subjects_ttypesdiffs, axis=0)
    path_out = os.path.join(config['dirs']['output'], f"subjects_ttypesdiffs--{','.join(subjects)}--{meta_data}.csv")
    df_ttypesdiffs.to_csv(path_out)


if __name__ == '__main__':
    args_add = [{'name_abbrev': '-cp',
                 'name': '--config_path',
                 'required': True,
                 'help': 'path to config'}]
    config = load_config(get_args(args_add).config_path)
    # print(f"\nCONFIG")
    # for k, v in config.items():
    #     print(f"  {k}")
    #     try:
    #         for k_, v_ in v.items():
    #             print(f"    {k_} = {v_}")
    #             try:
    #                 for k__, v__ in v_.items():
    #                     print(f"    {k__} = {v__}")
    #             except:
    #                 print(f"    {v_}")
    #     except:
    #         print(f"    {v}")
    # config = validate_config(config)
    config['read_func'] = FILETYPES_READFUNCS[config['file_type']]
    main(config)
