import copy
import itertools
import os
import sys
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_TS_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ts_forecaster')
_HTM_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'htm_streamer')

sys.path.append(_SOURCE_DIR)
sys.path.append(_TS_SOURCE_DIR)
sys.path.append(_HTM_SOURCE_DIR)

from source.model.model import train_save_models
from source.utils.utils import get_args, load_files, make_dirs_subj, combine_dicts
from source.preprocess.preprocess import update_colnames, preprocess_data, get_wllevels_totaldfs, get_wllevels_indsend
from source.analyze.tlx import make_boxplots, get_tlx_overlaps
from source.analyze.plot import make_data_plots, plot_outputs_boxes, plot_outputs_lines, plot_outputs_bars, get_accum, \
    plot_write_data
from source.analyze.anomaly import get_ascores_entropy, get_ascores_naive, \
    get_ascores_pyod, get_ascore_pyod, get_subjects_wldiffs, get_clscores, get_ascore_entropy, get_entropy_ts, get_ascore_naive
from ts_source.utils.utils import add_timecol, load_config, load_models as load_models_darts

from ts_source.model.model import get_model_lag, LAG_MIN, get_modname, get_preds_rolling
from htm_source.pipeline.htm_batch_runner import run_batch
from htm_source.utils.fs import load_models as load_models_htm, save_models

FILETYPES_READFUNCS = {
    'xls': pd.read_excel,
    'csv': pd.read_csv
}


def get_wllevels_outputs(filenames_ascores, filenames_pcounts, filenames_wllevels, wllevels_filenames, levels_order):
    wllevels_ascores = {wllevel: [] for wllevel in wllevels_filenames if wllevel != 'training'}
    wllevels_pcounts = {wllevel: [] for wllevel in wllevels_filenames if wllevel != 'training'}

    # get htm anomaly scores - filenames
    for fn, ascores in filenames_ascores.items():
        wllevel = filenames_wllevels[fn]
        wllevels_ascores[wllevel] += ascores
        if fn in filenames_pcounts:
            wllevels_pcounts[wllevel] += filenames_pcounts[fn]

    # get total WL - levels
    wllevels_coeffs = {wllevel: len(wllevels_filenames['baseline']) / len(fileslist) for wllevel, fileslist in
                       wllevels_filenames.items()}
    print(f"  ** wllevels_coeffs = {wllevels_coeffs}\n")
    wllevels_totalascores = {wllevel: np.sum(ascores) * wllevels_coeffs[wllevel] for wllevel, ascores in
                             wllevels_ascores.items()}

    # sort - levels
    index_map = {v: i for i, v in enumerate(levels_order)}
    wllevels_totalascores = dict(sorted(wllevels_totalascores.items(), key=lambda pair: index_map[pair[0]]))

    return wllevels_ascores, wllevels_pcounts, wllevels_totalascores


def get_filenames_outputs(cfg,
                          modname,
                          filenames_data,
                          modname_model):
    filenames_ascores = {}
    filenames_pcounts = {}

    for fn, data in filenames_data.items():

        if cfg['alg'] == 'HTM':
            # if model_for_each_feature - drop from cfg_htm_user['features'] all but modname & 'timestamp'
            cfg_htm_user = {k: v for k, v in cfg['htm_config_user'].items()}
            if cfg['htm_config_user']['models_state']['model_for_each_feature']:  # 'megamodel' not in modname:
                feats_model = [modname, config['time_col']]
                cfg_htm_user['features'] = {k: v for k, v in cfg_htm_user['features'].items() if k in feats_model}

            feats_models, features_outputs = run_batch(cfg_user=cfg_htm_user,
                                                       cfg_model=cfg['htm_config_model'],
                                                       config_path_user=None,
                                                       config_path_model=None,
                                                       learn=cfg['learn_in_testing'],
                                                       data=data,
                                                       iter_print=1000,
                                                       features_models=modname_model)
            ascores = features_outputs[modname]['anomaly_score']
            filenames_pcounts[fn] = features_outputs[modname]['pred_count']

        elif config['alg'] == 'SteeringEntropy':
            ascores = get_ascores_entropy(data[modname].values)

        elif config['alg'] == 'Naive':
            ascores = get_ascores_naive(data[modname].values)

        elif config['alg'] == 'IPSD':
            Pxx, freqs = plt.psd(x=data[modname].values, Fs=cfg['hzs']['convertto'])  # NFFT=1024
            total_power = np.trapz(Pxx, freqs)
            ascores = [total_power]

        elif config['alg'] == 'FPSD':
            Pxx, freqs = plt.psd(x=data[modname].values, Fs=cfg['hzs']['convertto'])  # NFFT=1024
            highest_peak_freq = freqs[np.argmax(Pxx)]
            ascores = [highest_peak_freq]

        elif config['alg'] in ['IForest', 'OCSVM', 'KNN', 'LOF', 'AE', 'VAE', 'KDE']:
            ascores = get_ascores_pyod(data[modname], modname_model[
                modname])  # data[cfg['columns_model']], features_models[cfg['columns_model'][0]]

        # else:  # ts_source alg
        #     for feat, model in features_models.items():  # Assumes single model
        #         break
        #     mod_name = get_modname(model)
        #     features = model.training_series.components
        #     preds = get_preds_rolling(model=model,
        #                               df=data,
        #                               features=features,
        #                               LAG=max(LAG_MIN, get_model_lag(mod_name, model)),
        #                               time_col=cfg['time_col'],
        #                               forecast_horizon=cfg['forecast_horizon'])
        #     preds_df = pd.DataFrame(preds, columns=list(features))
        #     data_ = data.tail(preds_df.shape[0])
        #     ascores = list(abs(data_[cfg['columns_model'][0]].values - preds_df[cfg['columns_model'][
        #         0]].values))  # list(abs(data_['steering angle'].values - preds_df['steering angle'].values))

        filenames_ascores[fn] = ascores

    return filenames_ascores, filenames_pcounts


def run_subject(cfg, modname, df_train, dir_out, filenames_data, filenames_wllevels, wllevels_filenames, modname_model,
                levels_order):
    # get columns_model
    columns_model = [modname]
    if 'megamodel' in modname:
        columns_model = cfg['columns_model']
    # make subj dirs
    outnames_dirs = make_dirs_subj(dir_out, outputs=['anomaly', 'data_files', 'data_plots', 'models'])
    # save models
    save_models(modname_model, outnames_dirs['models'])
    # plot filenames_data
    make_data_plots(filenames_data=filenames_data, modname=modname, columns_model=columns_model,
                    file_type=cfg['file_type'], out_dir_plots=outnames_dirs['data_plots'])
    # remove training data from filenames_data - assumes not files in training are repeated at other level(s)
    filenames_data = {fn: data for fn, data in filenames_data.items() if
                      fn not in cfg['wllevels_filenames']['training']}
    # get behavior data (dfs) for all wllevels
    wllevels_totaldfs = get_wllevels_totaldfs(wllevels_filenames=wllevels_filenames,  # cfg['wllevels_filenames'],
                                              filenames_data=filenames_data,
                                              columns_model=columns_model,
                                              out_dir_files=outnames_dirs['data_files'])
    # plot levels data
    plot_write_data(df_train, out_name='training', out_dir_plots=outnames_dirs['data_plots'],
                    out_dir_files=outnames_dirs['data_files'])
    for wllevel, df in wllevels_totaldfs.items():
        plot_write_data(df, out_name=wllevel, out_dir_plots=outnames_dirs['data_plots'],
                        out_dir_files=outnames_dirs['data_files'])

    # get indicies dividing the wllevels
    # wllevels_indsend = get_wllevels_indsend(wllevels_totaldfs)

    # set valid colors for plots
    colors = ['grey', 'yellow', 'orange', 'red', 'blue', 'green', 'cyan', 'magenta', 'black']
    assert len(wllevels_totaldfs) <= len(
        colors), f"more wllevels found ({len(wllevels_totaldfs)}) than colors ({len(colors)})"
    levels_colors = {}
    for _, wllevel in enumerate(wllevels_totaldfs):
        levels_colors[wllevel] = colors[_]

    # get model outputs for each 'static' file
    filenames_ascores, filenames_pcounts = get_filenames_outputs(cfg=cfg,
                                                                 modname=modname,
                                                                 filenames_data=filenames_data,
                                                                 modname_model=modname_model)
    # agg ascore to wllevel
    wllevels_ascores, wllevels_pcounts, wllevels_totalascores = get_wllevels_outputs(
        filenames_ascores=filenames_ascores,
        filenames_pcounts=filenames_pcounts,
        filenames_wllevels=filenames_wllevels,
        wllevels_filenames=cfg['wllevels_filenames'],
        levels_order=levels_order)
    # write outputs
    print(f"  Writing outputs to --> {outnames_dirs['anomaly']}")
    print("    Boxplots...")
    make_boxplots(data_dict=wllevels_ascores,
                  levels_colors=levels_colors,
                  xlabel='WL Levels',
                  ylabel=f"{cfg['alg']} WL",
                  title=f"{cfg['alg']} WL Scores by Run Mode",
                  suptitle=None,
                  path_out=os.path.join(outnames_dirs['anomaly'], "wllevels--aScoreTotals--box.png"),
                  ylim=None)
    plot_outputs_bars(mydict=wllevels_totalascores,
                      levels_colors=levels_colors,
                      title=f"{cfg['alg']} WL Scores by Level",
                      xlabel='WL Levels',
                      ylabel=f"{cfg['alg']} WL",
                      path_out=os.path.join(outnames_dirs['anomaly'], "wllevels--aScoreTotals--bar.png"),
                      xtickrotation=0)
    plot_outputs_boxes(data_plot1=wllevels_ascores,
                       data_plot2=wllevels_pcounts,
                       levels_colors=levels_colors,
                       title_1=f"Anomaly Scores by WL Level\nhz={cfg['hzs']['convertto']},; features={columns_model}",
                       title_2=f"Prediction Counts by WL Level\nhz={cfg['hzs']['convertto']},; features={columns_model}",
                       out_dir=outnames_dirs['anomaly'],
                       xlabel='WL Levels',
                       ylabel='HTM Metric')
    print("    Lineplots...")
    plot_outputs_lines(wllevels_ascores_=wllevels_ascores,
                       wllevels_pcounts_=wllevels_pcounts,
                       # wllevels_indsend=wllevels_indsend,
                       columns_model=columns_model,
                       filenames_data=filenames_data,
                       filenames_ascores=filenames_ascores,
                       filenames_wllevels=filenames_wllevels,
                       get_pcounts=True if cfg['alg'] == 'HTM' else False,
                       levels_order=levels_order,
                       levels_colors=levels_colors,
                       out_dir=outnames_dirs['anomaly'])

    return filenames_ascores, wllevels_ascores, wllevels_totalascores, levels_colors


def get_scores(subjects_wldiffs, subjects_wllevels_totalascores):
    # percent_change_from_baseline
    percent_change_from_baseline = round(sum(subjects_wldiffs.values()), 3)
    # subjects_wldiffs_positive
    subjects_wldiffs_positive = {subj: diff for subj, diff in subjects_wldiffs.items() if diff > 0}
    # subjs_baseline_lowest
    subjs_baseline_lowest = []
    for subj, wllevels_totalascores in subjects_wllevels_totalascores.items():
        ascorestotal_baseline = wllevels_totalascores['baseline']
        baseline_lowest = True
        for wllevel, totalascore in wllevels_totalascores.items():
            if wllevel == 'baseline':
                continue
            if totalascore <= ascorestotal_baseline:
                baseline_lowest = False
        if baseline_lowest:
            subjs_baseline_lowest.append(subj)
    return percent_change_from_baseline, list(subjects_wldiffs_positive.keys()), subjs_baseline_lowest


def rename_dirs_by_scores(subjects_wldiffs, subjects_increased_from_baseline, subjects_baseline_lowest, dir_out):
    for subj in subjects_wldiffs:
        dir_output_subj = os.path.join(dir_out, subj)
        subj_score_mark = ''
        if subj in subjects_increased_from_baseline:
            subj_score_mark += '*'
        if subj in subjects_baseline_lowest:
            subj_score_mark += '*'
        dir_output_subj_scored = dir_output_subj.replace(f"/{subj}", f"/{subj_score_mark}{subj}")
        os.rename(dir_output_subj, dir_output_subj_scored)


def run_posthoc(cfg, dir_out, subjects_filenames_data, subjects_dfs_train, subjects_features_models):
    # gather WL results
    subj = list(subjects_filenames_data.keys())[0]
    modnames = list(subjects_features_models[subj].keys())
    print(f'\nrun_posthoc; modnames = {modnames}')
    modnames_percentchangesfrombaseline = {}

    # make filename --> wllevel mapping
    filenames_wllevels = {}
    for level, fns in cfg['wllevels_filenames'].items():
        if level == 'training':
            continue
        for fn in fns:
            filenames_wllevels[fn] = level

    # loop over modnames
    for modname in modnames:
        dir_out_modname = os.path.join(dir_out, f"modname={modname}")
        os.makedirs(dir_out_modname, exist_ok=True)
        print(f"  --> {dir_out_modname.replace(dir_out, '')}")

        # run modname for all data
        percent_change_from_baseline = run_modname(modname, cfg, filenames_wllevels, cfg['wllevels_filenames'],
                                                   subjects_dfs_train, subjects_filenames_data,
                                                   subjects_features_models, dir_out_modname)
        modnames_percentchangesfrombaseline[modname] = percent_change_from_baseline

    mean_percentchangefrombaseline = np.mean(list(modnames_percentchangesfrombaseline.values()))
    return mean_percentchangefrombaseline


def get_entropy(errors_in_window, alpha):
    inds_bins = {
        1: [-np.inf, -5*alpha],
        2: [-5*alpha, -2.5*alpha],
        3: [-2.5*alpha, -1*alpha],
        4: [-1*alpha, -0.5*alpha],
        5: [-0.5*alpha, 0.5*alpha],
        6: [0.5*alpha, 1*alpha],
        7: [1*alpha, 2.5*alpha],
        8: [2.5*alpha, 5*alpha],
        9: [5*alpha, np.inf],
    }
    inds_errors = {}
    for i, bin in inds_bins.items():
        inds_errors[i] = [v for v in errors_in_window if v>bin[0] and v<bin[1]]
    inds_props = {}
    for i, errors in inds_errors.items():
        inds_props[i] = len(errors) / len(errors_in_window)
    entropy = 0
    for i, prop in inds_props.items():
        if prop == 0:
            continue
        i_entropy = -prop * math.log(prop,9)  ##−pi × Log9(pi)
        entropy += i_entropy
    return entropy


def get_ascore_fessonia(pred_errors, alpha, ind, window_errors, window_alpha):
    if ind < window_errors:
        aScore = 0
    else:
        start_i = ind-window_errors
        end_i = ind
        errors_in_window = pred_errors[start_i:end_i]
        if ind%window_alpha == 0:
            alpha = np.percentile(pred_errors, 90)
        aScore = get_entropy(errors_in_window, alpha)
    return aScore, alpha


def run_realtime(config, dir_out, subjects_features_models, subjects_filenames_data, subjects_features_alphas):
    subj = list(subjects_features_models.keys())[0]
    modnames = list(subjects_features_models[subj].keys())
    modnames_df1s = {}
    print(f'\nrun_realtime; modnames = {modnames}')
    # make dir - windows_ascore
    windows_ascore_str = ''
    for k,v in config['windows_ascore'].items():
        windows_ascore_str += f"{k}={v}; "
    dir_out_ = os.path.join(dir_out, windows_ascore_str)
    os.makedirs(dir_out_, exist_ok=True)

    # loop over modnames
    for modname in modnames:
        dir_out_modname = os.path.join(dir_out_, f"modname={modname}")
        os.makedirs(dir_out_modname, exist_ok=True)
        print(f"  --> {dir_out_modname}")

        # loop over subjects
        rows = list()
        for subj, features_models in subjects_features_models.items():
            print(f"\n  subj = {subj}")
            dir_out_subj = os.path.join(dir_out_modname, subj)
            os.makedirs(dir_out_subj, exist_ok=True)

            # initial alpha
            if config['alg'] == 'Fessonia':
                alpha = subjects_features_alphas[subj][modname]

            # loop over testfiles
            print(f"  \n\nsubjects_filenames_data --> {subjects_filenames_data}")
            for subj, filenames_data in subjects_filenames_data.items():
                print(f"    subj = {subj}")
                for fn, data in filenames_data.items():
                    print(f"      {fn} --> {data.shape}")

            model = features_models[modname]
            for testfile, times in config['subjects_testfiles_wltogglepoints'][subj].items():
                print(f"      testfile = {testfile}")
                data_test = subjects_filenames_data[subj][testfile]
                data_test.reset_index(drop=True, inplace=True)

                # get wl_changepoints
                wl_changepoints = [int(t / times['time_total'] * data_test.shape[0]) for t in
                                   times['times_wltoggle']]
                print(f"        wl_changepoints = {wl_changepoints}")

                # run data thru model
                aScores, pred_errors, wl_changepoints_detected = list(), list(), list()
                pred_prev = None
                for _, row in data_test.iterrows():
                    if config['alg'] == 'HTM':
                        aScore, aLikl, pCount, sPreds = model.run(features_data=dict(row), timestep=_ + 1,
                                                                  learn=config['learn_in_testing'])
                    elif config['alg'] == 'Fessonia':
                        pred_error, pred_prev = get_ascore_entropy(_, row, modname, model, data_test, pred_prev)
                        pred_errors.append(pred_error)
                        aScore, alpha = get_ascore_fessonia(pred_errors, alpha, ind=_, window_errors=20, window_alpha=100)
                    elif config['alg'] == 'Naive':
                        aScore, pred_prev = get_ascore_naive(_, row, modname, model, data_test, pred_prev)
                    else:
                        aScore = 0  #get_ascore_dndeb()
                    """
                    elif config['alg'] == 'SteeringEntropy':
                        aScore, pred_prev = get_ascore_entropy(_, row, modname, model, data_test, pred_prev)
                    elif config['alg'] in ['IForest', 'OCSVM', 'KNN', 'LOF', 'AE', 'VAE', 'KDE']:
                        aScore = get_ascore_pyod(_, data_test[config['columns_model']], model)
                    else:
                        aScore, pred_prev = get_entropy_ts(_, model, row, data_test, config, pred_prev, LAG_MIN)
                    """
                    aScores.append(aScore)
                    if _ < config['windows_ascore']['previous'] + config['windows_ascore']['recent']:
                        continue
                    wl_change_detected = is_wl_detected(aScores,
                                                        config['windows_ascore']['change_thresh_percent'],
                                                        config['windows_ascore']['recent'],
                                                        config['windows_ascore']['previous'])
                    if wl_change_detected:
                        wl_changepoints_detected.append(_)

                # score detected wl-changepoints
                print(f"        wl_changepoints_detected = {wl_changepoints_detected}")
                scores, wl_changepoints_windows = score_wl_detections(data_test.shape[0],
                                                                      wl_changepoints,
                                                                      wl_changepoints_detected,
                                                                      config['windows_ascore']['change_detection_window'])
                f1_score, precision, recall, cl_accuracy = get_clscores(scores)
                path_out = os.path.join(dir_out_subj,
                                        f"{modname}--{testfile.replace(config['file_type'], '')}--confusion_matrix.csv")  # feat
                pd.DataFrame(scores, index=[0]).to_csv(path_out, index=False)

                # plot detected wl-changepoints over changepoint windows
                feats_plot = [modname]
                if 'megamodel' in modname:
                    feats_plot = [c for c in config['columns_model'] if c != config['time_col']]
                plot_wlchangepoints(feats_plot,
                                    config['file_type'],
                                    aScores,
                                    testfile,
                                    data_test,
                                    dir_out_subj,
                                    wl_changepoints_windows,
                                    wl_changepoints_detected)
                rows.append(
                    {'subj': subj, 'modname': modname, 'testfile': testfile, 'precision': precision, 'recall': recall,
                     'cl_accuracy': cl_accuracy, 'f1_score': f1_score})
        df_f1 = pd.DataFrame(rows)
        path_out = os.path.join(dir_out_modname, "classification_scores.csv")
        df_f1.to_csv(path_out, index=False)
        modnames_df1s[modname] = df_f1

    df_f1_concat = pd.concat(modnames_df1s.values(), axis=0)
    return df_f1_concat


def run_modname(modname, cfg, filenames_wllevels, wllevels_filenames, subjects_dfs_train, subjects_filenames_data,
                subjects_features_models, dir_out):
    subjects_wllevels_ascores = {}
    subjects_wllevels_totalascores = {}
    subjects_filenames_totalascores = {}

    # loop over subjects
    for subj, filenames_data in subjects_filenames_data.items():
        dir_out_subj = os.path.join(dir_out, subj)
        modname_model = {modname: subjects_features_models[subj][modname]}
        filenames_data = {fn: data for fn, data in filenames_data.items() if 'realtime' not in fn}
        filenames_ascores, wllevels_ascores, wllevels_totalascores, levels_colors = run_subject(cfg=cfg,
                                                                                                dir_out=dir_out_subj,
                                                                                                df_train=
                                                                                                subjects_dfs_train[
                                                                                                    subj],
                                                                                                modname=modname,
                                                                                                modname_model=modname_model,
                                                                                                filenames_data=filenames_data,
                                                                                                filenames_wllevels=filenames_wllevels,
                                                                                                wllevels_filenames=wllevels_filenames,
                                                                                                levels_order=[v for v in
                                                                                                              cfg[
                                                                                                                  'wllevels_filenames']
                                                                                                              if
                                                                                                              v != 'training'])
        subjects_wllevels_ascores[subj] = wllevels_ascores
        subjects_wllevels_totalascores[subj] = wllevels_totalascores
        subjects_filenames_totalascores[subj] = {fn: np.sum(ascores) for fn, ascores in filenames_ascores.items()}

    # get normalized diffs for all subjs
    subjects_wldiffs, subjects_levels_wldiffs = get_subjects_wldiffs(subjects_wllevels_totalascores)

    # get scores
    percent_change_from_baseline, subjects_increased_from_baseline, subjects_baseline_lowest = get_scores(
        subjects_wldiffs, subjects_wllevels_totalascores)
    percent_subjects_baseline_lowest = round(100 * len(subjects_baseline_lowest) / len(subjects_wllevels_ascores), 2)

    # get overlap w/TLX
    subscales_meanoverlaps, df_overlaps = {}, pd.DataFrame()
    if os.path.exists(cfg['tlx']['path']):
        subscales_meanoverlaps, df_overlaps = get_tlx_overlaps(subjects_wllevels_totalascores,
                                                               cfg['tlx']['modes_convert'], cfg['tlx']['path'])

    # rename dirs based on scores
    rename_dirs_by_scores(subjects_wldiffs, subjects_increased_from_baseline, subjects_baseline_lowest, dir_out)

    # save results
    path_out_scores = os.path.join(dir_out, 'scores.csv')
    path_out_subjects_filenames_totalascores = os.path.join(dir_out, 'subjects_filenames_totalascores.csv')
    path_out_subjects_wllevels_totalascores = os.path.join(dir_out, 'subjects_wllevels_totalascores.csv')
    path_out_subjects_wldiffs = os.path.join(dir_out, 'subjects_wldiffs.csv')
    path_out_subjects_levels_wldiffs = os.path.join(dir_out, 'subjects_wllevels_wldiffs.csv')
    path_out_subjects_overlaps = os.path.join(dir_out, 'subjects_tlxoverlaps.csv')
    scores = {'Total sensitivity to increased task demands': percent_change_from_baseline,
              'Rate of subjects with baseline lowest': percent_subjects_baseline_lowest}
    for subscale, meanoverlap in subscales_meanoverlaps.items():
        scores[subscale] = meanoverlap
    scores = pd.DataFrame(scores, index=[0])
    df_subjects_levels_wldiffs = pd.DataFrame(subjects_levels_wldiffs).T
    df_subjects_filenames_totalascores = pd.DataFrame(subjects_filenames_totalascores).round(3)
    df_subjects_wllevels_totalascores = pd.DataFrame(subjects_wllevels_totalascores).round(3)
    df_subjects_wldiffs = pd.DataFrame(subjects_wldiffs, index=[0]).T
    df_subjects_wldiffs.columns = ['Difference from Baseline']
    scores.to_csv(path_out_scores)
    df_overlaps.to_csv(path_out_subjects_overlaps, index=True)
    df_subjects_wldiffs.to_csv(path_out_subjects_wldiffs, index=True)
    df_subjects_levels_wldiffs.to_csv(path_out_subjects_levels_wldiffs, index=True)
    df_subjects_filenames_totalascores.to_csv(path_out_subjects_filenames_totalascores, index=True)
    df_subjects_wllevels_totalascores.to_csv(path_out_subjects_wllevels_totalascores, index=True)
    make_save_plots(dir_out=dir_out,
                    levels_colors=levels_colors,
                    subjects_wldiffs=subjects_wldiffs,
                    subjects_wllevelsascores=subjects_wllevels_ascores,
                    percent_change_from_baseline=percent_change_from_baseline)
    return percent_change_from_baseline


def make_save_plots(dir_out,
                    levels_colors,
                    subjects_wldiffs,
                    percent_change_from_baseline,
                    subjects_wllevelsascores):
    # subjects WLdiffs
    # plot_outputs_bars(mydict=subjects_wldiffs,
    #                   levels_colors=None,
    #                   title=f'Total WL Difference = {round(percent_change_from_baseline, 3)}',
    #                   xlabel='Subjects',
    #                   ylabel='WL Difference from Level 0 to 1-3',
    #                   path_out=os.path.join(dir_out, f'WL_Diffs.png'),
    #                   xtickrotation=90,
    #                   print_barheights=False)
    # WL across Task WL (agg. all subjects)
    fname = 'TaskWL_aScores'
    title = "Perceived WL vs Task WL"
    xlabel = 'Task WL'
    ylabel = 'Perceived WL'
    ## box
    wllevels_ascores = combine_dicts(dicts=list(subjects_wllevelsascores.values()))
    fig, ax = plt.subplots()
    ax.boxplot(wllevels_ascores.values())
    ax.set_xticklabels(wllevels_ascores.keys(), rotation=90)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True)
    out_path = os.path.join(dir_out, f'{fname}--box.png')
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    ## violin
    dfdict = {'Task WL': [], 'Anomaly Score': []}
    for wllevel, ascores in wllevels_ascores.items():
        dfdict['Task WL'] += [wllevel for _ in range(len(ascores))]
        dfdict['Anomaly Score'] += ascores
    df = pd.DataFrame(dfdict)
    vplot_anom = sns.violinplot(data=df,
                                x="Task WL",
                                y='Anomaly Score',
                                pallette=levels_colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=90)
    out_path = os.path.join(dir_out, f'{fname}--violin.png')
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def gridsearch_htm(config, dir_out, HZS, SPS, PERMDECS, PADDINGS):
    # Get scores
    modtypes_scores = {}
    for hz in HZS:
        print(f"\nHZ = {hz}")
        for sp in SPS:
            print(f"  SP = {sp}")
            for permdec in PERMDECS:
                print(f"    PERMDEC = {permdec}")
                for padding in PADDINGS:
                    print(f"      PADDING = {padding}")
                    config['hzs']['convertto'] = hz
                    config['htm_config_user']['models_state']['use_sp'] = sp
                    config['htm_config_model']['models_encoders']['p_padding'] = padding
                    config['htm_config_model']['models_params']['tm']['permanenceDec'] = permdec
                    # Make output dir
                    meta = f"HZ={hz}; SP={sp}; PERMDEC={permdec}, PADDING={padding}"
                    dir_out_meta = os.path.join(dir_out, meta)
                    os.makedirs(dir_out_meta, exist_ok=True)
                    # Run WL
                    score = run_wl(config=config, dir_in=config['dirs']['input'], dir_out=dir_out_meta)
                    modtypes_scores[meta] = score
    # Save scores
    modtypes_scores = dict(sorted(modtypes_scores.items(), key=lambda item: item[1]))
    for mt, score in modtypes_scores.items():
        print(f"{mt} = {round(score, 3)}")
    df_modtypes_scores = pd.DataFrame(modtypes_scores, index=['Score']).T
    df_modtypes_scores.sort_values(by='Score', ascending=True, inplace=True)
    path_out_csv = os.path.join(dir_out, "GRIDSEARCH.csv")
    path_out_png = os.path.join(dir_out, "GRIDSEARCH.png")
    df_modtypes_scores.to_csv(path_out_csv, index=True)
    plot_outputs_bars(mydict=modtypes_scores,
                      title='Gridsearch Performance',
                      xlabel='Model Setting',
                      ylabel='Performance',
                      xtickrotation=90,
                      path_out=path_out_png)
    feats_vals_scores = gather_scores(dir_out, config['htm_gridsearch'])
    plot_features_scores(feats_vals_scores, dir_out)
    return modtypes_scores


def gather_scores(dir_results, htm_gridsearch):
    dir_files = [f for f in os.listdir(dir_results) if f != '.DS_Store']
    grid_scores = {f: {} for f in htm_gridsearch}
    for feat, vals in htm_gridsearch.items():
        print(f"\n\nfeat = {feat}")
        for v in vals:
            print(f"\n  v = {v}")
            fv_files = [os.path.join(dir_results, f) for f in dir_files if f"{feat}{v}" in f]
            fv_scores = []
            for fd in fv_files:
                fd_files = [f for f in os.listdir(fd)]
                if not any("SUMMARY" in fd for fd in fd_files):
                    continue
                f_sum = [f for f in fd_files if 'SUMMARY' in f][0]
                score = float(f_sum.split("%change_from_baseline=")[1].split(';')[
                                  0])  # f_sum.split('SUMMARY (score=')[1].replace(')', '')
                print(f"    ** score = {score}")
                fv_scores.append(float(score))
            grid_scores[feat][v] = fv_scores
    return grid_scores


def plot_features_scores(features_vals_scores, dir_out):
    for feat, vals_scores in features_vals_scores.items():
        print(f"\n{feat}")
        path_out = os.path.join(dir_out, f'Score Summary - {feat}')
        fig, ax = plt.subplots()
        ax.boxplot(vals_scores.values())
        ax.set_xticklabels(vals_scores.keys(), rotation=90)
        ax.set_xlabel(feat)
        ax.set_ylabel('WL Scores')
        ax.yaxis.grid(True)
        plt.savefig(path_out, bbox_inches="tight")
        plt.close()


def is_wl_detected(aScores, change_thresh_percent, window_recent, window_previous):
    wl_detected = False
    # Get percent change (recent from previous)
    aScores_recent = aScores[-window_recent:]
    aScores_previous = aScores[-window_previous:-window_recent]
    recent_diff = np.mean(aScores_recent) - np.mean(aScores_previous)
    if np.mean(aScores_previous) == 0:
        percent_change = recent_diff * 100
    else:
        percent_change = (recent_diff / np.mean(aScores_previous)) * 100
    # Decide if WL change detected
    if percent_change >= change_thresh_percent:
        wl_detected = True
    return wl_detected


def make_training(paths_train, read_func, colnames):
    datas_train = list()
    for path_t in paths_train:
        data = read_func(path_t)
        data.columns = colnames
        datas_train.append(data)
    data_train = pd.concat(datas_train, axis=0)
    return data_train


def score_wl_detections(data_size, wl_changepoints, wl_changepoints_detected, change_detection_window):
    scores = {'true_pos': 0, 'false_pos': 0, 'true_neg': 0, 'false_neg': 0}
    # check if known changepoints detected
    wl_changepoints_windows = {}
    for cp in wl_changepoints:
        cp_detected = False
        cp_window = [cp + 1, min((cp + change_detection_window), data_size)]
        cp_detected_in_window = [v for v in wl_changepoints_detected if v in range(cp_window[0], cp_window[1])]
        wl_changepoints_windows[cp] = cp_window
        if len(cp_detected_in_window):
            cp_detected = True
        # if not cp_detected:
        #     scores['false_neg'] += 1
        if cp_detected:
            scores['true_pos'] += 1
        else:
            scores['false_neg'] += 1
    # check for false pos
    for cp in wl_changepoints_detected:
        cp_in_detection_window = False
        for cp_, window in wl_changepoints_windows.items():
            if cp in range(window[0], window[1]):
                cp_in_detection_window = True
        if not cp_in_detection_window:
            scores['false_pos'] += 1
        # else:  # cp does fall in a detection window
        #     scores['true_pos'] += 1
    # check for true neg
    rows_neg = [_ for _ in range(data_size)]
    for wl, cp_window in wl_changepoints_windows.items():
        for _ in range(cp_window[0], cp_window[1]):
            if _ in rows_neg:
                rows_neg.remove(_)
    true_neg = [v for v in rows_neg if v not in wl_changepoints_detected]
    scores['true_neg'] = len(true_neg)

    return scores, wl_changepoints_windows


def plot_wlchangepoints(feats_plot, file_type, aScores, testfile, data_test, dir_out_subj, wl_changepoints_windows,
                        wl_changepoints_detected):
    for feat in feats_plot:
        path_out = os.path.join(dir_out_subj,
                                f"{feat}--{testfile.replace(file_type, '')}--timeplot.png")

        behavior = data_test[feat]
        aScores_accum = get_accum(aScores)
        t = [_ for _ in range(len(behavior))]

        fig, ax1 = plt.subplots()

        color = 'black'
        ax1.set_xlabel('time')
        ax1.set_ylabel(feat, color=color)
        ax1.plot(t, behavior, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'blue'
        ax2.set_ylabel('Behavioral Entropy', color=color)  # we already handled the x-label with ax1
        ax2.plot(t, aScores_accum, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        # plot wl changes detected
        label_done = False
        for cp in wl_changepoints_detected:
            if not label_done:
                ax2.axvline(cp, color='green', lw=0.5, alpha=1, label='WL Change Detected')
                label_done = True
            else:
                ax2.axvline(cp, color='green', lw=0.5, alpha=1)

        # plot detection windows
        label_done = False
        for cp, window in wl_changepoints_windows.items():
            if not label_done:
                ax2.axvspan(window[0], window[1], alpha=0.5, color='red', label='WL Detection Window')
                label_done = True
            else:
                ax2.axvspan(window[0], window[1], alpha=0.5, color='red')
        plt.legend()
        plt.savefig(path_out)
        plt.close()


def get_subjects_data(cfg, filenames, subjects, subjects_spacesadd):
    print('Gathering subjects data...')
    subjects_filenames_data = dict()
    subjects_dfs_train = dict()
    for subj in sorted(subjects):
        dir_input = os.path.join(cfg['dirs']['input'], subj)
        # Load
        filenames_data = load_files(dir_input=dir_input, file_type=cfg['file_type'], read_func=cfg['read_func'],
                                    filenames=filenames)
        # Update columns
        filenames_data = update_colnames(filenames_data=filenames_data, colnames=cfg['colnames'])
        # Preprocess
        filenames_data, df_train = preprocess_data(subj=subj, cfg=cfg, filenames_data=filenames_data,
                                                   subjects_spacesadd=subjects_spacesadd)
        # Store
        subjects_dfs_train[subj] = df_train
        subjects_filenames_data[subj] = filenames_data

    return subjects_dfs_train, subjects_filenames_data


def get_subjects_models(config, dir_out, subjects_dfs_train):
    print('Training subjects models...')
    subjects_features_models = dict()
    subjects_features_alphas = dict()
    for subj, df_train in subjects_dfs_train.items():
        print(f"  --> {subj}")
        # Train model(s)
        if config['train_models']:
            config, features_models, features_alphas = train_save_models(df_train=df_train,
                                                                         alg=config['alg'],
                                                                         dir_output=dir_out,  # dir_out_subj_models,
                                                                         config=config,
                                                                         htm_config_user=config['htm_config_user'],
                                                                         htm_config_model=config['htm_config_model'])
        # Load model(s)
        else:
            if config['alg'] == 'HTM':
                features_models = load_models_htm(dir_out)  # dir_out_subj_models
            else:
                features_models = load_models_darts(dir_out, alg=config['alg'])  # dir_out_subj_models
        # Store models
        subjects_features_models[subj] = features_models
        subjects_features_alphas[subj] = features_alphas

    return config, subjects_features_models, subjects_features_alphas


def has_expected_files(dir_in, files_exp, any_all):
    files_found = [f for f in os.listdir(dir_in)]
    files_missing = [f for f in files_exp if f not in files_found]
    missing_max = 0 if any_all == 'all' else len(files_exp)-1
    if len(files_missing) <= missing_max:
        return True
    else:
        return False


def run_wl(config, subjects_filenames_data, subjects_dfs_train, subjects_features_models, subjects_features_alphas, dir_out):
    # Run
    if config['mode'] == 'post-hoc':
        print('\nMODE = post-hoc')
        score = run_posthoc(config, dir_out, subjects_filenames_data, subjects_dfs_train, subjects_features_models)
    else:
        print('\nMODE = real-time')
        score = np.mean(run_realtime(config, dir_out, subjects_features_models, subjects_filenames_data, subjects_features_alphas)['f1_score'])

    # return score


def get_subjects(dir_in, subjs_lim=100):
    # Collect subjects
    subjects_all = [f for f in os.listdir(dir_in) if os.path.isdir(os.path.join(dir_in, f)) if 'drop' not in f]
    subjects_all = subjects_all[:subjs_lim]  # HACK - limit number of subjects
    files_exp = list(config['wllevels_filenames'].values())
    files_exp = list(itertools.chain.from_iterable(files_exp))
    print(f"subjects_all = {subjects_all}")
    subjects = [s for s in subjects_all if has_expected_files(os.path.join(dir_in, s), files_exp, 'any')]
    print(f"subjects = {len(subjects)}")
    subjects_invalid = [s for s in subjects_all if s not in subjects]
    print(f"Subjects Found (valid) = {len(subjects)}")
    for s in sorted(subjects):
        print(f"  --> {s}")
    print(f"Subjects Found (invalid) = {len(subjects_invalid)}")
    for s in sorted(subjects_invalid):
        print(f"  --> {s}")
    subjects_spacesadd = {}
    subj_lens = [len(subj) for subj in subjects]
    print(f"subj_lens = {subj_lens}")
    subj_maxlen = max(subj_lens)
    for subj in subjects:
        diff_maxlen = subj_maxlen - len(subj)
        subjects_spacesadd[subj] = ' ' * diff_maxlen
    return subjects, subjects_spacesadd


def filter_config_group(group, config):
    filenames_group = []
    config_group = copy.deepcopy(config)
    files_eligible = config_group['wllevels_filenames'].pop('training') + config_group['wllevels_filenames'].pop(
        'baseline')
    files_training = [f for f in files_eligible if group not in f]
    files_baseline = [f for f in files_eligible if f not in files_training]
    config_group['wllevels_filenames']['training'] = files_training
    config_group['wllevels_filenames']['baseline'] = files_baseline
    wllevels_higher = [lev for lev in config_group['wllevels_filenames'] if lev not in ['training', 'baseline']]
    for lev in wllevels_higher:
        config_group['wllevels_filenames'][lev] = [f for f in config['wllevels_filenames'][lev] if group in f]
    for wllevel, filenames in config_group['wllevels_filenames'].items():
        filenames_group += filenames
        print(f"  {wllevel}\n  --> {filenames}")
    # reorder wllevels_filenames
    wllevels_order = ['training', 'baseline'] + wllevels_higher
    index_map = {v: i for i, v in enumerate(wllevels_order)}
    config_group['wllevels_filenames'] = dict(
        sorted(config_group['wllevels_filenames'].items(), key=lambda pair: index_map[pair[0]]))
    return config_group, filenames_group


def main(config):
    # set output dir
    dir_out = os.path.join(config['dirs']['output'], config['mode'], config['alg'],
                           f"hz={config['hzs']['convertto']}; features={'.'.join(config['columns_model'])}")
    os.makedirs(dir_out, exist_ok=True)

    # get subjects
    subjects, subjects_spacesadd = get_subjects(config['dirs']['input'], subjs_lim=100)  #100

    ## get inputs
    filenames = list(itertools.chain.from_iterable(config['wllevels_filenames'].values()))
    subj1 = list(config['subjects_testfiles_wltogglepoints'].keys())[0]
    filenames_realtime = list(config['subjects_testfiles_wltogglepoints'][subj1].keys())
    filenames += filenames_realtime
    subjects_dfs_train, subjects_filenames_data = get_subjects_data(config, filenames, subjects, subjects_spacesadd)

    ## train models
    config, subjects_features_models, subjects_features_alphas = get_subjects_models(config, dir_out, subjects_dfs_train)  #dir_out_total

    ## run
    run_wl(config=config,
           subjects_filenames_data=subjects_filenames_data,
           subjects_dfs_train=subjects_dfs_train,
           subjects_features_models=subjects_features_models,
           subjects_features_alphas=subjects_features_alphas,
           dir_out=dir_out)


if __name__ == '__main__':
    # load config
    args_add = [{'name_abbrev': '-cp', 'name': '--config_path', 'required': True, 'help': 'path to config'}]
    config = load_config(get_args(args_add).config_path)
    config['read_func'] = FILETYPES_READFUNCS[config['file_type']]

    # run main
    main(config)
