import itertools
import os
import sys

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
from source.preprocess.preprocess import update_colnames, get_wllevelsdf, preprocess_data, get_wllevels_alldata
from source.analyze.tlx import make_boxplots, get_tlx_overlaps
from source.analyze.plot import plot_data, plot_boxes, plot_lines, plot_bars
from source.analyze.anomaly import get_wllevels_diffs, get_ascores_entropy, get_ascores_naive, \
    get_ascores_pyod, get_ascore_pyod, get_subjects_wldiffs, get_f1score, get_ascore_entropy, get_entropy_ts
from ts_source.utils.utils import add_timecol, load_config, load_models as load_models_darts

from ts_source.model.model import get_model_lag, LAG_MIN, get_modname, get_preds_rolling
from htm_source.pipeline.htm_batch_runner import run_batch
from htm_source.utils.fs import load_models as load_models_htm

FILETYPES_READFUNCS = {
    'xls': pd.read_excel,
    'csv': pd.read_csv
}


def get_wllevels_outputs(filenames_ascores, filenames_predcounts, wllevels_filenames,
                         levels_order=['baseline', 'distraction', 'rain', 'fog'], ):
    wllevels_ascores = {wllevel: [] for wllevel in wllevels_filenames if wllevel != 'training'}
    wllevels_pcounts = {wllevel: [] for wllevel in wllevels_filenames if wllevel != 'training'}
    wllevels_totalascores = {wllevel: [] for wllevel in wllevels_filenames if wllevel != 'training'}
    filenames_wllevels = {}

    for level, fns in wllevels_filenames.items():
        if level == 'training':
            continue
        for fn in fns:
            filenames_wllevels[fn] = level

    for fn, ascores in filenames_ascores.items():
        wllevel = filenames_wllevels[fn]
        wllevels_ascores[wllevel] += ascores
        wllevels_totalascores[wllevel].append(np.sum(ascores))
        if fn in filenames_predcounts:
            wllevels_pcounts[wllevel] += filenames_predcounts[fn]

    # reorder dict
    wllevels_totalascores2 = {}
    for k in levels_order:
        wllevels_totalascores2[k] = wllevels_totalascores[k]

    return wllevels_ascores, wllevels_pcounts, wllevels_totalascores2


def get_filenames_outputs(cfg,
                          filenames_data,
                          features_models):
    filenames_ascores = {fn: [] for fn in filenames_data if 'static' in fn}
    filenames_pcounts = {fn: [] for fn in filenames_data if 'static' in fn}

    for fn in filenames_ascores:
        data = add_timecol(filenames_data[fn], config['time_col'])
        if cfg['alg'] == 'HTM':
            feats_models, features_outputs = run_batch(cfg_user=cfg['htm_config_user'],
                                                       cfg_model=cfg['htm_config_model'],
                                                       config_path_user=None,
                                                       config_path_model=None,
                                                       learn=cfg['learn_in_testing'],
                                                       data=data,
                                                       iter_print=1000,
                                                       features_models=features_models)
            ascores = features_outputs[f"megamodel_features={len(cfg['columns_model'])}"]['anomaly_score']
            pcounts = features_outputs[f"megamodel_features={len(cfg['columns_model'])}"]['pred_count']
            filenames_pcounts[fn] += pcounts

        elif config['alg'] == 'SteeringEntropy':
            ascores = get_ascores_entropy(data[cfg['columns_model'][0]].values)  # data['steering angle'].values

        elif config['alg'] == 'Naive':
            ascores = get_ascores_naive(data[cfg['columns_model'][0]].values)  # data['steering angle'].values

        elif config['alg'] in ['IForest', 'OCSVM', 'KNN', 'LOF', 'AE', 'VAE', 'KDE']:
            ascores = get_ascores_pyod(data[cfg['columns_model']], features_models[
                cfg['columns_model'][0]])  # data[['steering angle']], features_models['steering angle']

        else:  # ts_source alg
            for feat, model in features_models.items():  # Assumes single model
                break
            mod_name = get_modname(model)
            features = model.training_series.components
            preds = get_preds_rolling(model=model,
                                      df=data,
                                      features=features,
                                      LAG=max(LAG_MIN, get_model_lag(mod_name, model)),
                                      time_col=cfg['time_col'],
                                      forecast_horizon=cfg['forecast_horizon'])
            preds_df = pd.DataFrame(preds, columns=list(features))
            data_ = data.tail(preds_df.shape[0])
            ascores = list(abs(data_[cfg['columns_model'][0]].values - preds_df[cfg['columns_model'][
                0]].values))  # list(abs(data_['steering angle'].values - preds_df['steering angle'].values))

        filenames_ascores[fn] = ascores
        data.drop(columns=[config['time_col']], inplace=True)

    return filenames_ascores, filenames_pcounts


def run_subject(cfg, df_train, dir_output, filenames_data, features_models, save_results=True):
    # get data & outputs by wllevel
    filenames_ascores, filenames_predcounts = get_filenames_outputs(config, filenames_data, features_models)
    wllevels_anomscores, wllevels_predcounts, wllevels_totalascores = get_wllevels_outputs(
        filenames_ascores,
        filenames_predcounts,
        config['wllevels_filenames'])
    wllevels_alldata = get_wllevels_alldata(wllevels_anomscores=wllevels_anomscores,
                                            wllevels_filenames=cfg['wllevels_filenames'],
                                            filenames_data=filenames_data,
                                            columns_model=cfg['columns_model'],
                                            dir_output=dir_output,
                                            save_results=save_results)

    # Get WL diffs btwn wllevels
    wllevels_diffs = get_wllevels_diffs(wllevels_anomscores=wllevels_anomscores)
    if save_results:
        # Plot data
        plot_data(filenames_data=filenames_data, file_type=cfg['file_type'], dir_output=dir_output)
        # Write outputs
        print(f"  Writing outputs to --> {dir_output}")
        print("    Boxplots...")
        make_boxplots(wllevels_totalascores,
                      ylabel=f"{cfg['alg']} WL",
                      title=f"{cfg['alg']} WL Scores by Run Mode",
                      suptitle=None,
                      path_out=os.path.join(dir_output, 'anomaly', "levels--aScoreTotals--box.png"),
                      ylim=None)
        wllevels_totalascores_sum = {wllevel: np.sum(ascores) for wllevel, ascores in wllevels_totalascores.items()}
        plot_bars(mydict=wllevels_totalascores_sum,
                  title=f"{cfg['alg']} WL Scores by Run Mode",
                  xlabel='WL Levels',
                  ylabel=f"{cfg['alg']} WL",
                  path_out=os.path.join(dir_output, 'anomaly', "levels--aScoreTotals--bar.png"),
                  xtickrotation=0,
                  colors=['grey', 'orange', 'blue', 'green'])
        plot_boxes(data_plot1=wllevels_anomscores,
                   data_plot2=wllevels_predcounts,
                   title_1=f"Anomaly Scores by WL Level\nhz={cfg['hzs']['convertto']},; features={cfg['columns_model']}",
                   title_2=f"Prediction Counts by WL Level\nhz={cfg['hzs']['convertto']},; features={cfg['columns_model']}",
                   out_dir=os.path.join(dir_output, 'anomaly'),
                   xlabel='MWL Levels',
                   ylabel='HTM Metric')
        print("    Lineplots...")
        plot_lines(wllevels_anomscores_=wllevels_anomscores,
                   wllevels_predcounts_=wllevels_predcounts,
                   wllevels_alldata_=wllevels_alldata,
                   get_pcounts=True if cfg['alg'] == 'HTM' else False,
                   df_train=df_train,
                   columns_model=cfg['columns_model'],
                   out_dir=os.path.join(dir_output, 'anomaly'))
    return wllevels_anomscores, wllevels_diffs


def get_scores(subjects_wldiffs, subjects_wllevelsascores):
    # percent_change_from_baseline
    percent_change_from_baseline = round(sum(subjects_wldiffs.values()))
    # subjects_wldiffs_positive
    subjects_wldiffs_positive = {subj: diff for subj, diff in subjects_wldiffs.items() if diff > 0}
    # subjs_baseline_lowest
    subjs_baseline_lowest = []
    for subj, wllevels_ascores in subjects_wllevelsascores.items():
        ascorestotal_baseline = np.sum(wllevels_ascores['baseline'])
        baseline_lowest = True
        for wllevel, ascores in wllevels_ascores.items():
            if wllevel == 'baseline':
                continue
            if np.sum(ascores) < ascorestotal_baseline:
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
    # Gather WL results
    dfs_wllevelsdiffs = []
    subjects_wllevelsdiffs = {}
    subjects_wllevelsascores = {}
    for subj, filenames_data in subjects_filenames_data.items():
        dir_output_subj = os.path.join(dir_out, subj)
        wllevels_ascores, wllevels_diffs = run_subject(cfg=cfg,
                                                       df_train=subjects_dfs_train[subj],
                                                       dir_output=dir_output_subj,
                                                       filenames_data=filenames_data,
                                                       features_models=subjects_features_models[subj],
                                                       save_results=cfg['make_plots'])
        dfs_wllevelsdiffs.append(get_wllevelsdf(subj, wllevels_diffs))
        subjects_wllevelsdiffs[subj] = wllevels_diffs
        subjects_wllevelsascores[subj] = wllevels_ascores

    # Get Scores
    subjects_wldiffs, subjects_levels_wldiffs = get_subjects_wldiffs(subjects_wllevelsascores)
    subjects_wldiffs_capped = {k: min(v, 1000) for k, v in subjects_wldiffs.items()}
    percent_change_from_baseline, subjects_increased_from_baseline, subjects_baseline_lowest = get_scores(
        subjects_wldiffs, subjects_wllevelsascores)
    percent_subjects_baseline_lowest = round(100 * len(subjects_baseline_lowest) / len(subjects_wllevelsascores))
    percent_subjects_increased_from_baseline = round(
        100 * len(subjects_increased_from_baseline) / len(subjects_wldiffs))

    # get overlap w/TLX
    mean_tlx_overlap, df_overlaps = get_tlx_overlaps(subjects_wllevelsascores)

    # rename dirs based on scores
    rename_dirs_by_scores(subjects_wldiffs, subjects_increased_from_baseline, subjects_baseline_lowest, dir_out)

    # Save Results
    path_out_scores = os.path.join(dir_out, 'scores.csv')
    path_out_subjects_wldiffs = os.path.join(dir_out, 'subjects_wldiffs.csv')
    path_out_subjects_levels_wldiffs = os.path.join(dir_out, 'subjects_levels_wldiffs.csv')
    path_out_subjects_overlaps = os.path.join(dir_out, 'subjects_overlaps.csv')
    scores = pd.DataFrame({'percent_change_from_baseline': percent_change_from_baseline,
                           'percent_subjects_increased_from_baseline': percent_subjects_increased_from_baseline,
                           'percent_subjects_baseline_lowest': percent_subjects_baseline_lowest,
                           'mean_tlx_overlap': mean_tlx_overlap}, index=[0])
    df_subjects_levels_wldiffs = pd.DataFrame(subjects_levels_wldiffs)
    df_subjects_wldiffs = pd.DataFrame(subjects_wldiffs, index=[0]).T
    df_subjects_wldiffs.columns = ['%Change from Baseline']
    scores.to_csv(path_out_scores)
    df_overlaps.to_csv(path_out_subjects_overlaps, index=True)
    df_subjects_wldiffs.to_csv(path_out_subjects_wldiffs, index=True)
    df_subjects_levels_wldiffs.to_csv(path_out_subjects_levels_wldiffs, index=True)
    make_save_plots(dir_out=dir_out,
                    dfs_wllevelsdiffs=dfs_wllevelsdiffs,
                    subjects_wldiffs=subjects_wldiffs_capped,
                    percent_change_from_baseline=percent_change_from_baseline,
                    subjects_wllevelsascores=subjects_wllevelsascores)
    return percent_change_from_baseline


def run_realtime(config, dir_out, subjects_features_models):
    rows = list()
    for subj, features_models in subjects_features_models.items():
        print(f"  subj = {subj}")
        dir_out_subj = os.path.join(dir_out, subj)
        dir_in_subj = os.path.join(config['dirs']['input'], subj)
        print("    testing...")
        for feat, model in features_models.items():
            print(f"      feat = {feat}")
            for testfile, times in config['subjects_testfiles_wltogglepoints'][subj].items():
                print(f"        testfile = {testfile}")
                aScores, wl_changepoints_detected = list(), list()
                path_test = os.path.join(dir_in_subj, testfile)
                data_test = config['read_func'](path_test)
                data_test.columns = config['colnames']
                data_test.drop(columns=[config['time_col']], inplace=True)
                data_test = add_timecol(data_test, config['time_col'])

                # get wl_changepoints
                wl_changepoints = [int(t / times['time_total'] * data_test.shape[0]) for t in
                                   times['times_wltoggle']]
                print(f"        wl_changepoints = {wl_changepoints}")

                # run data thru model
                pred_prev = None
                for _, row in data_test[config['columns_model']].iterrows():
                    if config['alg'] == 'HTM':
                        aScore, aLikl, pCount, sPreds = model.run(features_data=dict(row), timestep=_ + 1,
                                                                  learn=config['learn_in_testing'])
                    elif config['alg'] == 'SteeringEntropy':
                        aScore, pred_prev = get_ascore_entropy(_, row, feat, model, data_test, pred_prev)
                    elif config['alg'] in ['IForest', 'OCSVM', 'KNN', 'LOF', 'AE', 'VAE', 'KDE']:
                        aScore = get_ascore_pyod(_, data_test[config['columns_model']], model)
                    else:
                        aScore, pred_prev = get_entropy_ts(_, model, row, data_test, config, pred_prev, LAG_MIN)
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
                scores, wl_changepoints_windows = score_wl_detections(data_test.shape[0],
                                                                      wl_changepoints,
                                                                      wl_changepoints_detected,
                                                                      config['windows_ascore']['change_detection'])
                f1 = get_f1score(scores['true_pos'], scores['false_pos'], scores['false_neg'])
                path_out = os.path.join(dir_out_subj,
                                        f"{feat}--{testfile.replace(config['file_type'], '')}--confusion_matrix.csv")
                pd.DataFrame(scores, index=[0]).to_csv(path_out, index=False)

                # plot detected wl-changepoints over changepoint windows
                plot_wlchangepoints(config['columns_model'],
                                    config['file_type'],
                                    testfile,
                                    data_test,
                                    dir_out_subj,
                                    wl_changepoints_windows,
                                    wl_changepoints_detected)
                rows.append({'subj': subj, 'feat': feat, 'testfile': testfile, 'f1': f1})
    df_f1 = pd.DataFrame(rows)
    path_out = os.path.join(dir_out, "f1_scores.csv")
    df_f1.to_csv(path_out, index=False)
    return df_f1


def make_save_plots(dir_out,
                    dfs_wllevelsdiffs,
                    subjects_wldiffs,
                    percent_change_from_baseline,
                    subjects_wllevelsascores):
    # df_wllevelsdiffs
    # df_wllevelsdiffs = pd.concat(dfs_wllevelsdiffs, axis=0)
    # df_wllevelsdiffs.to_csv(os.path.join(dir_out, f"WL_Diffs.csv"))
    # cols_plot = [c for c in df_wllevelsdiffs if c != 'subject']
    # img = sns.heatmap(df_wllevelsdiffs[cols_plot], annot=True).get_figure()
    # img.savefig(os.path.join(dir_out, f"WL_Diffs-heatmap.png"))
    # plt.close()

    # subjects WLdiffs
    plot_bars(mydict=subjects_wldiffs,
              title=f'WL Change from WL Levels 0 to 1-3\n  Total % Change={round(percent_change_from_baseline, 3)}',
              xlabel='Subjects',
              ylabel='WL % Change from Level 0 to 1-3',
              path_out=os.path.join(dir_out, f'WL_Diffs.png'),
              xtickrotation=90,
              colors=['grey', 'orange', 'blue', 'green'])
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
                                y='Anomaly Score')
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
    plot_bars(mydict=modtypes_scores,
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
        cp_window = [cp + 1, cp + change_detection_window]
        cp_detected_in_window = [v for v in wl_changepoints_detected if v in range(cp_window[0], cp_window[1])]
        wl_changepoints_windows[cp] = cp_window
        if len(cp_detected_in_window):
            cp_detected = True
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
    # check for true neg
    rows_neg = [_ for _ in range(data_size)]
    for wl, cp_window in wl_changepoints_windows.items():
        for _ in range(cp_window[0], cp_window[1]):
            rows_neg.remove(_)
    true_neg = [v for v in rows_neg if v not in wl_changepoints_detected]
    scores['true_neg'] = len(true_neg)

    return scores, wl_changepoints_windows


def plot_wlchangepoints(columns_model, file_type, testfile, data_test, dir_out_subj, wl_changepoints_windows,
                        wl_changepoints_detected):
    for feat in columns_model:
        path_out = os.path.join(dir_out_subj,
                                f"{feat}--{testfile.replace(file_type, '')}--timeplot.png")
        plt.plot(data_test[feat])
        plt.xlabel('time')
        plt.ylabel(feat)
        for cp in wl_changepoints_detected:
            plt.axvline(cp, color='green', lw=0.5, alpha=0.3)
        for cp, window in wl_changepoints_windows.items():
            plt.axvspan(window[0], window[1], alpha=0.5, color='red')
        plt.savefig(path_out)
        plt.close()


def get_subjects_data(cfg, subjects, subjects_spacesadd, dir_out):
    print('Gathering subjects data...')
    subjects_filenames_data = dict()
    subjects_dfs_train = dict()
    for subj in sorted(subjects):

        # if subj not in ['aranoff', 'balaji', 'charles']:
        #     continue

        dir_input = os.path.join(cfg['dirs']['input'], subj)
        dir_output = os.path.join(dir_out, subj)
        folders = ['anomaly', 'models', 'scalers', 'data']
        dirs_out = [os.path.join(dir_output, f) for f in folders]
        for d in dirs_out:
            os.makedirs(d, exist_ok=True)
        # Load
        filenames_data = load_files(dir_input=dir_input, file_type=cfg['file_type'], read_func=cfg['read_func'])
        # Update columns
        filenames_data = update_colnames(filenames_data=filenames_data, colnames=cfg['colnames'])
        # Preprocess
        filenames_data, df_train = preprocess_data(subj, cfg, dir_output, filenames_data, subjects_spacesadd)
        # Store
        subjects_dfs_train[subj] = df_train
        subjects_filenames_data[subj] = filenames_data

    return subjects_dfs_train, subjects_filenames_data


def get_subjects_models(config, dir_out, subjects_dfs_train):
    print('Training subjects models...')
    subjects_features_models = dict()
    for subj, df_train in subjects_dfs_train.items():
        print(f"  --> {subj}")
        dir_output = os.path.join(dir_out, subj)
        # Train model(s)
        if config['train_models']:
            features_models = train_save_models(df_train=df_train,
                                                alg=config['alg'],
                                                dir_output=dir_output,
                                                config=config,
                                                htm_config_user=config['htm_config_user'],
                                                htm_config_model=config['htm_config_model'])
        # Load model(s)
        else:
            dir_models = os.path.join(dir_output, 'models')
            if config['alg'] == 'HTM':
                features_models = load_models_htm(dir_models)
            else:
                features_models = load_models_darts(dir_models, alg=config['alg'])
        # Store data & models
        subjects_features_models[subj] = features_models
    return subjects_features_models


def has_expected_files(dir_in, files_exp):
    files_found = [f for f in os.listdir(dir_in)]
    files_missing = [f for f in files_exp if f not in files_found]
    if len(files_missing) == 0:
        return True
    else:
        return False


def run_wl(cfg, dir_in, dir_out, make_dir_alg=True, make_dir_metadata=True):
    # Collect subjects
    subjects_all = [f for f in os.listdir(dir_in) if os.path.isdir(os.path.join(dir_in, f)) if 'drop' not in f]
    files_exp = list(cfg['wllevels_filenames'].values())
    files_exp = list(itertools.chain.from_iterable(files_exp))
    subjects = [s for s in subjects_all if has_expected_files(os.path.join(dir_in, s), files_exp)]
    subjects_invalid = [s for s in subjects_all if s not in subjects]
    print(f"Subjects Found (valid) = {len(subjects)}")
    for s in sorted(subjects):
        print(f"  --> {s}")
    print(f"Subjects Found (invalid) = {len(subjects_invalid)}")
    for s in sorted(subjects_invalid):
        print(f"  --> {s}")
    subjects_spacesadd = {}
    subj_maxlen = max([len(subj) for subj in subjects])
    for subj in subjects:
        diff_maxlen = subj_maxlen - len(subj)
        subjects_spacesadd[subj] = ' ' * diff_maxlen

    # make alg dir & reset dir_out
    if make_dir_alg:
        dir_out_alg = os.path.join(dir_out, cfg['alg'])
        os.makedirs(dir_out_alg, exist_ok=True)
        dir_out = dir_out_alg

    # make metadata dir
    if make_dir_metadata:
        preproc = '-'.join([f"{k}={v}" for k, v in cfg['preprocess'].items() if v])
        metadata_dir = os.path.join(dir_out, f"preproc--{preproc}; hz={cfg['hzs']['convertto']}")
        os.makedirs(metadata_dir, exist_ok=True)
        dir_out = metadata_dir

    # Make subjects' output dirs
    for subj in subjects:
        make_dirs_subj(os.path.join(dir_out, subj))

    # Get subjects data
    subjects_dfs_train, subjects_filenames_data = get_subjects_data(config, subjects, subjects_spacesadd, dir_out)

    # Train subjects models
    subjects_features_models = get_subjects_models(config, dir_out, subjects_dfs_train)

    """
    UPDATE
    # Plot EDA
    plot_subjects_timeserieses(dir_data="/Users/samheiserman/Desktop/repos/workload_assessor/data",
                               dir_out="/Users/samheiserman/Desktop/repos/workload_assessor/eda",
                               path_cfg="/Users/samheiserman/Desktop/repos/workload_assessor/configs/run_pipeline.yaml")
    """

    if config['mode'] == 'post-hoc':
        print('\nMODE = post-hoc')
        score = run_posthoc(config, dir_out, subjects_filenames_data, subjects_dfs_train, subjects_features_models)
    else:
        print('\nMODE = real-time')
        score = np.mean(run_realtime(config, dir_out, subjects_features_models)['f1'])

    return score


if __name__ == '__main__':

    # Load config
    args_add = [{'name_abbrev': '-cp', 'name': '--config_path', 'required': True, 'help': 'path to config'}]
    config = load_config(get_args(args_add).config_path)
    config['read_func'] = FILETYPES_READFUNCS[config['file_type']]

    # Set output dir
    dir_out = os.path.join(config['dirs']['output'], config['mode'])  # ,config['alg']
    os.makedirs(dir_out, exist_ok=True)

    if config['alg'] == 'HTM' and config['do_gridsearch']:
        modtypes_scores = gridsearch_htm(config=config,
                                         dir_out=dir_out,
                                         SPS=config['htm_gridsearch']['SP='],
                                         HZS=config['htm_gridsearch']['HZ='],
                                         PERMDECS=config['htm_gridsearch']['PERMDEC='],
                                         PADDINGS=config['htm_gridsearch']['PADDING%='])
    else:
        run_wl(cfg=config, dir_in=config['dirs']['input'], dir_out=dir_out)
