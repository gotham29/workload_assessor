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

from darts import TimeSeries
from source.model.model import train_save_models
from source.utils.utils import get_args, load_config, load_files, make_dirs_subj
from source.preprocess.preprocess import update_colnames, agg_data, clip_data, get_ttypesdf, get_dftrain
from source.analyze.plot import plot_data, plot_boxes, plot_lines, plot_bars
from source.analyze.anomaly import get_testtypes_outputs, get_testtypes_diffs
from ts_source.utils.utils import add_timecol, load_models as load_models_darts
from ts_source.preprocess.preprocess import reshape_datats
from ts_source.model.model import get_model_lag, LAG_MIN


FILETYPES_READFUNCS = {
    'xls': pd.read_excel,
    'csv': pd.read_csv
}


def combine_dicts(dicts):
    d1 = dicts[0]
    d_comb = {k: [] for k in d1}
    for d_ in dicts:
        for k, v in d_.items():
            d_comb[k] += v
    return d_comb


def subtract_mean(filenames_data, wllevels_filenames, time_col='timestamp'):
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
            data[feat] = data[feat] - mean
    return filenames_data


def run_subject(config, df_train, dir_output, wllevels_tlx, filenames_data, features_models, save_results=True):
    wllevels_anomscores, wllevels_predcounts, wllevels_alldata = get_testtypes_outputs(alg=config['alg'],
                                                                                       htm_config_user=config[
                                                                                           'htm_config_user'],
                                                                                       htm_config_model=config[
                                                                                           'htm_config_model'],
                                                                                       dir_output=dir_output,
                                                                                       learn_in_testing=config[
                                                                                           'learn_in_testing'],
                                                                                       columns_model=config[
                                                                                           'columns_model'],
                                                                                       filenames_data=filenames_data,
                                                                                       features_models=features_models,
                                                                                       testtypes_filenames=config[
                                                                                           'testtypes_filenames'],
                                                                                       save_results=save_results)
    # Get WL diffs btwn testtypes
    wllevels_diffs = get_testtypes_diffs(testtypes_anomscores=wllevels_anomscores)
    if save_results:
        # Plot data
        plot_data(filenames_data=filenames_data, file_type=config['file_type'], dir_output=dir_output)
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
                   get_pcounts=True if config['alg'] == 'HTM' else False,
                   df_train=df_train,
                   columns_model=config['columns_model'],
                   out_dir=os.path.join(dir_output, 'anomaly'))
        print("    Barplots...")
        plot_bars(mydict=wllevels_tlx,
                  title='NASA TLX vs Task WL',
                  xlabel='Task WL',
                  ylabel='NASA TLX',
                  path_out=os.path.join(dir_output, 'bars--TLXs.png'))
    return wllevels_anomscores, wllevels_diffs


def get_subjects_wldiffs(subjects_ttypesascores):
    subjects_wldiffs = {}
    for subj, wllevels_anomscores in subjects_ttypesascores.items():
        wlmean_l0 = np.mean(wllevels_anomscores['Level 0'])
        diff_pct_total = 0
        for wllevel, ascores in wllevels_anomscores.items():
            if wllevel == 'Level 0':
                continue
            diff_l0 = (np.mean(ascores) - wlmean_l0)
            diff_pct = (diff_l0 / wlmean_l0) * 100
            diff_pct_total += diff_pct
        subjects_wldiffs[subj] = round(diff_pct_total, 1)
    return subjects_wldiffs


def get_f1score(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    denom = (precision + recall)
    if denom == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / denom
    return round(f1, 3)


def run_posthoc(config, dir_out, subjects_filenames_data, subjects_dfs_train, subjects_features_models):
    dfs_ttypesdiffs = []
    subjects_ttypesdiffs = {}
    subjects_ttypesascores = {}
    # Gather WL results
    for subj, filenames_data in subjects_filenames_data.items():
        dir_output_subj = os.path.join(dir_out, subj)
        ttypes_ascores, ttypes_diffs = run_subject(config=config,
                                                   df_train=subjects_dfs_train[subj],
                                                   dir_output=dir_output_subj,
                                                   wllevels_tlx=config['subjects_wllevels_tlx'][subj],
                                                   filenames_data=filenames_data,
                                                   features_models=subjects_features_models[subj],
                                                   save_results=True)
        dfs_ttypesdiffs.append(get_ttypesdf(subj, ttypes_diffs))
        subjects_ttypesdiffs[subj] = ttypes_diffs
        subjects_ttypesascores[subj] = ttypes_ascores
    # Get Score
    subjects_wldiffs = get_subjects_wldiffs(subjects_ttypesascores)
    print(f"\nsubjects_wldiffs...")
    for subj, wld in subjects_wldiffs.items():
        print(f"  {subj} --> {wld}")
    diff_from_WL0 = sum(subjects_wldiffs.values())
    # Save Results
    dir_out_summary = os.path.join(dir_out, f'SUMMARY (score={round(diff_from_WL0, 3)})')
    os.makedirs(dir_out_summary, exist_ok=True)
    make_save_plots(dir_out=dir_out_summary,
                    dfs_ttypesdiffs=dfs_ttypesdiffs,
                    subjects_wldiffs=subjects_wldiffs,
                    diff_from_WL0=diff_from_WL0,
                    subjects_ttypesascores=subjects_ttypesascores)
    return diff_from_WL0


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
                    dfs_ttypesdiffs,
                    subjects_wldiffs,
                    diff_from_WL0,
                    subjects_ttypesascores):
    # df_ttypesdiffs
    df_ttypesdiffs = pd.concat(dfs_ttypesdiffs, axis=0)
    path_out = os.path.join(dir_out, f"WL_Diffs.csv")
    df_ttypesdiffs.to_csv(path_out)
    # subjects WLdiffs
    plot_bars(mydict=subjects_wldiffs,
              title=f'WL Change from WL Levels 0 to 1-3\n  Total % Change={round(diff_from_WL0, 3)}',
              xlabel='Subjects',
              ylabel='WL % Change from Level 0 to 1-3',
              path_out=os.path.join(dir_out, f'WL_Diffs.png'))
    # WL across Task WL (agg. all subjects)
    fname = 'TaskWL_aScores'
    title = "Perceived WL vs Task WL"
    xlabel = 'Task WL'
    ylabel = 'Perceived WL'
    ## box
    ttypes_ascores = combine_dicts(dicts=list(subjects_ttypesascores.values()))
    plt.cla()
    fig, ax = plt.subplots()
    ax.boxplot(ttypes_ascores.values())
    ax.set_xticklabels(ttypes_ascores.keys(), rotation=90)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True)
    out_path = os.path.join(dir_out, f'{fname}--box.png')
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight")
    ## violin
    dfdict = {'Task WL': [], 'Anomaly Score': []}
    for ttype, ascores in ttypes_ascores.items():
        dfdict['Task WL'] += [ttype for _ in range(len(ascores))]
        dfdict['Anomaly Score'] += ascores
    df = pd.DataFrame(dfdict)
    plt.cla()
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
                score = f_sum.split('SUMMARY (score=')[1].replace(')', '')
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
        plt.cla()
        plt.plot(data_test[feat])
        plt.xlabel('time')
        plt.ylabel(feat)
        for cp in wl_changepoints_detected:
            plt.axvline(cp, color='green', lw=0.5, alpha=0.3)
        for cp, window in wl_changepoints_windows.items():
            plt.axvspan(window[0], window[1], alpha=0.5, color='red')
        plt.savefig(path_out)
        plt.cla()


def get_ascore_entropy(_, row, feat, model, data_test, pred_prev, LAG=3):
    aScore, do_pred = 0, True
    if _ < LAG:
        pred_prev, do_pred = None, False
    if pred_prev:
        aScore = abs(pred_prev - row[feat])
    if do_pred:
        lag1, lag2, lag3 = data_test[feat][_ - 3], data_test[feat][_ - 2], data_test[feat][_ - 1]
        pred_prev = model.predict(lag1, lag2, lag3)
    return aScore, pred_prev


def get_ascore_pyod(_, data, model):
    aScore = [0]
    if _ > 0:
        aScore = model.decision_function(data[(_-1):_])  # outlier scores
    return abs(aScore[0])


def get_entropy_ts(_, model, row, data_test, config, pred_prev, LAG_MIN):
    aScore, do_pred = 0, True
    features_model = list(model.training_series.components)
    features = features_model + [config['time_col']]
    LAG = max(LAG_MIN, get_model_lag(config['alg'], model))
    if _ < LAG:
        pred_prev, do_pred = None, False
    if pred_prev:
        aScore = abs(pred_prev - row[features_model])
    if do_pred:
        df_lag = data_test[features][_ - LAG:_]
        ts = TimeSeries.from_dataframe(df_lag, time_col=config['time_col'])
        pred = model.predict(n=config['forecast_horizon'], series=ts)
        pred_prev = reshape_datats(ts=pred, shape=(len(features_model)))
    return aScore, pred_prev


def get_subjects_data(config, subjects, dir_out):
    print('Gathering subjects data...')
    subjects_filenames_data = dict()
    subjects_dfs_train = dict()
    for subj in subjects:
        print(f"  --> {subj}")
        dir_input = os.path.join(config['dirs']['input'], subj)
        dir_output = os.path.join(dir_out, subj)
        folders = ['anomaly', 'models', 'scalers', 'data']
        dirs_out = [os.path.join(dir_output, f) for f in folders]
        for d in dirs_out:
            os.makedirs(d, exist_ok=True)
        # Load data
        filenames_data = load_files(dir_input=dir_input, file_type=config['file_type'], read_func=config['read_func'])
        # Update columns
        filenames_data = update_colnames(filenames_data=filenames_data, colnames=config['colnames'])
        # Agg data
        filenames_data = agg_data(filenames_data=filenames_data, hz_baseline=config['hzs']['baseline'],
                                  hz_convertto=config['hzs']['convertto'])

        """ preprocess - (detrending/differencing/seasonal differencing) """

        # Clip data
        filenames_data = clip_data(filenames_data=filenames_data, clip_percents=config['clip_percents'])
        # Subtract mean
        filenames_data = subtract_mean(filenames_data=filenames_data, wllevels_filenames=config['testtypes_filenames'],
                                   time_col=config['time_col'])
        # Train models
        df_train = get_dftrain(wllevels_filenames=config['testtypes_filenames'], filenames_data=filenames_data,
                               columns_model=config['columns_model'], dir_data=os.path.join(dir_output, 'data'))
        # Add timecol
        df_train = add_timecol(df_train, config['time_col'])
        # Store data
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


def run_wl(config, dir_in, dir_out):

    # Collect subjects
    subjects = [f for f in os.listdir(dir_in) if os.path.isdir(os.path.join(dir_in, f))]
    print(f"  Subjects Found = {len(subjects)}")

    # Make subjects' output dirs
    for subj in subjects:
        make_dirs_subj(os.path.join(dir_out, subj))

    # Get subjects data
    subjects_dfs_train, subjects_filenames_data = get_subjects_data(config, subjects, dir_out)

    # Train subjects models
    subjects_features_models = get_subjects_models(config, dir_out, subjects_dfs_train)

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
    dir_out = os.path.join(config['dirs']['output'], config['mode'], config['alg'])
    os.makedirs(dir_out, exist_ok=True)

    if config['alg'] == 'HTM' and config['do_gridsearch']:
        modtypes_scores = gridsearch_htm(config=config,
                                         dir_out=dir_out,
                                         SPS=config['htm_gridsearch']['SP='],
                                         HZS=config['htm_gridsearch']['HZ='],
                                         PERMDECS=config['htm_gridsearch']['PERMDEC='],
                                         PADDINGS=config['htm_gridsearch']['PADDING%='])
    else:
        run_wl(config=config, dir_in=config['dirs']['input'], dir_out=dir_out)
