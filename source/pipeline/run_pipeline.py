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
from source.preprocess.preprocess import update_colnames, preprocess_data, get_wllevels_totaldfs, prep_data, \
    get_wllevels_indsend
from source.analyze.tlx import make_boxplots, get_tlx_overlaps
from source.analyze.plot import make_data_plots, plot_outputs_boxes, plot_outputs_lines, plot_outputs_bars, get_accum, \
    plot_training
from source.analyze.anomaly import get_ascores_entropy, get_ascores_naive, \
    get_ascores_pyod, get_ascore_pyod, get_subjects_wldiffs, get_f1score, get_ascore_entropy, get_entropy_ts
from ts_source.utils.utils import add_timecol, load_config, load_models as load_models_darts

from ts_source.model.model import get_model_lag, LAG_MIN, get_modname, get_preds_rolling
from htm_source.pipeline.htm_batch_runner import run_batch
from htm_source.utils.fs import load_models as load_models_htm, save_models


FILETYPES_READFUNCS = {
    'xls': pd.read_excel,
    'csv': pd.read_csv
}


# def get_wllevels_outputs(modnames_filenames_ascores, modnames_filenames_pcounts, wllevels_filenames,
#                          features_models, levels_order=['baseline', 'distraction', 'rain', 'fog']):
def get_wllevels_outputs(filenames_ascores, filenames_pcounts, wllevels_filenames, levels_order=['baseline', 'distraction', 'rain', 'fog']):

    # wllevels_totals = {wllevel: [] for wllevel in wllevels_filenames if wllevel != 'training'}
    # modnames_wllevels_ascores = {modname: wllevels_totals for modname in features_models}
    # modnames_wllevels_pcounts = {modname: wllevels_totals for modname in features_models}
    # modnames_wllevels_totalascores = {modname: wllevels_totals for modname in features_models}

    wllevels_ascores = {wllevel: [] for wllevel in wllevels_filenames if wllevel != 'training'}
    wllevels_pcounts = {wllevel: [] for wllevel in wllevels_filenames if wllevel != 'training'}

    filenames_wllevels = {}
    for level, fns in wllevels_filenames.items():
        if level == 'training':
            continue
        for fn in fns:
            filenames_wllevels[fn] = level

    for fn, ascores in filenames_ascores.items():
        wllevel = filenames_wllevels[fn]
        wllevels_ascores[wllevel] += ascores
        wllevels_pcounts[wllevel] += filenames_pcounts[fn]

    wllevels_totalascores = {wllevel: np.sum(ascores) for wllevel, ascores in wllevels_ascores.items()}
    wllevels_totalascores_ = {}
    for k in levels_order:
        wllevels_totalascores_[k] = wllevels_totalascores[k]

    # for modname, filenames_ascores in modnames_filenames_ascores.items():
    #     filenames_ascores = {fn:ascores for fn,ascores in filenames_ascores.items() if 'static' in fn}
    #     for fn, ascores in filenames_ascores.items():
    #         wllevel = filenames_wllevels[fn]
    #         pcounts = modnames_filenames_pcounts[modname][fn]
    #         modnames_wllevels_ascores[modname][wllevel] += ascores
    #         modnames_wllevels_pcounts[modname][wllevel] += pcounts

    # for modname, wllevels_ascores in modnames_wllevels_ascores.items():
    #     wllevels_totalascores = {wllevel: np.sum(ascores) for wllevel, ascores in wllevels_ascores.items()}
    #     wllevels_totalascores_ = {}
    #     for k in levels_order:
    #         wllevels_totalascores_[k] = wllevels_totalascores[k]
    #     modnames_wllevels_totalascores[modname] = wllevels_totalascores_

    return wllevels_ascores, wllevels_pcounts, wllevels_totalascores  #modnames_wllevels_ascores, modnames_wllevels_pcounts, modnames_wllevels_totalascores


def get_filenames_outputs(cfg,
                          modname,
                          filenames_data,
                          modname_model):

    filenames_ascores = {}
    filenames_pcounts = {}

    for fn, data in filenames_data.items():
        if 'static' not in fn:
            continue
        if cfg['alg'] == 'HTM':
            cfg_htm_user = {k:v for k,v in cfg['htm_config_user'].items()}
            # if model_for_each_feature - drop from htm_user['feautres'] all but modname & 'timestamp'
            if 'megamodel' not in modname:
                feats_models = [modname, 'timestamp']
                cfg_htm_user['features'] = {k:v for k,v in cfg_htm_user['features'].items() if k in feats_models}
            feats_models, features_outputs = run_batch(cfg_user=cfg_htm_user,
                                                       cfg_model=cfg['htm_config_model'],
                                                       config_path_user=None,
                                                       config_path_model=None,
                                                       learn=cfg['learn_in_testing'],
                                                       data=data,
                                                       iter_print=1000,
                                                       features_models=modname_model)
            filenames_ascores[fn] = features_outputs[modname]['anomaly_score']
            filenames_pcounts[fn] = features_outputs[modname]['pred_count']

        # elif config['alg'] == 'SteeringEntropy':
        #     ascores = get_ascores_entropy(data[cfg['columns_model'][0]].values)  # data['steering angle'].values
        #
        # elif config['alg'] == 'Naive':
        #     ascores = get_ascores_naive(data[cfg['columns_model'][0]].values)  # data['steering angle'].values
        #
        # elif config['alg'] in ['IForest', 'OCSVM', 'KNN', 'LOF', 'AE', 'VAE', 'KDE']:
        #     ascores = get_ascores_pyod(data[cfg['columns_model']], features_models[cfg['columns_model'][0]])
        #
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

        # filenames_ascores[fn] = ascores
        # data.drop(columns=[config['time_col']], inplace=True)

    return filenames_ascores, filenames_pcounts


def run_subject(cfg, modname, df_train, dir_out, filenames_data, modname_model):
    # make subj dirs
    outnames_dirs = make_dirs_subj(dir_out, outputs=['anomaly', 'data_files', 'data_plots', 'models'])
    # save models
    save_models(modname_model, outnames_dirs['models'])
    # plot training data
    plot_training(df_train, columns_model=cfg['columns_model'], out_dir_plots=outnames_dirs['data_plots'], out_dir_files=outnames_dirs['data_files'])
    # plot filenames_data
    make_data_plots(filenames_data=filenames_data, file_type=cfg['file_type'], out_dir_plots=outnames_dirs['data_plots'])
    # get behavior data (dfs) for all wllevels
    wllevels_totaldfs = get_wllevels_totaldfs(wllevels_filenames=cfg['wllevels_filenames'],
                                              filenames_data=filenames_data,
                                              columns_model=cfg['columns_model'],
                                              out_dir_files=outnames_dirs['data_files'])
    # get indicies dividing the wllevels
    wllevels_indsend = get_wllevels_indsend(wllevels_totaldfs)
    # get model outputs for each file
    filenames_ascores, filenames_pcounts = get_filenames_outputs(cfg=config,
                                                                 modname=modname,
                                                                 filenames_data=filenames_data,
                                                                 modname_model=modname_model)

    # agg ascore to wllevel
    wllevels_ascores, wllevels_pcounts, wllevels_totalascores = get_wllevels_outputs(filenames_ascores=filenames_ascores,
                                                                                     filenames_pcounts=filenames_pcounts,
                                                                                     wllevels_filenames=config['wllevels_filenames'])

    # write outputs
    print(f"  Writing outputs to --> {outnames_dirs['anomaly']}")
    print("    Boxplots...")
    make_boxplots(data_dict=wllevels_ascores,
                  ylabel=f"{cfg['alg']} WL",
                  title=f"{cfg['alg']} WL Scores by Run Mode",
                  suptitle=None,
                  path_out=os.path.join(outnames_dirs['anomaly'], "levels--aScoreTotals--box.png"),
                  ylim=None)
    plot_outputs_bars(mydict=wllevels_totalascores,
                      title=f"{cfg['alg']} WL Scores by Run Mode",
                      xlabel='WL Levels',
                      ylabel=f"{cfg['alg']} WL",
                      path_out=os.path.join(outnames_dirs['anomaly'], "levels--aScoreTotals--bar.png"),
                      xtickrotation=0,
                      colors=['grey', 'blue', 'green', 'orange'])
    plot_outputs_boxes(data_plot1=wllevels_ascores,
                       data_plot2=wllevels_pcounts,
                       title_1=f"Anomaly Scores by WL Level\nhz={cfg['hzs']['convertto']},; features={cfg['columns_model']}",
                       title_2=f"Prediction Counts by WL Level\nhz={cfg['hzs']['convertto']},; features={cfg['columns_model']}",
                       out_dir=outnames_dirs['anomaly'],
                       xlabel='WL Levels',
                       ylabel='HTM Metric')
    print("    Lineplots...")
    plot_outputs_lines(wllevels_anomscores_=wllevels_ascores,
                       wllevels_predcounts_=wllevels_pcounts,
                       wllevels_indsend=wllevels_indsend,
                       get_pcounts=True if cfg['alg'] == 'HTM' else False,
                       out_dir=outnames_dirs['anomaly'])

    return wllevels_ascores


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
    subj = list(subjects_filenames_data.keys())[0]
    modnames = list(subjects_features_models[subj].keys())
    print(f'\nrun_posthoc; modnames = {modnames}')
    for modname in modnames:
        dir_out_modname = os.path.join(dir_out, f"modname={modname}")
        os.makedirs(dir_out_modname, exist_ok=True)
        print(f"  --> {dir_out_modname}")
        subjects_wllevels_ascores = {}
        for subj, filenames_data in subjects_filenames_data.items():
            dir_out_subj = os.path.join(dir_out_modname, subj)
            modname_model = {modname: subjects_features_models[subj][modname]}
            wllevels_ascores = run_subject(cfg=cfg,
                                           modname=modname,
                                           df_train=subjects_dfs_train[subj],
                                           dir_out=dir_out_subj,
                                           filenames_data=filenames_data,
                                           modname_model=modname_model)
            subjects_wllevels_ascores[subj] = wllevels_ascores

        # Write results for each modname
        subjects_wldiffs, subjects_levels_wldiffs = get_subjects_wldiffs(subjects_wllevels_ascores)
        percent_change_from_baseline, subjects_increased_from_baseline, subjects_baseline_lowest = get_scores(
            subjects_wldiffs, subjects_wllevels_ascores)
        percent_subjects_baseline_lowest = round(100 * len(subjects_baseline_lowest) / len(subjects_wllevels_ascores))

        # get overlap w/TLX
        subscales_meanoverlaps, df_overlaps = get_tlx_overlaps(subjects_wllevels_ascores)

        # rename dirs based on scores
        rename_dirs_by_scores(subjects_wldiffs, subjects_increased_from_baseline, subjects_baseline_lowest, dir_out_modname)

        # Save Results
        path_out_scores = os.path.join(dir_out_modname, 'scores.csv')
        path_out_subjects_wldiffs = os.path.join(dir_out_modname, 'subjects_wldiffs.csv')
        path_out_subjects_levels_wldiffs = os.path.join(dir_out_modname, 'subjects_levels_wldiffs.csv')
        path_out_subjects_overlaps = os.path.join(dir_out_modname, 'subjects_tlxoverlaps.csv')
        scores = {'Total sensitivity to increased task demands': percent_change_from_baseline,
                  'Rate of subjects with baseline lowest': percent_subjects_baseline_lowest}
        for subscale, meanoverlap in subscales_meanoverlaps.items():
            scores[subscale] = meanoverlap
        scores = pd.DataFrame(scores, index=[0])
        df_subjects_levels_wldiffs = pd.DataFrame(subjects_levels_wldiffs)
        df_subjects_wldiffs = pd.DataFrame(subjects_wldiffs, index=[0]).T
        df_subjects_wldiffs.columns = ['Difference from Baseline']
        scores.to_csv(path_out_scores)
        df_overlaps.to_csv(path_out_subjects_overlaps, index=True)
        df_subjects_wldiffs.to_csv(path_out_subjects_wldiffs, index=True)
        df_subjects_levels_wldiffs.to_csv(path_out_subjects_levels_wldiffs, index=True)
        make_save_plots(dir_out=dir_out_modname,
                        subjects_wldiffs=subjects_wldiffs,
                        percent_change_from_baseline=percent_change_from_baseline,
                        subjects_wllevelsascores=subjects_wllevels_ascores)
    return percent_change_from_baseline


def run_realtime(config, dir_out, subjects_features_models):
    rows = list()
    for subj, features_models in subjects_features_models.items():
        print(f"\n  subj = {subj}")
        dir_out_subj = os.path.join(dir_out, subj)
        dir_in_subj = os.path.join(config['dirs']['input'], subj)
        # print("    testing...")
        for feat, model in features_models.items():
            print(f"    feat = {feat}")
            for testfile, times in config['subjects_testfiles_wltogglepoints'][subj].items():
                print(f"      testfile = {testfile}")
                aScores, wl_changepoints_detected = list(), list()
                path_test = os.path.join(dir_in_subj, testfile)
                data_test = config['read_func'](path_test)
                data_test.columns = config['colnames']
                cols_drop = [config['time_col']] + [c for c in config['colnames'] if c not in config['columns_model']]
                data_test.drop(columns=cols_drop, inplace=True)

                # Proprocess data -- DON'T select_by_autocorr() since it'll make the wl_changepoints invalid
                # Agg
                agg = int(config['hzs']['baseline'] / config['hzs']['convertto'])
                data_test = data_test.groupby(data_test.index // agg).mean()
                # Clip both ends
                clip_count_start = int(data_test.shape[0] * (config['clip_percents']['start'] / 100))
                clip_count_end = data_test.shape[0] - int(data_test.shape[0] * (config['clip_percents']['end'] / 100))
                data_test = data_test[clip_count_start:clip_count_end]
                # Subtract mean
                feats_medians = {feat: m for feat, m in dict(data_test.median()).items()}
                for feat, median in feats_medians.items():
                    data_test[feat] = data_test[feat] - median
                # Transform
                data_test = prep_data(data_test, config['preprocess'])
                # Add Timecol
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
                print(f"        wl_changepoints_detected = {wl_changepoints_detected}")
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
                                    aScores,
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
                    subjects_wldiffs,
                    percent_change_from_baseline,
                    subjects_wllevelsascores):
    # subjects WLdiffs
    plot_outputs_bars(mydict=subjects_wldiffs,
                      title=f'Total WL Difference = {round(percent_change_from_baseline, 3)}',
                      xlabel='Subjects',
                      ylabel='WL Difference from Level 0 to 1-3',
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
            if _ in rows_neg:
                rows_neg.remove(_)
    true_neg = [v for v in rows_neg if v not in wl_changepoints_detected]
    scores['true_neg'] = len(true_neg)

    return scores, wl_changepoints_windows


def plot_wlchangepoints(columns_model, file_type, aScores, testfile, data_test, dir_out_subj, wl_changepoints_windows,
                        wl_changepoints_detected):
    for feat in columns_model:
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
        ax2.set_ylabel('WL Perceived', color=color)  # we already handled the x-label with ax1
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


def get_subjects_data(cfg, subjects, subjects_spacesadd, dir_out):
    print('Gathering subjects data...')
    subjects_filenames_data = dict()
    subjects_dfs_train = dict()
    for subj in sorted(subjects):
        dir_input = os.path.join(cfg['dirs']['input'], subj)
        # Load
        filenames_data = load_files(dir_input=dir_input, file_type=cfg['file_type'], read_func=cfg['read_func'])
        # Update columns
        filenames_data = update_colnames(filenames_data=filenames_data, colnames=cfg['colnames'])
        # Preprocess
        filenames_data, df_train = preprocess_data(subj, cfg, filenames_data, subjects_spacesadd)
        # Store
        subjects_dfs_train[subj] = df_train
        subjects_filenames_data[subj] = filenames_data

    return subjects_dfs_train, subjects_filenames_data


def get_subjects_models(config, dir_out, subjects_dfs_train):
    print('Training subjects models...')
    subjects_features_models = dict()
    for subj, df_train in subjects_dfs_train.items():
        print(f"  --> {subj}")
        # Train & save model(s)
        if config['train_models']:
            features_models = train_save_models(df_train=df_train,
                                                alg=config['alg'],
                                                dir_output= dir_out,  #dir_out_subj_models,
                                                config=config,
                                                htm_config_user=config['htm_config_user'],
                                                htm_config_model=config['htm_config_model'])
        # Load model(s)
        else:
            if config['alg'] == 'HTM':
                features_models = load_models_htm(dir_out)  #dir_out_subj_models
            else:
                features_models = load_models_darts(dir_out, alg=config['alg'])  #dir_out_subj_models
        # Store models
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

    subjects_all = ['aranoff', 'balaji']

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

    # # Make subjects' output dirs
    # for subj in subjects:
    #     """ nest into /modname={modname} """
    #     make_dirs_subj(os.path.join(dir_out, subj))

    # Get subjects data
    subjects_dfs_train, subjects_filenames_data = get_subjects_data(config, subjects, subjects_spacesadd, dir_out)

    # Train subjects models
    subjects_features_models = get_subjects_models(config, dir_out, subjects_dfs_train)
    print(f"\nsubjects_features_models = {subjects_features_models}")

    # Run
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
