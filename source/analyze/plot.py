import operator
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_TS_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ts_forecaster')
sys.path.append(_SOURCE_DIR)
sys.path.append(_TS_SOURCE_DIR)

from source.preprocess.preprocess import preprocess_data, diff_data, standardize_data, movingavg_data
from ts_source.utils.utils import load_config


def make_data_plots(filenames_data: dict, modname: str, columns_model: list, file_type: str, out_dir_plots: str):
    columns_plot = columns_model if 'megamodel' in modname else [modname]
    for fname, data in filenames_data.items():
        fname = fname.replace(f".{file_type}", "")
        path_out = os.path.join(out_dir_plots, f"{fname}.png")
        plt.figure(figsize=(15, 3))
        for c in columns_plot:
            plt.plot(data[c], label=c)
        plt.title(fname)
        plt.legend()
        plt.savefig(path_out)
        plt.close()


def plot_outputs_boxes(data_plot1, data_plot2, levels_colors, title_1, title_2, out_dir, xlabel, ylabel):
    outtypes_paths = {'aScores': os.path.join(out_dir, 'wllevels_aScores.png'),
                      'pCounts': os.path.join(out_dir, 'wllevels_pCounts.png')}
    # Plot -- violin
    outtypes_data = {ot: [] for ot in outtypes_paths}
    for testlevel, ascores in data_plot1.items():
        df_dict_1 = {'Task WL': [testlevel for _ in range(len(ascores))],
                     'Anomaly Score': ascores}
        df1 = pd.DataFrame(df_dict_1)
        outtypes_data['aScores'].append(df1)
        if data_plot2 != {}:
            pcounts = data_plot2[testlevel]
            df_dict_2 = {'Task WL': [testlevel for _ in range(len(pcounts))],
                         'Prediction Count': pcounts}
            df2 = pd.DataFrame(df_dict_2)
            outtypes_data['pCounts'].append(df2)
    df_1 = pd.concat(outtypes_data['aScores'], axis=0)
    if data_plot2 != {}:
        df_2 = pd.concat(outtypes_data['pCounts'], axis=0)
    vplot_anom = sns.violinplot(data=df_1,
                                x="Task WL",
                                y='Anomaly Score',
                                pallette=levels_colors)
    plt.title(title_1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1.0)
    plt.savefig(outtypes_paths['aScores'].replace('.png', '--violin.png'), bbox_inches="tight")
    plt.close()
    if len(data_plot2['baseline']) > 0:
        vplot_pred = sns.violinplot(data=df_2,
                                    x="Task WL",
                                    y='Prediction Count',
                                    pallette=levels_colors)
        plt.title(title_2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(outtypes_paths['pCounts'].replace('.png', '--violin.png'), bbox_inches="tight")
        plt.close()

    # Plot -- box
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    outtypes_ddicts = {'aScores': data_plot1}
    if data_plot2 != {}:
        outtypes_ddicts['pCounts'] = data_plot2
    for outtype, ddict in outtypes_ddicts.items():
        fig, ax = plt.subplots()
        ax.boxplot(ddict.values())
        ax.set_xticklabels(ddict.keys(), rotation=90)
        ax.yaxis.grid(True)
        outp = outtypes_paths[outtype].replace('.png', '--box.png')
        plt.savefig(outp, bbox_inches="tight")
        plt.close()


# def plot_wllevels_totaldfs(wllevels_totaldfs, columns_model, colors=['grey', 'blue', 'orange', 'red']):
#     assert len(wllevels_totaldfs) <= len(colors), f"more wllevels found ({len(wllevels_totaldfs)}) than colors ({len(colors)})"
#     colors = ['grey', 'blue', 'orange', 'red']
#     levels_colors = {}
#     for wllevel in wllevels_totaldfs:
#         color = colors[0]
#         levels_colors[wllevel] = color
#         colors.remove(color)
#
#     ## plot - behavior (total)
#     alldata_task = pd.concat(list(wllevels_totaldfs.values()), axis=0)
#     for feat in columns_model:
#         feat_data = alldata_task[feat].values
#         f_min, f_max = min(feat_data), max(feat_data)
#         # plt.cla()
#         plt.figure(figsize=(15, 3))
#         # plt.rcParams['axes.facecolor'] = 'grey'
#         plt.plot(feat_data)
#         plt.title(f'Behavior - Validation')
#         # plt.xlabel('time')
#         plt.ylabel(feat)
#         ind_prev = 0
#         for wllevel, ind in wllevels_indsend.items():
#             plt.axvline(x=ind, color='r', linestyle='--')
#             plt.axvspan(ind_prev, ind, facecolor=levels_colors[wllevel], alpha=0.5, label=wllevel)
#             ind_prev = ind
#         # plt.grid(False)
#         plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
#         out_path = os.path.join(out_dir, f'time--validation--{feat}.png')
#         plt.savefig(out_path)
#         plt.close()
#
#         ## by level
#         for level, data in wllevels_totaldfs.items():
#             d_feat = data[feat].values
#             # plt.cla()
#             plt.figure(figsize=(15, 3))
#             plt.plot(d_feat, color=levels_colors[level])
#             # plt.xlabel('time')
#             plt.ylabel(feat)
#             plt.ylim(f_min, f_max)
#             # plt.title(f'{level}')  #f'Behavior - {feat} - {level}'
#             out_path = os.path.join(out_dir, f'time--validation--{feat}--{level}.png')
#             plt.savefig(out_path)
#             plt.close()


def plot_write_data(df: pd.DataFrame,
                    out_dir_plots: str,
                    out_dir_files: str,
                    out_name: str):
    df.to_csv(os.path.join(out_dir_files, f"{out_name}.csv"))  # 'training.csv'
    for feat in df:
        plt.figure(figsize=(15, 3))
        plt.plot(df[feat].values)
        plt.title(f"Behavior - {out_name}")
        plt.xlabel('time')
        plt.ylabel(feat)
        out_path = os.path.join(out_dir_plots, f'{out_name}--all--{feat}.png')
        plt.savefig(out_path)
        plt.close()


def plot_outputs_lines(wllevels_ascores_: dict,
                       wllevels_pcounts_: dict,
                       # wllevels_indsend: dict,
                       columns_model,
                       filenames_data: dict,
                       filenames_ascores: dict,
                       filenames_wllevels: dict,
                       get_pcounts: bool,
                       levels_order: list,
                       levels_colors: dict,
                       out_dir: str):
    # REORDER --> wllevels_alldata, wllevels_anomscores, wllevels_pcounts
    wllevels_ascores, wllevels_pcounts = {}, {}
    for k in levels_order:
        wllevels_ascores[k] = wllevels_ascores_[k]
        if k in wllevels_pcounts_:
            wllevels_pcounts[k] = wllevels_pcounts_[k]

    # TEST
    ## plot - aScores accum (by filename)
    out_dir_filenames = os.path.join(out_dir, 'filenames--aScoresAccum--time')
    os.makedirs(out_dir_filenames)
    for fn, ascores in filenames_ascores.items():
        ascores_accum = get_accum(ascores)

        for feat in columns_model:
            behavior = filenames_data[fn][feat]
            fig, ax1 = plt.subplots()

            second_axis = False
            prop = len(ascores) / len(behavior)
            if prop > 0.95:
                second_axis = True
                behavior = behavior[-len(ascores):]
            t = [_ for _ in range(len(behavior))]

            color1 = 'cornflowerblue'
            ax1.set_xlabel('time')
            ax1.set_ylabel(feat, color=color1)
            ax1.plot(t, behavior, color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)

            if second_axis:
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                color2 = 'black'
                ax2.set_ylabel('Perceived WL', color=color2)  # we already handled the x-label with ax1
                ax2.plot(t, ascores_accum, color=color2)
                ax2.tick_params(axis='y', labelcolor=color2)

            plt.plot(ascores_accum)
            plt.title("Perceived WL")
            plt.xlabel('Time')
            plt.ylabel('Accumulated Anomaly Scores')
            out_path = os.path.join(out_dir_filenames, f'{filenames_wllevels[fn]}--{fn}--{feat}.png')
            plt.savefig(out_path)
            plt.close()

    ## get data & wl inds
    # ascores_accum, pcounts_accum = [], []
    wllevels_ascoresaccum = {}
    for wllevel, ascores in wllevels_ascores.items():
        ascores_accum_wllevel = get_accum(ascores)
        # ascores_accum += ascores_accum_wllevel
        wllevels_ascoresaccum[wllevel] = ascores_accum_wllevel
        # if get_pcounts:
        #     pcounts_accum += get_accum(wllevels_pcounts[wllevel])

    wllevels_pcountsaccum = {}
    for wllevel, pcounts in wllevels_pcounts.items():
        pcounts_accum_wllevel = get_accum(pcounts)
        # pcounts_accum += pcounts_accum_wllevel
        wllevels_pcountsaccum[wllevel] = pcounts_accum_wllevel

    # ## plot - aScores accum (by wllevel)
    # plt.plot(ascores_accum)
    # plt.title("Perceived WL")
    # plt.xlabel('Time')
    # plt.ylabel('Accumulated Anomaly Scores')
    # prev_ind = 0
    # for wllevel, ind in wllevels_indsend.items():
    #     plt.axvline(x=ind, color='r', label=wllevel)
    #     fdata_level = wllevels_ascores[wllevel]
    #     loc_x = np.percentile([ind, prev_ind], 25)
    #     loc_y = np.percentile(fdata_level, 25)
    #     fdata_total = round(np.sum(fdata_level), 3)
    #     plt.text(loc_x, loc_y, f"total={fdata_total}")
    #     prev_ind = ind
    # out_path = os.path.join(out_dir, f'time--validation--aScores.png')
    # plt.savefig(out_path)
    # plt.close()

    ## plot - aScores accum overlapped
    plt.figure(figsize=(15, 3))
    max_ascoreaccum = 0
    for wllevel, ascoresaccum in wllevels_ascoresaccum.items():
        wllevel_ascores_total = round(np.sum(wllevels_ascores[wllevel]), 4)
        plt.plot(ascoresaccum, label=f"{wllevel} (total={wllevel_ascores_total})", color=levels_colors[wllevel])
        max_ascoreaccum = max(max_ascoreaccum, max(ascoresaccum))
        # loc_x = len(ascoresaccum)
        # loc_y = ascoresaccum[-1]
        # plt.text(loc_x, loc_y, f"avg={wllevel_ascores_mean}")
    max_ascoreaccum = max_ascoreaccum
    plt.ylim(0, max_ascoreaccum)
    plt.title("Perceived WL by Task Level")
    plt.ylabel('Accumulated Anomaly Scores')
    plt.legend()
    out_path = os.path.join(out_dir, f'wllevels--aScoresAccum--time.png')
    plt.savefig(out_path)
    plt.close()

    ## plot - pCounts accum overlapped
    if get_pcounts:
        plt.figure(figsize=(15, 3))
        max_pcountaccum = 0
        for wllevel, pcountssaccum in wllevels_pcountsaccum.items():
            wllevel_pcounts_total = round(np.sum(wllevels_pcounts[wllevel]), 4)
            plt.plot(pcountssaccum, label=f"{wllevel} (total={wllevel_pcounts_total})", color=levels_colors[wllevel])
            max_pcountaccum = max(max_pcountaccum, max(pcountssaccum))
            # loc_x = len(ascoresaccum)
            # loc_y = ascoresaccum[-1]
            # plt.text(loc_x, loc_y, f"avg={wllevel_ascores_mean}")
        max_pcountaccum = max_pcountaccum
        plt.ylim(0, max_pcountaccum)
        plt.title("Prediction Counts by Task Level")
        plt.ylabel('Accumulated Predictions Counts')
        plt.legend()
        out_path = os.path.join(out_dir, f'wllevels--pCountsAccum--time.png')
        plt.savefig(out_path)
        plt.close()

    # ## plot - pCounts
    # if get_pcounts:
    #     plt.plot(pcounts_accum)
    #     plt.title("Perceived WL")
    #     plt.xlabel('Time')
    #     plt.ylabel('Accumulated Prediction Counts')
    #     prev_ind = 0
    #     for wllevel, ind in wllevels_indsend.items():
    #         plt.axvline(x=ind, color='r', label=wllevel)
    #         fdata_level = wllevels_pcounts[wllevel]
    #         loc_x = np.percentile([ind, prev_ind], 25)
    #         loc_y = np.percentile(fdata_level, 25)
    #         fdata_total = round(np.sum(fdata_level), 3)
    #         plt.text(loc_x, loc_y, f"total={fdata_total}")
    #         prev_ind = ind
    #     out_path = os.path.join(out_dir, f'pCountsAccum--time--wllevels.png')
    #     plt.savefig(out_path)
    #     plt.close()


def plot_hists(algs_data, dir_out, title, density=False):
    plt.cla()
    for alg, data in algs_data.items():
        plt.hist(data, label=f'{alg}', alpha=0.5, density=density)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.tight_layout()
    path_out = os.path.join(dir_out, f"{title}.png")
    plt.savefig(path_out)
    plt.close()


def plot_outputs_bars(mydict, levels_colors, title, xlabel, ylabel, path_out, xtickrotation=0, print_barheights=True):
    colors = list(levels_colors.values()) if levels_colors else None
    mydict = {k: round(v, 3) for k, v in mydict.items()}
    plt.cla()
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.bar(range(len(mydict)), list(mydict.values()), align='center', color=colors, alpha=0.5)
    plt.xticks(range(len(mydict)), list(mydict.keys()), rotation=xtickrotation)
    if print_barheights:
        xlocs = [i + 1 for i in range(0, len(mydict))]
        for i, v in enumerate(list(mydict.values())):
            plt.text(xlocs[i] - 1.15, v + 0.1, str(v))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path_out, bbox_inches="tight")
    plt.close()


def get_accum(values):
    accum_vals, accum_sum = [], 0
    for v in values:
        accum_vals.append(v + accum_sum)
        accum_sum += v
    return accum_vals


def plot_subjects_timeserieses(dir_data, dir_out, path_cfg):
    cfg = load_config(path_cfg)
    subjects_found = [f for f in os.listdir(dir_data) if os.path.isdir(os.path.join(dir_data, f))]
    path_realtimewl = os.path.join(dir_data, 'realtime_wl.csv')
    realtime_wl = pd.read_csv(path_realtimewl)
    for subj in subjects_found:
        print(f"{subj}...")
        dir_data_subj = os.path.join(dir_data, subj)
        dir_out_subj = os.path.join(dir_out, subj)
        realtime_wl_subj = realtime_wl[realtime_wl['Subject'] == subj]
        os.makedirs(dir_out_subj, exist_ok=True)
        plot_subject_timeserieses(cfg['preprocess'], cfg['file_type'], dir_data_subj, dir_out_subj, realtime_wl_subj,
                                  hz_baseline=cfg['hzs']['baseline'], hz_convertto=cfg['hzs']['convertto'])


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def plot_subject_timeserieses(cfg_prep, file_type, dir_data_subj, dir_out_subj, realtime_wl_subj, hz_baseline=100,
                              hz_convertto=3):
    dir_files = [f for f in os.listdir(dir_data_subj) if f'.{file_type}' in f]
    files_train = [f for f in dir_files if 'training' in f]
    files_static = [f for f in dir_files if 'static' in f]
    files_realtime = [f for f in dir_files if 'realtime' in f]
    files_train.sort(key=natural_keys)
    files_static.sort(key=natural_keys)
    files_realtime.sort(key=natural_keys)
    # training
    plot_timeseries_combined(cfg_prep=cfg_prep, file_type=file_type, files=files_train, title='Training',
                             dir_data_subj=dir_data_subj, dir_out_subj=dir_out_subj, hz_baseline=hz_baseline,
                             hz_convertto=hz_convertto)
    # static
    plot_timeseries_combined(cfg_prep=cfg_prep, file_type=file_type, files=files_static, title='Static',
                             dir_data_subj=dir_data_subj, dir_out_subj=dir_out_subj, hz_baseline=hz_baseline,
                             hz_convertto=hz_convertto)
    # realtime
    plot_timeseries(cfg_prep=cfg_prep, file_type=file_type, files=files_realtime, title='RealTime',
                    dir_data_subj=dir_data_subj, dir_out_subj=dir_out_subj, realtime_wl_subj=realtime_wl_subj,
                    hz_baseline=hz_baseline, hz_convertto=hz_convertto)


def plot_timeseries(cfg_prep, file_type, files, title, dir_data_subj, dir_out_subj, realtime_wl_subj, hz_baseline=100,
                    hz_convertto=6.67):
    agg = int(hz_baseline / hz_convertto)
    # ind_prev = 0
    for fn in files:
        # import
        dpath = os.path.join(dir_data_subj, fn)
        fn = fn.replace(f'.{file_type}', '')
        data = pd.read_excel(dpath)
        """ BUG --> HARD CODING"""
        data.columns = ['time', 'steering angle', 'break']
        # agg
        data = data.groupby(data.index // agg).mean()
        # subtract mean
        mean_ = np.mean(data['steering angle'])
        data['steering angle'] = [v - mean_ for v in data['steering angle']]
        # preprocess
        filenames_data = preprocess_data({'data': data}, cfg_prep)
        data = filenames_data['data']
        # drop low autocorr timesteps
        diff_pcts = get_autocorrs(data['steering angle'].values)
        data_selected = select_by_autocorr(data['steering angle'].values, diff_pcts,
                                           diff_thresh=cfg_prep['autocorr_thresh'])
        data = pd.DataFrame({'steering angle': data_selected})
        # get times wl imposed
        run_number = int(fn.split('realtime')[1])
        wl_imposed1_col = f"Run {run_number}-1"
        wl_imposed2_col = f"Run {run_number}-2"
        wl_runtime_col = f"Run {run_number}-Total"
        wl_imposed1_seconds = realtime_wl_subj[wl_imposed1_col].values[0]
        wl_imposed2_seconds = realtime_wl_subj[wl_imposed2_col].values[0]
        wl_total_seconds = realtime_wl_subj[wl_runtime_col].values[0]
        wl_imposed1 = int((wl_imposed1_seconds / wl_total_seconds) * data.shape[0])
        wl_imposed2 = int((wl_imposed2_seconds / wl_total_seconds) * data.shape[0])
        # plot
        steering_angles = data['steering angle'].values
        # plt.cla()
        plt.plot(steering_angles)
        plt.axvline(x=wl_imposed1, color='orange', linestyle='-.', linewidth=1, label='WL Imposed 1')
        plt.axvline(x=wl_imposed2, color='green', linestyle='-.', linewidth=1, label='WL Imposed 2')
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.title(title)
        plt.tight_layout()
        path_out = os.path.join(dir_out_subj, f"{title}--{fn}.png")
        plt.savefig(path_out)
        plt.close()


def plot_timeseries_combined(cfg_prep, file_type, files, dir_data_subj, dir_out_subj, title='Training', hz_baseline=100,
                             hz_convertto=6.67):
    """ BUG --> HARD CODING """
    files_types = {'static1': 'rain',
                   'static2': 'fog',
                   'static3': 'pennies',
                   'static4': 'baseline',
                   'static5': 'fog',
                   'static6': 'rain',
                   'static7': 'pennies',
                   'static8': 'fog',
                   'static9': 'baseline',
                   'static10': 'pennies',
                   'static11': 'rain',
                   'static12': 'baseline'
                   }
    types_colors = {'rain': 'blue',
                    'fog': 'grey',
                    'pennies': 'orange',
                    'baseline': 'white'}
    agg = int(hz_baseline / hz_convertto)
    count = 0
    filenames_vlineindices = {}
    datas = []
    for f in files:
        dpath = os.path.join(dir_data_subj, f)
        data = pd.read_excel(dpath)
        data.columns = ['time', 'steering angle', 'break']
        # agg
        data = data.groupby(data.index // agg).mean()
        # subtract mean
        mean_ = np.mean(data['steering angle'])
        data['steering angle'] = [v - mean_ for v in data['steering angle']]
        # preprocess
        filenames_data = preprocess_data({'data': data}, cfg_prep)
        data = filenames_data['data']
        datas.append(data)
        vline_ind = len(data) + count
        f = f.replace(f'.{file_type}', '').replace('--', '').replace('crash', '')
        filenames_vlineindices[f] = int(vline_ind)
        count += len(data)
    data_total = pd.concat(datas, axis=0)

    # drop low autocorr timesteps
    diff_pcts = get_autocorrs(data_total['steering angle'].values)
    data_selected = select_by_autocorr(data_total['steering angle'].values, diff_pcts,
                                       diff_thresh=cfg_prep['autocorr_thresh'])

    # steering_angles = data_total['steering angle'].values
    # plt.cla()
    plt.figure(figsize=(15, 3))
    plt.plot(data_selected)  # steering_angles
    # plot lines and color areas to separate runs and run types
    ind_prev = 0
    runtypes_labeled = []
    for fn, vl in filenames_vlineindices.items():
        plt.axvline(x=vl, color='r', linestyle='--')
        if title == 'Static':
            runtype = files_types[fn]
            color = types_colors[runtype]
            if runtype not in runtypes_labeled:
                plt.axvspan(ind_prev, vl, facecolor=color, alpha=0.5, label=runtype)
            else:
                plt.axvspan(ind_prev, vl, facecolor=color, alpha=0.5)
            runtypes_labeled.append(runtype)
        else:
            pass
        ind_prev = vl
    if title == 'Static':
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    path_out = os.path.join(dir_out_subj, f"{title}.png")
    plt.savefig(path_out)
    plt.close()


def prep_data(data, cfg_prep):
    if cfg_prep['difference']:
        data = diff_data(data, cfg_prep['difference'])
    if cfg_prep['standardize']:
        data = standardize_data(data)
    if cfg_prep['movingaverage']:
        data = movingavg_data(data, cfg_prep['movingaverage'])
    return data


def save_tsplot(data, cfg_prep, path_out):
    mean_ = np.mean(data)
    s_a = [v - mean_ for v in data]
    d = prep_data(s_a, cfg_prep)  # data
    # plt.cla()
    plt.figure(figsize=(15, 3))
    plt.plot(d)
    title = ""
    for k, v in cfg_prep.items():
        title += f"{k}={v}; "
    plt.title(title)
    plt.savefig(path_out)
    plt.close()


def get_autocorr(data_t1, data_t0):
    diff = data_t1 - data_t0
    diff_pct = (diff / data_t0) * 100
    return diff_pct


def get_autocorrs(steering_angles):
    diff_pcts = [0]
    for _, v in enumerate(steering_angles):
        if _ == 0:
            continue
        diff = get_autocorr(v, steering_angles[_ - 1])
        diff_pcts.append(diff)
    return diff_pcts


def select_by_autocorr(steering_angles, diff_pcts, diff_thresh=5):
    data_selected = []
    for _, v in enumerate(steering_angles):
        if diff_pcts[_] > diff_thresh:
            data_selected.append(v)
    return data_selected


def plot_venues(paper_min, venues_impactscores, venues_waittimes, path_in, path_out):
    # GET DATA
    df = pd.read_csv(path_in)
    venues_counts = dict(df['Publication Venue'].value_counts())
    venues_counts = dict(sorted(venues_counts.items(), key=operator.itemgetter(1), reverse=True))
    venues_counts2 = {k: v for k, v in venues_counts.items() if v >= paper_min}
    # PLOT
    # plt.cla()
    plt.figure(figsize=(20, 10))
    paper_counts = list(venues_counts2.values())
    wait_times = list(venues_waittimes.values())
    impact_scores = [v * 10 for v in list(venues_impactscores.values())]
    x = np.array([_ for _ in range(len(venues_counts2))])
    plt.bar(x - 0.2, paper_counts, width=0.2, color='b', align='center', label='Paper Counts', alpha=0.5)  # x - 0.5
    plt.bar(x, wait_times, width=0.2, color='g', align='center', label='Wait Times (weeks)', alpha=0.5)
    plt.bar(x + 0.2, impact_scores, width=0.2, color='r', align='center', label='Impact Scores (x10)',
            alpha=0.5)  # x + 0.5
    plt.xticks(range(len(venues_counts2)), list(venues_counts2.keys()), rotation=90)
    plt.title("Common Publication Venues - WL")
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(path_out)
    plt.close()


if __name__ == "__main__":
    venues_impactscores = {
        'Applied Ergonomics': 2.963,
        'Transportation Research Part F: Traffic Psychology and Behaviour': 3.153,
        'International Journal of Industrial Ergonomics': 2.705,
        'Frontiers in Human Neuroscience': 3.634,
        'Human Factors': 2.952,
        'IEEE Transactions on Neural Systems and Rehabilitation Engineering': 3.480,
        'Frontiers in Neuroscience': 3.711,
        'International Journal of Psychophysiology': 3.157,
        'Surgical Endoscopy': 3.570,
        'Journal of Medical Systems': 2.131,
        'IEEE Access': 3.745,
        'Accident Analysis & Prevention': 3.145,
        'Frontiers in Psychology': 2.848,
        'International Journal of Environmental Research and Public Health': 2.849,
        'Ergonomics': 2.526,
        'International Journal of Human-Computer Interaction': 1.597,
        'IEEE Transactions on Human-Machine Systems': 2.944,
        'Journal of Neural Engineering': 3.352,
        'International Journal of Human-Computer Studies': 2.799,
        'Journal of Cognitive Engineering and Decision Making': 1.541,
        'Human Factors and Ergonomics in Manufacturing & Service Industries': 1.058,
        'Applied Sciences': 3.689,
        'Aerospace Medicine and Human Performance': 1.743,
        'Sensors': 3.275,
        'PLoS One': 2.740,
        'Work: A Journal of Prevention, Assessment and Rehabilitation': 1.178,
        'Journal of Experimental Psychology: Applied': 2.931,
        'Journal of Medical Internet Research': 5.034,
        'Scientific Reports': 4.122,
        'Journal of Safety Research': 1.689,
        'Journal of Surgical Education': 2.101,
        'Journal of Ambient Intelligence and Humanized Computing': 2.385
    }
    venues_waittimes = {
        'Applied Ergonomics': 12,
        'Transportation Research Part F: Traffic Psychology and Behaviour': 12,
        'International Journal of Industrial Ergonomics': 14,
        'Frontiers in Human Neuroscience': 14,
        'Human Factors': 12,
        'IEEE Transactions on Neural Systems and Rehabilitation Engineering': 20,
        'Frontiers in Neuroscience': 14,
        'International Journal of Psychophysiology': 14,
        'Surgical Endoscopy': 14,
        'Journal of Medical Systems': 12,
        'IEEE Access': 10,
        'Accident Analysis & Prevention': 10,
        'Frontiers in Psychology': 14,
        'International Journal of Environmental Research and Public Health': 10,
        'Ergonomics': 14,
        'International Journal of Human-Computer Interaction': 12,
        'IEEE Transactions on Human-Machine Systems': 20,
        'Journal of Neural Engineering': 14,
        'International Journal of Human-Computer Studies': 14,
        'Journal of Cognitive Engineering and Decision Making': 12,
        'Human Factors and Ergonomics in Manufacturing & Service Industries': 14,
        'Applied Sciences': 14,
        'Aerospace Medicine and Human Performance': 14,
        'Sensors': 14,
        'PLoS One': 10,
        'Work: A Journal of Prevention, Assessment and Rehabilitation': 14,
        'Journal of Experimental Psychology: Applied': 12,
        'Journal of Medical Internet Research': 12,
        'Scientific Reports': 14,
        'Journal of Safety Research': 10,
        'Journal of Surgical Education': 14,
        'Journal of Ambient Intelligence and Humanized Computing': 14
    }
    plot_venues(paper_min=3,
                venues_impactscores=venues_impactscores,
                venues_waittimes=venues_waittimes,
                path_in="/Users/samheiserman/Desktop/MWL -- research by app. domain.csv",
                path_out="/Users/samheiserman/Desktop/VENUES_COUNTS.png")
    # plot_subjects_timeserieses(dir_data="/Users/samheiserman/Desktop/repos/workload_assessor/data",
    #                            dir_out="/Users/samheiserman/Desktop/repos/workload_assessor/eda",
    #                            path_cfg="/Users/samheiserman/Desktop/repos/workload_assessor/configs/run_pipeline.yaml")
