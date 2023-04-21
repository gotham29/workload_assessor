import os
import re
import sys
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# _SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
# sys.path.append(_SOURCE_DIR)


def plot_data(filenames_data: dict, file_type: str, dir_output: str):
    dir_out = os.path.join(dir_output, 'data_plots')
    os.makedirs(dir_out, exist_ok=True)
    for fname, data in filenames_data.items():
        fname = fname.replace(f".{file_type}", "")
        path_out = os.path.join(dir_out, f"{fname}.png")
        plt.cla()
        for c in data:
            plt.plot(data[c], label=c)
        plt.title(fname)
        plt.legend()
        plt.savefig(path_out)


def plot_boxes(data_plot1, data_plot2, title_1, title_2, out_dir, xlabel, ylabel):
    outtypes_paths = {'aScores': os.path.join(out_dir, 'TaskWL_aScores.png'),
                      'pCounts': os.path.join(out_dir, 'TaskWL_pCounts.png'),
                      'WLScores': os.path.join(out_dir, 'TaskWL_aggScore.png')}
    # Plot -- Violin
    outtypes_data = {ot: [] for ot in outtypes_paths if ot != 'WLScores'}
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
    plt.cla()
    vplot_anom = sns.violinplot(data=df_1,
                                x="Task WL",
                                y='Anomaly Score')
    plt.title(title_1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1.0)
    plt.savefig(outtypes_paths['aScores'].replace('.png', '--violin.png'), bbox_inches="tight")
    plt.cla()
    if data_plot2 != {}:
        vplot_pred = sns.violinplot(data=df_2,
                                    x="Task WL",
                                    y='Prediction Count')
        plt.title(title_2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(outtypes_paths['pCounts'].replace('.png', '--violin.png'), bbox_inches="tight")
    # Plot -- Box
    plt.cla()
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
    # Plot -- Combined
    if data_plot2 != {}:
        plt.cla()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ttypes_scores = {}
        for ttype, ascores in data_plot1.items():
            pcounts = data_plot2[ttype]
            if np.median(ascores) == 0:
                score = 0
            else:
                score = 1 / (np.median(pcounts) / np.median(ascores))
            ttypes_scores[ttype] = score
        plt.bar(range(len(ttypes_scores)), list(ttypes_scores.values()), align='center')
        plt.xticks(range(len(ttypes_scores)), list(ttypes_scores.keys()))
        plt.savefig(outtypes_paths['WLScores'], bbox_inches="tight")


def plot_lines(wllevels_anomscores: dict,
               wllevels_predcounts: dict,
               wllevels_alldata: dict,
               df_train: pd.DataFrame,
               get_pcounts: bool,
               columns_model: dict,
               out_dir: str):

    # TRAIN
    ## plot - columns_model
    for feat in columns_model:
        plt.cla()
        plt.plot(df_train[feat].values)
        plt.title(f'Behavior - Training')
        plt.xlabel('time')
        plt.ylabel(feat)
        out_path = os.path.join(out_dir, f'time--training--{feat}.png')
        plt.savefig(out_path)

    # TEST
    ## get data & wl inds
    alldata_task = pd.concat(list(wllevels_alldata.values()), axis=0)
    ascores_accum, pcounts_accum = [], []
    wllevels_indsend, wllevels_ascoresaccum, wllevel_i = {}, {}, 0
    for wllevel, ascores in wllevels_anomscores.items():
        wllevel_i += len(ascores)
        wllevels_indsend[wllevel] = wllevel_i
        ascores_accum_wllevel = get_accum(ascores)
        ascores_accum += ascores_accum_wllevel
        wllevels_ascoresaccum[wllevel] = ascores_accum_wllevel
        if get_pcounts:
            pcounts_accum += get_accum(wllevels_predcounts[wllevel])

    ## plot - behavior (total)
    levels_colors = {'Level 0': 'black', 'Level 1': 'blue', 'Level 2': 'purple', 'Level 3': 'aqua'}
    for feat in columns_model:
        plt.cla()
        plt.plot(alldata_task[feat].values)
        plt.title(f'Behavior - Validation')
        plt.xlabel('time')
        plt.ylabel(feat)
        for wllevel, ind in wllevels_indsend.items():
            plt.axvline(x=ind, color='r', label=wllevel)
        out_path = os.path.join(out_dir, f'time--validation--{feat}.png')
        plt.savefig(out_path)
        ## by level
        for level, data in wllevels_alldata.items():
            d_feat = data[feat].values
            plt.cla()
            plt.plot(d_feat, color=levels_colors[level])
            # plt.xlabel('time')
            plt.ylabel(feat)
            # plt.title(f'Behavior - {feat} - {level}')
            out_path = os.path.join(out_dir, f'time--validation--{feat}--{level}.png')
            plt.savefig(out_path)

    ## plot - aScores
    plt.cla()
    plt.plot(ascores_accum)
    plt.title("Perceived WL")
    plt.xlabel('Time')
    plt.ylabel('Accumulated Anomaly Scores')
    prev_ind = 0
    for wllevel, ind in wllevels_indsend.items():
        plt.axvline(x=ind, color='r', label=wllevel)
        fdata_level = wllevels_anomscores[wllevel]
        loc_x = np.percentile([ind, prev_ind], 25)
        loc_y = np.percentile(fdata_level, 25)
        fdata_mean = round(np.mean(fdata_level), 2)
        plt.text(loc_x, loc_y, f"avg:{fdata_mean}")
        prev_ind = ind
    out_path = os.path.join(out_dir, f'time--validation--aScores.png')
    plt.savefig(out_path)

    ## plot - aScpres overlapped
    plt.cla()
    plt_ymax = 50
    max_ascoreaccum = 0
    for wllevel, ascoresaccum in wllevels_ascoresaccum.items():
        wllevel_ascores_mean = round(np.mean(wllevels_anomscores[wllevel]), 2)
        plt.plot(ascoresaccum, label=f"{wllevel} (avg={wllevel_ascores_mean})", color=levels_colors[wllevel])
        max_ascoreaccum = max(max_ascoreaccum, max(ascoresaccum))
        # loc_x = len(ascoresaccum)
        # loc_y = ascoresaccum[-1]
        # plt.text(loc_x, loc_y, f"avg={wllevel_ascores_mean}")
    max_ascoreaccum = max(max_ascoreaccum, plt_ymax)
    plt.ylim(0, max_ascoreaccum)
    plt.title("Perceived WL by Task Level")
    plt.xlabel('Time')
    plt.ylabel('Accumulated Anomaly Scores')
    plt.legend()
    out_path = os.path.join(out_dir, f'levels--validation--aScores.png')
    plt.savefig(out_path)

    ## plot - pCounts
    if get_pcounts:
        plt.cla()
        plt.plot(pcounts_accum)
        plt.title("Perceived WL")
        plt.xlabel('Time')
        plt.ylabel('Accumulated Prediction Counts')
        prev_ind = 0
        for wllevel, ind in wllevels_indsend.items():
            plt.axvline(x=ind, color='r', label=wllevel)
            fdata_level = wllevels_predcounts[wllevel]
            loc_x = np.percentile([ind, prev_ind], 25)
            loc_y = np.percentile(fdata_level, 25)
            fdata_mean = round(np.mean(fdata_level), 2)
            plt.text(loc_x, loc_y, f"avg:{fdata_mean}")
            prev_ind = ind
        out_path = os.path.join(out_dir, f'time--validation--pCounts.png')
        plt.savefig(out_path)


def plot_bars(mydict, title, xlabel, ylabel, path_out):
    plt.cla()
    plt.bar(range(len(mydict)), list(mydict.values()), align='center')
    plt.xticks(range(len(mydict)), list(mydict.keys()), rotation=90)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path_out,bbox_inches="tight")


def get_accum(values):
    accum_vals, accum_sum = [], 0
    for v in values:
        accum_vals.append(v + accum_sum)
        accum_sum += v
    return accum_vals


def plot_subjects_timeserieses(dir_data, dir_out):
    subjects_found = [f for f in os.listdir(dir_data) if os.path.isdir(os.path.join(dir_data, f))]
    path_realtimewl = os.path.join(dir_data, 'realtime_wl.csv')
    realtime_wl = pd.read_csv(path_realtimewl)
    for subj in subjects_found:
        print(f"{subj}...")
        dir_data_subj = os.path.join(dir_data, subj)
        dir_out_subj = os.path.join(dir_out, subj)
        realtime_wl_subj = realtime_wl[realtime_wl['Subject'] == subj]
        os.makedirs(dir_out_subj, exist_ok=True)
        plot_subject_timeserieses(dir_data_subj, dir_out_subj, realtime_wl_subj)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def plot_subject_timeserieses(dir_data_subj, dir_out_subj, realtime_wl_subj, hz_baseline=100, hz_convertto=3):
    dir_files = [f for f in os.listdir(dir_data_subj) if '.xls' in f]
    files_train = [f for f in dir_files if 'training' in f]
    files_static = [f for f in dir_files if 'static' in f]
    files_realtime = [f for f in dir_files if 'realtime' in f]
    files_train.sort(key=natural_keys)
    files_static.sort(key=natural_keys)
    files_realtime.sort(key=natural_keys)
    # training
    plot_timeseries_combined(files=files_train, title='Training', dir_data_subj=dir_data_subj, dir_out_subj=dir_out_subj, hz_baseline=hz_baseline, hz_convertto=hz_convertto)
    # static
    plot_timeseries_combined(files=files_static, title='Static', dir_data_subj=dir_data_subj, dir_out_subj=dir_out_subj, hz_baseline=hz_baseline, hz_convertto=hz_convertto)
    # realtime
    plot_timeseries(files=files_realtime, title='RealTime', dir_data_subj=dir_data_subj, dir_out_subj=dir_out_subj, realtime_wl_subj=realtime_wl_subj, hz_baseline=hz_baseline, hz_convertto=hz_convertto)


def plot_timeseries(files, title, dir_data_subj, dir_out_subj, realtime_wl_subj, hz_baseline=100, hz_convertto=6.67):
    agg = int(hz_baseline / hz_convertto)
    # ind_prev = 0
    for fn in files:
        # import
        dpath = os.path.join(dir_data_subj, fn)
        fn = fn.replace('.xls', '')
        data = pd.read_excel(dpath)
        data.columns = ['time', 'steering angle', 'break']
        # agg
        data = data.groupby(data.index // agg).mean()
        # subtract mean
        mean_ = np.mean(data['steering angle'])
        data['steering angle'] = [v-mean_ for v in data['steering angle']]
        # get times wl imposed
        run_number = int(fn.split('realtime')[1])
        wl_imposed1_col = f"Run {run_number}-1"
        wl_imposed2_col = f"Run {run_number}-2"
        wl_runtime_col = f"Run {run_number}-Total"
        wl_imposed1_seconds = realtime_wl_subj[wl_imposed1_col].values[0]
        wl_imposed2_seconds = realtime_wl_subj[wl_imposed2_col].values[0]
        wl_total_seconds = realtime_wl_subj[wl_runtime_col].values[0]
        wl_imposed1 = int((wl_imposed1_seconds/wl_total_seconds) * data.shape[0])
        wl_imposed2 = int((wl_imposed2_seconds / wl_total_seconds) * data.shape[0])
        # plot
        steering_angles = data['steering angle'].values
        plt.cla()
        plt.plot(steering_angles)
        plt.axvline(x=wl_imposed1, color='orange', linestyle='-.', linewidth=1, label='WL Imposed 1')
        plt.axvline(x=wl_imposed2, color='green', linestyle='-.', linewidth=1, label='WL Imposed 2')
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.title(title)
        plt.tight_layout()
        path_out = os.path.join(dir_out_subj, f"{title}--{fn}.png")
        plt.savefig(path_out)


def plot_timeseries_combined(files, dir_data_subj, dir_out_subj, title='Training', hz_baseline=100, hz_convertto=6.67):
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
        data['steering angle'] = [v-mean_ for v in data['steering angle']]
        datas.append(data)
        vline_ind = len(data) + count
        f = f.replace('.xls', '').replace('--', '').replace('crash', '')
        filenames_vlineindices[f] = int(vline_ind)
        count += len(data)
    data_total = pd.concat(datas, axis=0)
    steering_angles = data_total['steering angle'].values
    plt.cla()
    plt.plot(steering_angles)
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
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    path_out = os.path.join(dir_out_subj, f"{title}.png")
    plt.savefig(path_out)


if __name__ == "__main__":
    plot_subjects_timeserieses(dir_data="/Users/samheiserman/Desktop/repos/workload_assessor/data",
                               dir_out="/Users/samheiserman/Desktop/repos/workload_assessor/eda")
