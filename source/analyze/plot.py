import os
import sys
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.utils.utils import make_dir


def plot_data(filenames_data: dict, file_type: str, dir_output: str):
    dir_out = os.path.join(dir_output, 'data_plots')
    make_dir(dir_out)
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