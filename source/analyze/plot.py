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
    outtypes_paths = {'aScores': os.path.join(out_dir, 'MWLLevels_aScores.png'),
                      'pCounts': os.path.join(out_dir, 'MWLLevels_pCounts.png'),
                      'WLScores': os.path.join(out_dir, 'MWLLevels_WLScores.png')}
    # Plot -- Violin
    outtypes_data = {ot: [] for ot in outtypes_paths if ot != 'WLScores'}
    for testlevel, ascores in data_plot1.items():
        df_dict_1 = {'Task Difficulty Level': [testlevel for _ in range(len(ascores))],
                     'Anomaly Score': ascores}
        df1 = pd.DataFrame(df_dict_1)
        outtypes_data['aScores'].append(df1)
        if data_plot2 != {}:
            pcounts = data_plot2[testlevel]
            df_dict_2 = {'Task Difficulty Level': [testlevel for _ in range(len(pcounts))],
                         'Prediction Count': pcounts}
            df2 = pd.DataFrame(df_dict_2)
            outtypes_data['pCounts'].append(df2)
    df_1 = pd.concat(outtypes_data['aScores'], axis=0)
    if data_plot2 != {}:
        df_2 = pd.concat(outtypes_data['pCounts'], axis=0)
    plt.cla()
    vplot_anom = sns.violinplot(data=df_1,
                                x="Task Difficulty Level",
                                y='Anomaly Score')
    plt.title(title_1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1.0)
    plt.savefig(outtypes_paths['aScores'].replace('.png', '--violin.png'), bbox_inches="tight")
    plt.cla()
    if data_plot2 != {}:
        vplot_pred = sns.violinplot(data=df_2,
                                    x="Task Difficulty Level",
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
                   columns_model: dict,
                   out_dir: str):
    # Gather data to plot
    x = []
    ascores_all, predcounts_all = [], []
    ascores_accum, predcounts_accum = [], []
    features_alldata = {f: [] for f in columns_model}
    wllevels_indsend, wllevel_i = {}, 0
    for wllevel, ascores in wllevels_anomscores.items():
        pcounts = wllevels_predcounts[wllevel]
        # accum x for line plot
        x += [len(x) + _ for _ in range(len(ascores))]
        # find indices separating wl levels
        wllevel_i += len(ascores)
        wllevels_indsend[wllevel] = wllevel_i
        # accum ascores & predcounts
        ascores_all += ascores
        predcounts_all += pcounts
        wllevel_accum_ascores = get_accum(ascores)
        wllevel_accum_pcounts = get_accum(pcounts)
        ascores_accum += wllevel_accum_ascores
        predcounts_accum += wllevel_accum_pcounts
        # accum data for all columns_model
        for feat in columns_model:
            features_alldata[feat] += list(wllevels_alldata[wllevel][feat].values)
    # Make plots
    data_plot = {
        'AScores': {'accum': ascores_accum, 'all': ascores_all},
        'PCounts': {'accum': predcounts_accum, 'all': predcounts_all}
    }
    for feat, fdata in features_alldata.items():
        data_plot[feat] = {'accum': fdata}
    for feat, fdata in data_plot.items():
        plt.cla()
        plt.plot(fdata['accum'])
        plt.title(feat)
        prev_ind = 0
        for wllevel, ind in wllevels_indsend.items():
            plt.axvline(x=ind, color='r', label=wllevel)
            if feat in ["AScores", "PCounts"]:
                fdata_level = fdata['all'][prev_ind:ind]
                fdata_mean = round(np.mean(fdata_level), 2)
                loc_x, loc_y = np.mean([ind, prev_ind]), fdata_mean
                plt.text(loc_x, fdata_mean, f"{fdata_mean}")
            prev_ind = ind
        out_path = os.path.join(out_dir, f'lines--{feat}.png')
        plt.savefig(out_path)


def plot_bars(wllevels_tlx, title, xlabel, ylabel, out_dir):
    plt.cla()
    plt.bar(range(len(wllevels_tlx)), list(wllevels_tlx.values()), align='center')
    plt.xticks(range(len(wllevels_tlx)), list(wllevels_tlx.keys()))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    path_out = os.path.join(out_dir, 'bars--TLXs.png')
    plt.savefig(path_out)


def get_accum(values):
    accum_vals, accum_sum = [], 0
    for v in values:
        accum_vals.append(v + accum_sum)
        accum_sum += v
    return accum_vals