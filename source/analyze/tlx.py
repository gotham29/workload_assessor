import collections
import operator
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
PATH_TLX = os.path.join(_SOURCE_DIR, 'data', 'tlx.csv')
DIR_OUT = os.path.join(_SOURCE_DIR, 'results', 'tlx')

sys.path.append(_SOURCE_DIR)

from source.analyze.plot import plot_outputs_bars
from source.analyze.anomaly import get_subjects_wldiffs

MODES_CONVERT = {
    'B': 'baseline',
    'D': 'distraction',
    'D.C.': 'rain',
    'O.P.': 'fog',
}


def make_boxplots(data_dict, levels_colors, ylabel, title, path_out, suptitle=None, ylim=None):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    fig, ax = plt.subplots()
    bplot = ax.boxplot(data_dict.values(), patch_artist=True)
    ax.set_xticklabels(data_dict.keys())
    colors = list(levels_colors.values())  #['grey', 'orange', 'blue', 'green']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    if ylim:
        plt.ylim(ylim)
    plt.ylabel(ylabel)
    if suptitle:
        plt.suptitle(suptitle)
    plt.title(title)
    plt.savefig(path_out)
    plt.close()


def get_overlaps(subjects_wllevels_totalascores, subjects_tlxorders, order_already_set=False):
    subjects_orders_1 = {}
    for subj, wllevels_totalascores in subjects_wllevels_totalascores.items():
        subjects_orders_1[subj.lower().strip()] = list(wllevels_totalascores.keys())
    if order_already_set:
        subjects_orders_2 = subjects_tlxorders
    else:
        subjects_orders_2 = {}
        for subj, wllevels_ascores in subjects_tlxorders.items():
            wllevels_ascoresums = {wllevel: np.sum(ascores) for wllevel, ascores in wllevels_ascores.items()}
            wllevels_ascoresums = dict(sorted(wllevels_ascoresums.items(), key=operator.itemgetter(1)))
            subjects_orders_2[subj.lower().strip()] = list(wllevels_ascoresums.keys())
    # get overlaps
    subjects_overlaps = {}
    for subj, order_1 in subjects_orders_1.items():
        order_2 = subjects_orders_2[subj]
        overlaps = 0
        for _, wllevel in enumerate(order_1):
            if wllevel == order_2[_]:
                overlaps += 1
        overlap = overlaps / (len(order_1) - 1)
        subjects_overlaps[subj] = min(round(overlap, 3), 1.0)
    subjects_overlaps = dict(collections.OrderedDict(sorted(subjects_overlaps.items())))
    df_overlaps = pd.DataFrame(subjects_overlaps, ['overlaps']).T
    return df_overlaps


def get_tlx_overlaps(subjects_wllevels_totalascores, path_tlx):
    df_tlx = pd.read_csv(path_tlx)
    ## get tlx orders
    subscales = ['Mental Demand', 'Physical Demand', 'Temporal Demand', 'Performance', 'Effort', 'Frustration',
                 'Raw TLX']
    subscales_meanoverlaps = {}
    subscales_dfsoverlaps = {}
    gpby_subj = df_tlx.groupby('Subject')
    for subscale in subscales:
        subjects_tlxorders = {}
        for subj, df_subj in gpby_subj:
            modes_scores = {}
            subj = subj.lower().strip()
            gpby_mode = df_subj.groupby('Run Mode')
            for mode, df_mode in gpby_mode:
                modes_scores[MODES_CONVERT[mode]] = np.sum(df_mode[subscale].values)
            modes_scores = dict(sorted(modes_scores.items(), key=operator.itemgetter(1)))
            subjects_tlxorders[subj] = list(modes_scores.keys())
        ## get df_overlaps
        df_overlaps = get_overlaps(subjects_wllevels_totalascores, subjects_tlxorders, order_already_set=True)
        df_overlaps.columns = [subscale]
        mean_percent_overlap = 100 * round(np.mean(df_overlaps[subscale]), 3)
        subscales_meanoverlaps[subscale] = mean_percent_overlap
        subscales_dfsoverlaps[subscale] = df_overlaps
    df_overlaps_total = pd.concat(list(subscales_dfsoverlaps.values()), axis=1)
    return subscales_meanoverlaps, df_overlaps_total


def main():
    """

    Takes the csv file and extracts subject, run mode, raw tlx
    Newframe is the data frame to do anything specific
    Check matplotlib/pandas documentation for more information

    """
    SUBJS_DROP = ['Zaychik', 'Heiserman']
    columns = ["Subject", "Run #", "Run Mode", "Mental Demand", "Physical Demand", "Temporal Demand", "Performance",
               "Effort", "Frustration", "Raw TLX"]
    dataframe = pd.read_csv(PATH_TLX, usecols=columns)
    newframe = dataframe.loc[1:, ["Subject", "Run Mode", "Raw TLX"]]
    nftlx = newframe.loc[1:, ["Raw TLX"]]
    nftlx.apply(pd.to_numeric)

    """ box plots for run mode - all subjects combined """
    path_out = os.path.join(DIR_OUT, f"*modes_compared.png")
    modes_scores = {}
    gpby_mode = newframe.groupby('Run Mode')
    for mode, df_mode in gpby_mode:
        modes_scores[MODES_CONVERT[mode]] = df_mode['Raw TLX'].values
    modes_scores_sum = {mode: np.sum(scores) for mode, scores in modes_scores.items()}
    plot_outputs_bars(modes_scores_sum, title="TLX Scores by Run Mode -- All Subjects", xlabel="WL Levels", ylabel='Raw TLX',
              path_out=path_out, xtickrotation=0, colors=['grey', 'orange', 'blue', 'green'], print_barheights=True)
    # make_boxplots(modes_scores, ylabel='Raw TLX', title="All Subjects", path_out=path_out,
    #               suptitle='TLX Scores by Run Mode', ylim=(0, 1))

    """ box plots for run mode score by subject """
    subjects_baselinelower = {}
    subjects_baselinelowest = {}
    # subjects_normdiffs = {}
    subjects_wllevelsascores = {}
    gpby_subj = newframe.groupby('Subject')
    for subj, df_subj in gpby_subj:
        if subj in SUBJS_DROP:
            continue
        path_out = os.path.join(DIR_OUT, f"{subj}.png")
        modes_scores = {}
        gpby_mode = df_subj.groupby('Run Mode')
        for mode, df_mode in gpby_mode:
            modes_scores[MODES_CONVERT[mode]] = df_mode['Raw TLX'].values
        subjects_wllevelsascores[subj] = modes_scores
        modes_scores_sum = {mode: np.sum(scores) for mode, scores in modes_scores.items()}
        plot_bars(modes_scores_sum, title=f"TLX Scores by Run Mode -- {subj}", xlabel="WL Levels",
                  ylabel='Raw TLX',
                  path_out=path_out, xtickrotation=0, colors=['grey', 'orange', 'blue', 'green'],
                  print_barheights=False)
        # make_boxplots(modes_scores, ylabel='Raw TLX', title='TLX Scores by Run Mode', path_out=path_out, suptitle=None,
        #               ylim=(0, 1))

        wl_t0 = modes_scores['baseline']
        wl_t0_mean = np.mean(wl_t0)
        baseline_lowest = True
        baseline_lower = True
        w1_means = []
        for mode, scores in modes_scores.items():
            if mode == 'baseline':
                continue
            wl_t1_mean = np.mean(scores)
            w1_means.append(wl_t1_mean)
            if wl_t1_mean < wl_t0_mean:
                baseline_lowest = False
        wl_t1_mean = np.mean(w1_means)
        if wl_t0_mean > wl_t1_mean:
            baseline_lower = False
        subjects_baselinelower[subj] = baseline_lower
        subjects_baselinelowest[subj] = baseline_lowest
        # subjects_normdiffs[subj] = #100 * round((mean_nonbaseline - mean_baseline) / mean_baseline, 2)
    # save scores subjects
    subjects_wldiffs, subjects_levels_wldiffs = get_subjects_wldiffs(subjects_wllevelsascores)
    df_subjects_baselinelower = pd.DataFrame(subjects_baselinelower, index=['baseline_lower']).T
    df_subjects_baselinelowest = pd.DataFrame(subjects_baselinelowest, index=['baseline_lowest']).T
    df_subjects_normdiffs = pd.DataFrame(subjects_wldiffs, index=['Difference from Baseline']).T
    path_out = os.path.join(DIR_OUT, 'scores_subjects.csv')
    df_sum = pd.concat([df_subjects_baselinelower, df_subjects_baselinelowest, df_subjects_normdiffs], axis=1)
    df_sum.to_csv(path_out)
    # save scores sums
    percent_baselinelower = np.sum(df_sum['baseline_lower']) / len(df_sum)
    percent_baselinelowest = np.sum(df_sum['baseline_lowest']) / len(df_sum)
    normalized_diff = np.sum(df_sum['Difference from Baseline'])
    scores = {'percent_baselinelower': 100 * round(percent_baselinelower, 3),
              'percent_baselinelowest': 100 * round(percent_baselinelowest, 3),
              'Total sensitivity to increased task demand': normalized_diff}
    path_out = os.path.join(DIR_OUT, 'scores_sum.csv')
    pd.DataFrame(scores, index=[0]).to_csv(path_out, index=False)
    # save subjects_wllevelsascores
    path_out_subjects_levels_wldiffs = os.path.join(DIR_OUT, 'subjects_levels_wldiffs.csv')
    subjects_wldiffs, subjects_levels_wldiffs = get_subjects_wldiffs(subjects_wllevelsascores)
    df_subjects_levels_wldiffs = pd.DataFrame(subjects_levels_wldiffs)
    df_subjects_levels_wldiffs.to_csv(path_out_subjects_levels_wldiffs, index=True)


if __name__ == "__main__":
    main()
