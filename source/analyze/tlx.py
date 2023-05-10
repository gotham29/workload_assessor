import collections
import operator
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
PATH_TLX = os.path.join(_SOURCE_DIR, 'data', 'tlx.csv')
DIR_OUT = os.path.join(_SOURCE_DIR, 'results', 'tlx')

MODES_CONVERT = {
    'B': 'baseline',
    'D': 'distraction',
    'D.C.': 'rain',
    'O.P.': 'fog',
}


def make_boxplots(data_dict, ylabel, title, path_out, suptitle=None, ylim=None):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    # plt.grid(True)
    # plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    bplot = ax.boxplot(data_dict.values(), patch_artist=True)
    ax.set_xticklabels(data_dict.keys())
    colors = ['grey', 'orange', 'blue', 'green']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    if ylim:
        plt.ylim(ylim)
    # plt.figure(figsize=(15, 3))
    plt.ylabel(ylabel)
    if suptitle:
        plt.suptitle(suptitle)
    plt.title(title)
    plt.savefig(path_out)
    plt.close()


def save_tlx_overlaps(subjects_wllevelsascores, dir_out):
    PATH_TLX = os.path.join(_SOURCE_DIR, 'data', 'tlx.csv')
    df_tlx = pd.read_csv(PATH_TLX)
    ## get tlx orders
    subjects_tlxorders = {}
    gpby_subj = df_tlx.groupby('Subject')
    for subj, df_subj in gpby_subj:
        subj = subj.lower().strip()
        modes_scores = {}
        gpby_mode = df_subj.groupby('Run Mode')
        for mode, df_mode in gpby_mode:
            modes_scores[MODES_CONVERT[mode]] = np.sum(df_mode['Raw TLX'].values)
        modes_scores = dict(sorted(modes_scores.items(), key=operator.itemgetter(1)))
        subjects_tlxorders[subj] = list(modes_scores.keys())
    ## get ml orders
    subjects_mlorders = {}
    for subj, wllevels_ascores in subjects_wllevelsascores.items():
        subj = subj.lower().strip()
        wllevels_ascoresums = {wllevel: np.sum(ascores) for wllevel, ascores in wllevels_ascores.items()}
        wllevels_ascoresums = dict(sorted(wllevels_ascoresums.items(), key=operator.itemgetter(1)))
        subjects_mlorders[subj] = list(wllevels_ascoresums.keys())
    ## get TLX-ML overlaps
    subjects_mltlx_overlaps = {}
    for subj, ml_order in subjects_mlorders.items():
        tlx_order = subjects_tlxorders[subj]
        overlaps = 0
        for _, wllevel in enumerate(ml_order):
            if wllevel == tlx_order[_]:
                overlaps += 1
        overlap = overlaps / (len(ml_order) - 1)
        subjects_mltlx_overlaps[subj] = min(round(overlap, 3), 1.0)
    # save results
    subjects_mltlx_overlaps = dict(collections.OrderedDict(sorted(subjects_mltlx_overlaps.items())))
    df_overlaps = pd.DataFrame(subjects_mltlx_overlaps, ['overlaps']).T
    path_out_overlps = os.path.join(dir_out, 'tlx_overlaps.csv')
    df_overlaps.to_csv(path_out_overlps, index=True)
    return 100 * round(np.mean(df_overlaps['overlaps']), 3)


def main():
    """

    Takes the csv file and extracts subject, run mode, raw tlx
    Newframe is the data frame to do anything specific
    Check matplotlib/pandas documentation for more information

    """
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
    make_boxplots(modes_scores, ylabel='Raw TLX', title="All Subjects", path_out=path_out,
                  suptitle='TLX Scores by Run Mode', ylim=(0, 1))

    """ box plots for run mode score by subject """
    subjects_baselinelower = {}
    subjects_baselinelowest = {}
    subjects_percentchanges = {}
    gpby_subj = newframe.groupby('Subject')
    for subj, df_subj in gpby_subj:
        path_out = os.path.join(DIR_OUT, f"{subj}.png")
        modes_scores = {}
        gpby_mode = df_subj.groupby('Run Mode')
        for mode, df_mode in gpby_mode:
            modes_scores[MODES_CONVERT[mode]] = df_mode['Raw TLX'].values
        make_boxplots(modes_scores, ylabel='Raw TLX', title='TLX Scores by Run Mode', path_out=path_out, suptitle=None,
                      ylim=(0, 1))
        scores_baseline = modes_scores['baseline']
        mean_baseline = np.mean(scores_baseline)
        baseline_lowest = True
        baseline_lower = True
        means_nonbaseline = []
        for mode, scores in modes_scores.items():
            if mode == 'baseline':
                continue
            mean_mode = np.mean(scores)
            means_nonbaseline.append(mean_mode)
            if mean_mode < mean_baseline:
                baseline_lowest = False
        mean_nonbaseline = np.mean(means_nonbaseline)
        if mean_baseline > mean_nonbaseline:
            baseline_lower = False
        subjects_baselinelower[subj] = baseline_lower
        subjects_baselinelowest[subj] = baseline_lowest
        subjects_percentchanges[subj] = 100*round((mean_nonbaseline - mean_baseline) / mean_baseline, 2)
    # save scores subjects
    df_subjects_baselinelower = pd.DataFrame(subjects_baselinelower, index=['baseline_lower']).T
    df_subjects_baselinelowest = pd.DataFrame(subjects_baselinelowest, index=['baseline_lowest']).T
    df_subjects_percentchanges = pd.DataFrame(subjects_percentchanges, index=['percent_change']).T
    path_out = os.path.join(DIR_OUT, 'scores_subjects.csv')
    df_sum = pd.concat([df_subjects_baselinelower, df_subjects_baselinelowest, df_subjects_percentchanges], axis=1)
    df_sum.to_csv(path_out)
    # save scores sum
    percent_baselinelower = np.sum(df_sum['baseline_lower']) / len(df_sum)
    percent_baselinelowest = np.sum(df_sum['baseline_lowest']) / len(df_sum)
    percent_change = np.sum(df_sum['percent_change']) / len(df_sum)
    scores = {'percent_baselinelower': 100*round(percent_baselinelower, 2),
              'percent_baselinelowest': 100*round(percent_baselinelowest, 2),
              'percent_change': 100*round(percent_change, 2)}
    path_out = os.path.join(DIR_OUT, 'scores_sum.csv')
    pd.DataFrame(scores, index=[0]).to_csv(path_out, index=False)




if __name__ == "__main__":
    main()
