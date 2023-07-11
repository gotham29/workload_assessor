import copy
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, kstest, mannwhitneyu

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DIR_OUT = os.path.join(_SOURCE_DIR, 'results')

sys.path.append(_SOURCE_DIR)
from source.analyze.plot import plot_hists

ALGS_DIRS_IN = {
    'HTM': "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/HTM/preproc--autocorr_thresh=5; hz=5",
    'SE': "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/SteeringEntropy/preproc--autocorr_thresh=5; hz=5",
    'TLX': "/Users/samheiserman/Desktop/repos/workload_assessor/results/tlx"
}


def run_stat_tests(algs_data):
    # T-Test
    t_pval1 = ttest_ind(algs_data['HTM'], algs_data['SE'])[1]
    # KS-Test
    ks_pval1 = kstest(algs_data['HTM'], algs_data['SE'])[1]
    # MW-Test
    mw_pval1 = mannwhitneyu(algs_data['HTM'], algs_data['SE'])[1]
    print("  T")
    print(f"    HTM vs SE --> {t_pval1}")
    print("  KS")
    print(f"    HTM vs SE --> {ks_pval1}")
    print("  MW")
    print(f"    HTM vs SE --> {mw_pval1}")
    if 'TLX' in algs_data:
        # T-Test
        t_pval2 = ttest_ind(algs_data['HTM'], algs_data['TLX'])[1]
        t_pval3 = ttest_ind(algs_data['SE'], algs_data['TLX'])[1]
        # KS-Test
        ks_pval2 = kstest(algs_data['HTM'], algs_data['TLX'])[1]
        ks_pval3 = kstest(algs_data['SE'], algs_data['TLX'])[1]
        # MW-Test
        mw_pval2 = mannwhitneyu(algs_data['HTM'], algs_data['TLX'])[1]
        mw_pval3 = mannwhitneyu(algs_data['SE'], algs_data['TLX'])[1]
        print("  T")
        print(f"    HTM vs TLX --> {t_pval2}")
        print(f"    SE vs TLX --> {t_pval3}")
        print("  KS")
        print(f"    HTM vs TLX --> {ks_pval2}")
        print(f"    SE vs TLX --> {ks_pval3}")
        print("  MW")
        print(f"    HTM vs TLX --> {mw_pval2}")
        print(f"    SE vs TLX --> {mw_pval3}")


"""
Test for statistically significant performance differences between Algs
"""

"""
1) % Increase from Baseline to all other WL levels
    --> 1 value per subject
"""
# title = "% Increase from Baseline to ALL other WL levels"
# algs_paths = {
#     'HTM': os.path.join(ALGS_DIRS_IN['HTM'], "subjects_wldiffs.csv"),
#     'SE': os.path.join(ALGS_DIRS_IN['SE'], "subjects_wldiffs.csv"),
#     'TLX': os.path.join(ALGS_DIRS_IN['TLX'], "scores_subjects.csv")
# }
# algs_data = {}
# for alg, alg_path in algs_paths.items():
#     data = pd.read_csv(alg_path)
#     algs_data[alg] = data['%Change from Baseline'].values
# print("TEST 1")
# run_stat_tests(algs_data)


"""
2) % Increase from Baseline to EACH other WL level
    --> 3 values per subject
"""
# title = "Total sensitivity to increased task demands"
# algs_paths = {
#     'HTM': os.path.join(ALGS_DIRS_IN['HTM'], "subjects_levels_wldiffs.csv"),
#     'SE': os.path.join(ALGS_DIRS_IN['SE'], "subjects_levels_wldiffs.csv"),
#     'TLX': os.path.join(ALGS_DIRS_IN['TLX'], "subjects_levels_wldiffs.csv")
# }
# algs_data = {}
# for alg, alg_path in algs_paths.items():
#     data = pd.read_csv(alg_path)
#     values = []
#     cols = [c for c in data if c!='Unnamed: 0']
#     for c in cols:
#         values += list(data[c].values)
#     algs_data[alg] = values
#     print(alg)
# print(f"{title}")
# plot_hists(algs_data, DIR_OUT, title)
# run_stat_tests(algs_data)


"""
3) Degree of overlap with TLX
    --> 1 value per subject
"""
# title = "Correlation with NASA TLX"
# algs_paths = {
#     'HTM': os.path.join(ALGS_DIRS_IN['HTM'], "subjects_overlaps.csv"),
#     'SE': os.path.join(ALGS_DIRS_IN['SE'], "subjects_overlaps.csv"),
# }
# algs_data = {}
# for alg, alg_path in algs_paths.items():
#     data = pd.read_csv(alg_path)
#     algs_data[alg] = data['overlaps'].values
# plot_hists(algs_data, DIR_OUT, title)
# print(f"\n\n{title}")
# run_stat_tests(algs_data)


""" Check for performance variation across compensation algorithms """
dir_results = "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc"

def plot_violin(df_, xlabel, ylabel, title, feats_colors, path_out, y_lim=None):
    vplot_anom = sns.violinplot(data=df_,
                                x=xlabel,
                                y=ylabel,
                                pallette=feats_colors)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_lim:
        plt.ylim(y_lim[0], y_lim[1])
    plt.savefig(path_out, bbox_inches="tight")
    plt.close()

def plot_violins(df, feat_gpby, feat_score, y_lim, filter, dir_results):
    df_ = df[ [feat_gpby,feat_score] ]
    title = f"{feat_score} by {feat_gpby}\nfilter={filter}"
    path_out = os.path.join(dir_results, f"{title.replace(feat_score,'')}.png")
    plot_violin(df_, feat_gpby, feat_score, title, algs_colors, path_out, y_lim)

def filter_df(df, filter_dict):
    if filter_dict is None:
        return df
    df_ = copy.deepcopy(df)
    for feat, vals in filter_dict.items():
        df_ = df_[df_[feat].isin(vals)]
    return df_

# Gather scores from all runs
def get_df_summary(dir_results):
    rows = []
    algs_wl = [f for f in os.listdir(dir_results) if '.DS' not in f and '.png' not in f and 'Score' not in f]
    print(f"algs_wl\n  --> {algs_wl}")
    for alg_wl in algs_wl:
        dir_alg_wl = os.path.join(dir_results, alg_wl)
        hzs = [f for f in os.listdir(dir_alg_wl) if '.DS' not in f]
        for hz in hzs:
            dir_hzs = os.path.join(dir_alg_wl, hz)
            scenarios = [f for f in os.listdir(dir_hzs) if '.DS' not in f]
            for scen in scenarios:
                dir_scenario = os.path.join(dir_hzs, scen)
                algs_comp = [f for f in os.listdir(dir_scenario) if '.DS' not in f]
                for alg_comp in algs_comp:
                    dir_alg_comp = os.path.join(dir_scenario, alg_comp)
                    modnames = [f for f in os.listdir(dir_alg_comp) if '.DS' not in f]
                    for modname in modnames:
                        path_scores = os.path.join(dir_alg_comp, modname, 'scores.csv')
                        df_scores = pd.read_csv(path_scores)
                        sensitivity = df_scores['Total sensitivity to increased task demands'].values[0]
                        percent_baseline_lowest = df_scores['Rate of subjects with baseline lowest'].values[0]
                        row = {'WL-Alg': alg_wl, 'HZ': hz, 'MC-Scenario': scen, 'Comp-Alg': alg_comp, 'Model-Name': modname.replace('modname=',''), 'Score-Sensitivity': sensitivity, 'Score-%BaselineLowest': percent_baseline_lowest}
                        rows.append(row)
    df_summary = pd.DataFrame(rows)
    return df_summary


# MAKE PLOTS
df_summary = get_df_summary(dir_results)
algs_colors = {'HTM': 'blue', 'PSD': 'green', 'SteeringEntropy': 'red', 'Naive': 'grwy'}

pltdatas = [
## 1) by wl-algs
    {'filter':None,
     'y_lim': [0,100],
     'dir_out':'Score-Sensitivity',
     'feat_gpby':'WL-Alg',
     'feat_score':'Score-Sensitivity'},
    {'filter':None,
     'y_lim': [0, 100],
     'dir_out':'Score-%BaselineLowest',
     'feat_gpby':'WL-Alg',
     'feat_score':'Score-%BaselineLowest'},

## 2) by modnames
    {'filter':None,
     'y_lim': None,
     'dir_out': 'Score-Sensitivity',
     'feat_gpby': 'Model-Name',
     'feat_score': 'Score-Sensitivity'},
    {'filter':None,
     'y_lim': [0, 100],
     'dir_out': 'Score-%BaselineLowest',
     'feat_gpby': 'Model-Name',
     'feat_score': 'Score-%BaselineLowest'},

## 3) by scenarios
    {'filter':None,
     'y_lim': None,
     'dir_out': 'Score-Sensitivity',
     'feat_gpby': 'MC-Scenario',
     'feat_score': 'Score-Sensitivity'},
    {'filter':None,
     'y_lim': [0, 100],
     'dir_out': 'Score-%BaselineLowest',
     'feat_gpby': 'MC-Scenario',
     'feat_score': 'Score-%BaselineLowest'},

## 4) by comp-algs
    {'filter': {
        'MC-Scenario': ['offset'],
        'Model-Name': ['roll_stick']
    },
     'y_lim': None,
     'dir_out': 'Score-Sensitivity',
     'feat_gpby': 'Comp-Alg',
     'feat_score': 'Score-Sensitivity'},
    {'filter': {
        'MC-Scenario': ['offset'],
        'Model-Name': ['roll_stick']
    },
     'y_lim': [0, 100],
     'dir_out': 'Score-%BaselineLowest',
     'feat_gpby': 'Comp-Alg',
     'feat_score': 'Score-%BaselineLowest'},

    {'filter': None,
     'y_lim': None,
     'dir_out': 'Score-Sensitivity',
     'feat_gpby': 'Comp-Alg',
     'feat_score': 'Score-Sensitivity'},
    {'filter': None,
     'y_lim': [0, 100],
     'dir_out': 'Score-%BaselineLowest',
     'feat_gpby': 'Comp-Alg',
     'feat_score': 'Score-%BaselineLowest'},

## 5) by wl-alg (offset)
    {'filter': {
        'MC-Scenario': ['offset'],
        },
    'y_lim': None,
    'dir_out': 'Score-Sensitivity',
    'feat_gpby': 'WL-Alg',
    'feat_score': 'Score-Sensitivity'},

    {'filter': {
        'MC-Scenario': ['offset'],
        },
    'y_lim': [0, 100],
    'dir_out': 'Score-%BaselineLowest',
    'feat_gpby': 'WL-Alg',
    'feat_score': 'Score-%BaselineLowest'},

## 6) by wl-alg (offset, roll)
    {'filter': {
        'MC-Scenario': ['offset'],
        'Model-Name': ['roll_stick']
    },
    'y_lim': None,
    'dir_out': 'Score-Sensitivity',
    'feat_gpby': 'WL-Alg',
    'feat_score': 'Score-Sensitivity'},

    {'filter': {
        'MC-Scenario': ['offset'],
        'Model-Name': ['roll_stick']
        },
    'y_lim': [0, 100],
    'dir_out': 'Score-%BaselineLowest',
    'feat_gpby': 'WL-Alg',
    'feat_score': 'Score-%BaselineLowest'},
]

for pltdata in pltdatas:
    df_ = filter_df(df_summary, pltdata['filter'])
    dir_out = os.path.join(dir_results, pltdata['dir_out'])
    os.makedirs(dir_out, exist_ok=True)
    plot_violins(df_, pltdata['feat_gpby'], pltdata['feat_score'], pltdata['y_lim'], pltdata['filter'], dir_out)



#
# if __name__ == "__main__":
#     main()
