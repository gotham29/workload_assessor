import copy
import os
import sys

import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import ttest_ind, kstest, mannwhitneyu

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DIR_OUT = os.path.join(_SOURCE_DIR, 'results')

sys.path.append(_SOURCE_DIR)


# ALGS_DIRS_IN = {
#     'HTM': "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/HTM/preproc--autocorr_thresh=5; hz=5",
#     'SE': "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/SteeringEntropy/preproc--autocorr_thresh=5; hz=5",
#     'TLX': "/Users/samheiserman/Desktop/repos/workload_assessor/results/tlx"
# }


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


""" Violinplots - check performance variation """
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

def get_df_wllevels_totalascores(dir_results):
    comp_algs = [f for f in os.listdir(dir_results) if '.DS' not in f and f!='total']
    rows = []
    for comp_alg in comp_algs:
        dir_comp = os.path.join(dir_results, comp_alg)
        model_names = [f for f in os.listdir(dir_comp) if 'modname=' in f]
        for model_name in model_names:
            dir_mod = os.path.join(dir_comp, model_name)
            subjs_wllevels_totalascores = pd.read_csv( os.path.join(dir_mod, 'subjects_wllevels_totalascores.csv') )
            for _, row_ in subjs_wllevels_totalascores.iterrows():
                wllevel = row_['Unnamed: 0']
                total_ascores = [v for v in row_.values if isinstance(v,float)]
                row = {'comp-alg':comp_alg, 'model-name':model_name, 'wl-level':wllevel, 'total-ascores':total_ascores}
                rows.append(row)
    df_wllevels_totalascores = pd.DataFrame(rows)
    return df_wllevels_totalascores

def get_datasets(df_wllevels_totalascores, filter_, feat_groupby='comp-alg'):
    df_filter = filter_df(df_wllevels_totalascores,filter_)
    gpby = df_filter.groupby(feat_groupby)
    groutps_datasets = {}
    for gr, df_gr in gpby:
        wllevels_totalascores = {wl:[] for wl in df_gr['wl-level'].unique()}
        gpby_ = df_gr.groupby('wl-level')
        for wllevel, df_wllevel in gpby_:
            wllevels_totalascores[wllevel] +=  df_wllevel['total-ascores'].values[0]
        dataset = pd.DataFrame(wllevels_totalascores)
        groutps_datasets[gr] = dataset
    return groutps_datasets

def make_boxplot_groups(datasets, colours, groups, groups_legendnames, ylabel, title, whis, path_out):

    # Set figsize
    plt.rc('figure', figsize=[10, 3])
    # Set x-positions for boxes
    x_pos_range = np.arange(len(datasets)) / (len(datasets) - 1)
    x_pos = (x_pos_range * 0.5) + 0.75
    # Plot
    for i, data in enumerate(datasets):
        bp = plt.boxplot(
            np.array(data), sym='', whis=[0,100], widths=0.6 / len(datasets),
            labels=list(datasets[0]), patch_artist=True,
            positions=[x_pos[i] + j * 1 for j in range(len(data.T))]
        )
        # Fill the boxes with colours (requires patch_artist=True)
        k = i % len(colours)
        edge_color = 'black'
        fill_color = colours[k]
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)
    # Titles
    plt.title(title)
    plt.ylabel(ylabel)
    # Axis ticks and labels
    plt.xticks(np.arange(len(list(datasets[0]))) + 1)
    plt.gca().xaxis.set_minor_locator(ticker.FixedLocator(
        np.array(range(len(list(datasets[0])) + 1)) + 0.5)
    )
    plt.gca().tick_params(axis='x', which='minor', length=4)
    plt.gca().tick_params(axis='x', which='major', length=0)
    # Change the limits of the x-axis
    plt.xlim([0.5, len(list(datasets[0])) + 0.5])
    # Legend
    legend_elements = []
    for i in range(len(datasets)):
        j = i % len(groups)
        k = i % len(colours)
        legend_elements.append(Patch(facecolor=colours[k], edgecolor=edge_color, label=groups_legendnames[groups[j]] ))
    plt.legend(handles=legend_elements, fontsize=8)
    # Save
    plt.savefig(path_out)
    plt.close()


# MAKE PLOTS
make_plots_violin = False
make_boxplots_groups = True

if make_plots_violin:
    dir_results = "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc"
    algs_colors = {'HTM': 'blue', 'PSD': 'green', 'SteeringEntropy': 'red', 'Naive': 'grwy'}
    df_summary = get_df_summary(dir_results)
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

if make_boxplots_groups:
    wl_alg = 'HTM'
    mc_scenario = 'offset'  #'offset', 'straight-in'
    hz = '16.67'
    feat_groupby = 'comp-alg'
    groups_legendnames = {'nc': 'No Compensation',
                          'mc': 'McFarland Predictor',
                          'mfr': 'McFarland Predictor, Spike Reduced',
                          'ap': 'Adaptive Predictor',
                          'ss': 'State Space Predictor'
                        }
    # colours = ['white', 'yellow', 'aqua', 'purple', 'pink']
    colours = ['white', 'yellow', 'cyan', 'tab:purple', 'magenta']
    compalgs_order = list(groups_legendnames.keys())
    dir_results = f"/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/{wl_alg}/hz={hz}/{mc_scenario}"
    dir_out = "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc"
    ylabel = "Perceived WL"
    title = f"{ylabel} by delay & {feat_groupby} ({mc_scenario})"
    filter_ = {
        'model-name':['modname=roll_stick', 'modname=pitch_stick']
        }
    df_wllevels_totalascores = get_df_wllevels_totalascores(dir_results)
    groups_datasets = get_datasets(df_wllevels_totalascores, filter_, feat_groupby)
    groups_datasets = {k.replace('_',''):v for k,v in groups_datasets.items()}
    index_map = {v: i for i, v in enumerate(compalgs_order)}
    groups_datasets = dict(sorted(groups_datasets.items(), key=lambda pair: index_map[pair[0]]))
    groups = list(groups_datasets.keys())
    path_out = os.path.join(dir_out, f"{title} -- {str(filter_)}")
    make_boxplot_groups(list( groups_datasets.values()), colours, groups, groups_legendnames, ylabel, title, whis, path_out)


#
# if __name__ == "__main__":
#     main()
