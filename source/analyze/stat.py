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

from source.analyze.anomaly import get_subjects_wldiffs
from source.pipeline.run_pipeline import get_scores

# MAKE PLOTS
make_tlx_overlaps = True
make_boxplots_groups = False
make_boxplots_algs = False
make_plots_violin = False
convert_tlxs = False
eval_tlxs = False


# ALGS_DIRS_IN = {
#     'HTM': "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/HTM/preproc--autocorr_thresh=5; hz=5",
#     'SE': "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/SteeringEntropy/preproc--autocorr_thresh=5; hz=5",
#     'TLX': "/Users/samheiserman/Desktop/repos/workload_assessor/results/tlx"
# }


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


""" DEFINE FUNCTIONS """

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
    df_ = df[[feat_gpby, feat_score]]
    title = f"{feat_score} by {feat_gpby}\nfilter={filter}"
    path_out = os.path.join(dir_results, f"{title.replace(feat_score, '')}.png")
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
                        row = {'WL-Alg': alg_wl, 'HZ': hz, 'MC-Scenario': scen, 'Comp-Alg': alg_comp,
                               'Model-Name': modname.replace('modname=', ''), 'Score-Sensitivity': sensitivity,
                               'Score-%BaselineLowest': percent_baseline_lowest}
                        rows.append(row)
    df_summary = pd.DataFrame(rows)
    return df_summary


def get_df_wllevels_totalascores(dir_results):
    comp_algs = [f for f in os.listdir(dir_results) if '.DS' not in f and f != 'total']
    rows = []
    for comp_alg in comp_algs:
        dir_comp = os.path.join(dir_results, comp_alg)
        model_names = [f for f in os.listdir(dir_comp) if 'modname=' in f]
        for model_name in model_names:
            dir_mod = os.path.join(dir_comp, model_name)
            subjs_wllevels_totalascores = pd.read_csv(os.path.join(dir_mod, 'subjects_wllevels_totalascores.csv'))
            for _, row_ in subjs_wllevels_totalascores.iterrows():
                wllevel = row_['Unnamed: 0']
                total_ascores = [v for v in row_.values if isinstance(v, float)]
                row = {'comp-alg': comp_alg, 'model-name': model_name, 'wl-level': wllevel,
                       'total-ascores': total_ascores}
                rows.append(row)
    df_wllevels_totalascores = pd.DataFrame(rows)
    return df_wllevels_totalascores


def get_datasets(df_wllevels_totalascores, feat_groupby='comp-alg'):
    gpby = df_wllevels_totalascores.groupby(feat_groupby)
    groups_datasets = {}
    for gr, df_gr in gpby:
        wllevels_totalascores = {wl: [] for wl in df_gr['wl-level'].unique()}
        gpby_ = df_gr.groupby('wl-level')
        for wllevel, df_wllevel in gpby_:
            wllevels_totalascores[wllevel] += df_wllevel['total-ascores'].values[0]
        dataset = pd.DataFrame(wllevels_totalascores)
        groups_datasets[gr] = dataset
    return groups_datasets


def make_boxplot_groups(groups_datasets, colours, groups_legendnames, ylabel, title, path_out):
    groups = list(groups_datasets.keys())
    datasets = list(groups_datasets.values())
    # Set figsize
    plt.rc('figure', figsize=[10, 3])
    # Set x-positions for boxes
    x_pos_range = np.arange(len(datasets)) / (len(datasets) - 1)
    x_pos = (x_pos_range * 0.5) + 0.75
    # Plot
    for i, data in enumerate(datasets):
        bp = plt.boxplot(
            np.array(data), sym='', whis=[0, 100], widths=0.6 / len(datasets),
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
        legend_elements.append(Patch(facecolor=colours[k], edgecolor=edge_color, label=groups_legendnames[groups[j]]))
    plt.legend(handles=legend_elements, fontsize=8)
    # Save
    plt.savefig(path_out)
    plt.close()


def convert_txlfiles_toformat(dir_tlxs, dir_out, runs_modes):
    paths_tlx = [os.path.join(dir_tlxs, f) for f in os.listdir(dir_tlxs) if 'TLX' in f]
    rows_offset = []
    rows_straightin = []

    for path in paths_tlx:
        # get subj & load df
        subj = path.split('TLX TS')[-1].replace('.xls', '')
        df = pd.read_excel(path).dropna(how='all', axis=0)

        # split df by mc-scenario
        cols = list(df.iloc[0].values)
        df.columns = cols
        df = df.drop([2], axis=0)
        rownumber_offset = list(df[cols[0]].values).index('OFFSET')
        df_offset = df[rownumber_offset:]
        df_offset = df_offset.dropna(how='any', axis=0)
        df_straightin = df[:rownumber_offset]
        df_straightin = df_straightin.dropna(how='any', axis=0)

        # loop over dfs and populate df_out
        for _, row_ in df_offset.iterrows():
            row = {'Subject': f"subj{subj}", 'Run #': row_['RUN #'], 'Run Mode': runs_modes[str(row_['RUN #'])],
                   'Mental Demand': row_['MENTAL'], 'Physical Demand': row_['PHYSICAL'],
                   'Temporal Demand': row_['TEMPORAL'],
                   'Performance': row_['PERF'], 'Effort': row_['EFFORT'], 'Frustration': row_['FRUST'],
                   'Raw TLX': row_['TLX']}
            rows_offset.append(row)

        for _, row_ in df_straightin.iterrows():
            row = {'Subject': f"subj{subj}", 'Run #': row_['RUN #'], 'Run Mode': runs_modes[str(row_['RUN #'])],
                   'Mental Demand': row_['MENTAL'], 'Physical Demand': row_['PHYSICAL'],
                   'Temporal Demand': row_['TEMPORAL'],
                   'Performance': row_['PERF'], 'Effort': row_['EFFORT'], 'Frustration': row_['FRUST'],
                   'Raw TLX': row_['TLX']}
            rows_straightin.append(row)

    df_offset = pd.DataFrame(rows_offset)
    df_straight = pd.DataFrame(rows_straightin)

    path_out_offset = os.path.join(dir_out, 'tlx--offset.csv')
    path_out_straight = os.path.join(dir_out, 'tlx--straight-in.csv')

    df_offset.to_csv(path_out_offset, index=False)
    df_straight.to_csv(path_out_straight, index=False)


def eval_tlx(df_tlx, path_out):
    # get subjects_wllevels_totalascores
    subjects_wllevels_totalascores = {}
    for subj, df_subj in df_tlx.groupby('Subject'):
        subjects_wllevels_totalascores[subj] = {}
        for wllevel, df_subj_level in df_subj.groupby('Run Mode'):
            subjects_wllevels_totalascores[subj][wllevel] = df_subj_level['Raw TLX'].sum()
    # get normalized diffs for all subjs
    subjects_wldiffs, subjects_levels_wldiffs = get_subjects_wldiffs(subjects_wllevels_totalascores)
    # get scores
    percent_change_from_baseline, subjects_increased_from_baseline, subjects_baseline_lowest = get_scores(
        subjects_wldiffs, subjects_wllevels_totalascores)
    percent_subjects_baseline_lowest = round(100 * len(subjects_baseline_lowest) / len(subjects_wllevels_totalascores),
                                             2)
    scores = {'Total sensitivity to increased task demands': percent_change_from_baseline,
              'Rate of subjects with baseline lowest': percent_subjects_baseline_lowest}
    scores = pd.DataFrame(scores, index=[0])
    scores.to_csv(path_out)


def get_groups_datasets(dir_in, filter_, feat_groupby, compalgs_order):
    df_wllevels_totalascores = get_df_wllevels_totalascores(dir_in)
    if filter_:
        df_wllevels_totalascores = filter_df(df_wllevels_totalascores, filter_)
    groups_datasets = get_datasets(df_wllevels_totalascores, feat_groupby)
    groups_datasets = {k.replace('_', ''): v for k, v in groups_datasets.items()}
    index_map = {v: i for i, v in enumerate(compalgs_order)}
    groups_datasets = dict(sorted(groups_datasets.items(), key=lambda pair: index_map[pair[0]]))
    return groups_datasets


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    return bp


def get_compalgs_wlalgs_wllevelsdata(runs_comps, df_tlx, algs_colors, hz, features, mc_scenario, filter_, feat_groupby,
                                     compalgs_order, delays_order, dir_out):
    comps_col = [runs_comps[run] for run in df_tlx['Run #']]
    df_tlx.insert(loc=0, column='Comp Alg', value=comps_col)
    compalgs_wlalgs_wllevelsdata = {comp.replace('_', ''): {} for comp in df_tlx['Comp Alg'].unique()}
    compalgs_wlalgs_wllevelsdata['total'] = {}
    # get by comp-alg (TLX)
    for comp, df_comp in df_tlx.groupby('Comp Alg'):
        df_dict = {}
        for mode, df_mode in df_comp.groupby('Run Mode'):
            df_dict[mode] = df_mode['Raw TLX'].values
        compalgs_wlalgs_wllevelsdata[comp.replace('_', '')]['TLX'] = pd.DataFrame(df_dict)[delays_order]
    # get total (TLX)
    df_dict = {}
    for mode, df_mode in df_tlx.groupby('Run Mode'):
        df_dict[mode] = df_mode['Raw TLX'].values
    compalgs_wlalgs_wllevelsdata['total']['TLX'] = pd.DataFrame(df_dict)[delays_order]
    for wl_alg in algs_colors:
        if wl_alg == 'TLX':
            continue
        dir_out_alg = os.path.join(dir_out, wl_alg)
        dir_results = os.path.join(dir_out_alg, f"hz={hz}; features={features}/{mc_scenario}")
        # get by comp-alg
        groups_datasets = get_groups_datasets(dir_results, filter_, feat_groupby, compalgs_order)
        for comp, data in groups_datasets.items():
            compalgs_wlalgs_wllevelsdata[comp][wl_alg] = data
        # get total
        total_datasets = pd.concat(list(groups_datasets.values()), axis=0)
        compalgs_wlalgs_wllevelsdata['total'][wl_alg] = total_datasets
    return compalgs_wlalgs_wllevelsdata


""" RUN """

if make_plots_violin:
    dir_results = "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc"
    algs_colors = {'HTM': 'blue', 'PSD': 'green', 'SteeringEntropy': 'red', 'Naive': 'grwy'}
    df_summary = get_df_summary(dir_results)
    pltdatas = [
        ## 1) by wl-algs
        {'filter': None,
         'y_lim': [0, 100],
         'dir_out': 'Score-Sensitivity',
         'feat_gpby': 'WL-Alg',
         'feat_score': 'Score-Sensitivity'},
        {'filter': None,
         'y_lim': [0, 100],
         'dir_out': 'Score-%BaselineLowest',
         'feat_gpby': 'WL-Alg',
         'feat_score': 'Score-%BaselineLowest'},

        ## 2) by modnames
        {'filter': None,
         'y_lim': None,
         'dir_out': 'Score-Sensitivity',
         'feat_gpby': 'Model-Name',
         'feat_score': 'Score-Sensitivity'},
        {'filter': None,
         'y_lim': [0, 100],
         'dir_out': 'Score-%BaselineLowest',
         'feat_gpby': 'Model-Name',
         'feat_score': 'Score-%BaselineLowest'},

        ## 3) by scenarios
        {'filter': None,
         'y_lim': None,
         'dir_out': 'Score-Sensitivity',
         'feat_gpby': 'MC-Scenario',
         'feat_score': 'Score-Sensitivity'},
        {'filter': None,
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
    wl_algs = ['HTM', 'PSD', 'SteeringEntropy', 'TLX']
    mc_scenarios = ['offset', 'straight-in']
    filter_ = {'model-name': ['modname=roll_stick']}  # 'modname=pitch_stick', 'modname=roll_stick'
    features = 'pitch_stick.roll_stick.rudder_pedals.throttle'
    megamodel = False
    if len(filter_['model-name']) > 1:
        megamodel = True
        features = '.'.join([f.replace('modname=', '') for f in sorted(filter_['model-name'])]) + ' - MEGA'
    hz = '16.67'
    feat_groupby = 'comp-alg'
    ylabel = "Perceived WL"
    dir_in = "/Users/samheiserman/Desktop/repos/workload_assessor/data"
    dir_out = "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc"
    groups_legendnames = {'nc': 'No Compensation',
                          'mc': 'McFarland Predictor',
                          'mfr': 'McFarland Predictor, Spike Reduced',
                          'ap': 'Adaptive Predictor',
                          'ss': 'State Space Predictor'}
    runs_comps = {
        1: '_ss',
        34: '_ss',
        2: '_ss',
        39: '_ss',
        19: '_ss',
        21: '_ss',
        4: '_ss',
        38: '_ss',
        10: '_ap',
        31: '_ap',
        16: '_ap',
        29: '_ap',
        5: '_ap',
        28: '_ap',
        18: '_ap',
        35: '_ap',
        20: '_nc',
        33: '_nc',
        8: '_nc',
        37: '_nc',
        7: '_nc',
        23: '_nc',
        13: '_nc',
        27: '_nc',
        3: '_mc',
        30: '_mc',
        6: '_mc',
        22: '_mc',
        17: '_mc',
        40: '_mc',
        12: '_mc',
        36: '_mc',
        24: '_mfr',
        14: '_mfr',
        9: '_mfr',
        32: '_mfr',
        11: '_mfr',
        25: '_mfr',
        15: '_mfr',
        26: '_mfr'
    }
    colours = ['white', 'yellow', 'cyan', 'tab:purple', 'magenta']
    compalgs_order = list(groups_legendnames.keys())
    dir_out_boxes = os.path.join(dir_out, 'wl_by_delay&comp')
    os.makedirs(dir_out_boxes, exist_ok=True)
    for mc_scenario in mc_scenarios:
        dir_out_mc = os.path.join(dir_out_boxes, mc_scenario)
        os.makedirs(dir_out_mc, exist_ok=True)
        for wl_alg in wl_algs:
            if megamodel and wl_alg != 'HTM':
                continue
            dir_out_wl = os.path.join(dir_out_mc, wl_alg)
            os.makedirs(dir_out_wl, exist_ok=True)
            if wl_alg == 'TLX':
                path_out = os.path.join(dir_out_wl, 'NA')
                path_tlx = os.path.join(dir_in, f"tlx--{mc_scenario}.csv")
                df_tlx = pd.read_csv(path_tlx)
                comps_col = [runs_comps[run] for run in df_tlx['Run #']]
                df_tlx.insert(loc=0, column='Comp Alg', value=comps_col)
                groups_datasets = {}
                for comp, df_comp in df_tlx.groupby('Comp Alg'):
                    df_dict = {}
                    for mode, df_mode in df_comp.groupby('Run Mode'):
                        df_dict[mode] = df_mode['Raw TLX'].values
                    groups_datasets[comp.replace('_', '')] = pd.DataFrame(df_dict)
                    index_map = {v: i for i, v in enumerate(compalgs_order)}
                    groups_datasets = dict(sorted(groups_datasets.items(), key=lambda pair: index_map[pair[0]]))
            else:
                path_out = os.path.join(dir_out_wl, f"{'--'.join(filter_['model-name'])}")
                dir_results = os.path.join(dir_out, wl_alg, f"hz={hz}; features={features}/{mc_scenario}")
                filter__ = copy.deepcopy(filter_)
                if megamodel:
                    filter__ = None
                groups_datasets = get_groups_datasets(dir_results, filter__, feat_groupby, compalgs_order)
            title = f"WL by delay condition & {feat_groupby}"
            make_boxplot_groups(groups_datasets, colours, groups_legendnames, ylabel, title, path_out)
            # print means/medians by comp-alg & delay
            for comp, data_comp in groups_datasets.items():
                print(f"\n{comp}")
                for delay in data_comp:
                    print(f"  delay={delay}")
                    print(f"    mean = {round(np.mean(data_comp[delay]), 4)}")
                    print(f"    median = {round(np.median(data_comp[delay]), 4)}")

if convert_tlxs:
    runs_modes = {
        '1': 'baseline',
        '2': 'delay=0.048',
        '3': 'baseline',
        '4': 'delay=0.192',
        '5': 'delay=0.096',
        '6': 'delay=0.048',
        '7': 'delay=0.096',
        '8': 'delay=0.048',
        '9': 'delay=0.048',
        '10': 'baseline',
        '11': 'delay=0.096',
        '12': 'delay=0.192',
        '13': 'delay=0.192',
        '14': 'baseline',
        '15': 'delay=0.192',
        '16': 'delay=0.048',
        '17': 'delay=0.096',
        '18': 'delay=0.192',
        '19': 'delay=0.096',
        '20': 'baseline',
        '21': 'delay=0.096',
        '22': 'delay=0.048',
        '23': 'delay=0.096',
        '24': 'baseline',
        '25': 'delay=0.096',
        '26': 'delay=0.192',
        '27': 'delay=0.192',
        '28': 'delay=0.096',
        '29': 'delay=0.048',
        '30': 'baseline',
        '31': 'baseline',
        '32': 'delay=0.048',
        '33': 'baseline',
        '34': 'baseline',
        '35': 'delay=0.192',
        '36': 'delay=0.192',
        '37': 'delay=0.048',
        '38': 'delay=0.192',
        '39': 'delay=0.048',
        '40': 'delay=0.096'
    }
    dir_tlxs = "/Users/samheiserman/Desktop/PhD/paper2 - lewin /TLX"
    dir_out = "/Users/samheiserman/Desktop/repos/workload_assessor/data"
    convert_txlfiles_toformat(dir_tlxs, dir_out, runs_modes)

if eval_tlxs:
    dir_out = "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc"
    dir_in = "/Users/samheiserman/Desktop/repos/workload_assessor/data"
    runs = [20, 33, 8, 37, 7, 23, 13, 27]
    filter_ = {'Run #': runs}  # None
    mc_scenarios = ['offset', 'straight-in']
    dir_out_tlx = os.path.join(dir_out, 'tlx-scores')
    os.makedirs(dir_out_tlx, exist_ok=True)
    for scenario in mc_scenarios:
        path_tlx = os.path.join(dir_in, f"tlx--{scenario}.csv")
        path_out = os.path.join(dir_out_tlx, f'{scenario}--{filter_}.csv')
        df_tlx = pd.read_csv(path_tlx)
        if filter_:
            df_tlx = filter_df(df_tlx, filter_)
        eval_tlx(df_tlx, path_out)

if make_boxplots_algs:
    mc_scenarios = ['offset', 'straight-in']
    filter_ = {'model-name': ['modname=roll_stick', 'modname=pitch_stick']}  # roll_stick
    algs_colors = {'HTM': 'blue', 'PSD': 'green', 'TLX': 'red'}
    features = 'pitch_stick.roll_stick.rudder_pedals.throttle'
    dir_out = "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc"
    dir_in = "/Users/samheiserman/Desktop/repos/workload_assessor/data"
    delays_order = ['baseline', 'delay=0.048', 'delay=0.096', 'delay=0.192']
    hz = '16.67'
    feat_groupby = 'comp-alg'
    groups_legendnames = {'nc': 'No Compensation',
                          'mc': 'McFarland Predictor',
                          'mfr': 'McFarland Predictor, Spike Reduced',
                          'ap': 'Adaptive Predictor',
                          'ss': 'State Space Predictor'}
    compalgs_order = list(groups_legendnames.keys())
    runs_comps = {
        1: '_ss',
        34: '_ss',
        2: '_ss',
        39: '_ss',
        19: '_ss',
        21: '_ss',
        4: '_ss',
        38: '_ss',
        10: '_ap',
        31: '_ap',
        16: '_ap',
        29: '_ap',
        5: '_ap',
        28: '_ap',
        18: '_ap',
        35: '_ap',
        20: '_nc',
        33: '_nc',
        8: '_nc',
        37: '_nc',
        7: '_nc',
        23: '_nc',
        13: '_nc',
        27: '_nc',
        3: '_mc',
        30: '_mc',
        6: '_mc',
        22: '_mc',
        17: '_mc',
        40: '_mc',
        12: '_mc',
        36: '_mc',
        24: '_mfr',
        14: '_mfr',
        9: '_mfr',
        32: '_mfr',
        11: '_mfr',
        25: '_mfr',
        15: '_mfr',
        26: '_mfr'
    }
    dir_out_boxes = os.path.join(dir_out, 'wl_by_comp')
    os.makedirs(dir_out_boxes, exist_ok=True)
    for mc_scenario in mc_scenarios:
        dir_out_mc = os.path.join(dir_out_boxes, mc_scenario)
        dir_out_model = os.path.join(dir_out_mc, '--'.join(filter_['model-name']))
        os.makedirs(dir_out_mc, exist_ok=True)
        os.makedirs(dir_out_model, exist_ok=True)
        path_tlx = os.path.join(dir_in, f"tlx--{mc_scenario}.csv")
        df_tlx = pd.read_csv(path_tlx)
        compalgs_wlalgs_wllevelsdata = get_compalgs_wlalgs_wllevelsdata(runs_comps, df_tlx, algs_colors, hz, features,
                                                                        mc_scenario, filter_, feat_groupby,
                                                                        compalgs_order, delays_order, dir_out)
        for comp, wlalgs_wllevelsdata in compalgs_wlalgs_wllevelsdata.items():
            fig = plt.figure()
            axs_tup = fig.subplots(len(wlalgs_wllevelsdata), sharex=True)
            fig.set_size_inches(15, 10)
            fig.suptitle(f"Compensation Alg = {comp}", fontsize=25)
            axs_tup[2].set_xlabel('Delay Conditions', fontsize=15)
            axs_tup[1].set_ylabel('Perceived WL', fontsize=15)
            bps = []
            for i, (wlalg, wllevelsdata) in enumerate(wlalgs_wllevelsdata.items()):
                ax = axs_tup[i]
                ds = [list(wllevelsdata[c].values) for c in wllevelsdata]
                bp = ax.boxplot(ds, positions=np.arange(len(ds)) + 1, showfliers=True, labels=wllevelsdata.columns)
                bp = set_box_color(bp, color=algs_colors[wlalg])
                bps.append(bp)
            fig.legend([bp_["boxes"][0] for bp_ in bps], list(wlalgs_wllevelsdata.keys()), loc='upper right',
                       fontsize=15)
            plt.tight_layout()
            path_out = os.path.join(dir_out_model, f"{comp}.png")
            plt.savefig(path_out)

if make_tlx_overlaps:
    filter_ = {'model-name': ['modname=pitch_stick', 'modname=roll_stick']}
    mc_scenarios = ['offset', 'straight-in']
    wl_algs = ['HTM', 'PSD', 'TLX']  # 'SteeringEntropy'
    hz = '16.67'
    feat_groupby = 'comp-alg'
    features = 'pitch_stick.roll_stick.rudder_pedals.throttle'
    delays_order = ['baseline', 'delay=0.048', 'delay=0.096', 'delay=0.192']
    algs_colors = {'HTM': 'blue', 'PSD': 'green', 'TLX': 'red'}  # 'SteeringEntropy': 'purple'
    dir_out = "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc"
    dir_in = "/Users/samheiserman/Desktop/repos/workload_assessor/data"
    dir_out_tlx = os.path.join(dir_out, 'tlx-overlaps')
    os.makedirs(dir_out_tlx, exist_ok=True)
    groups_legendnames = {'nc': 'No Compensation',
                          'mc': 'McFarland Predictor',
                          'mfr': 'McFarland Predictor, Spike Reduced',
                          'ap': 'Adaptive Predictor',
                          'ss': 'State Space Predictor'}
    runs_comps = {
        1: '_ss',
        34: '_ss',
        2: '_ss',
        39: '_ss',
        19: '_ss',
        21: '_ss',
        4: '_ss',
        38: '_ss',
        10: '_ap',
        31: '_ap',
        16: '_ap',
        29: '_ap',
        5: '_ap',
        28: '_ap',
        18: '_ap',
        35: '_ap',
        20: '_nc',
        33: '_nc',
        8: '_nc',
        37: '_nc',
        7: '_nc',
        23: '_nc',
        13: '_nc',
        27: '_nc',
        3: '_mc',
        30: '_mc',
        6: '_mc',
        22: '_mc',
        17: '_mc',
        40: '_mc',
        12: '_mc',
        36: '_mc',
        24: '_mfr',
        14: '_mfr',
        9: '_mfr',
        32: '_mfr',
        11: '_mfr',
        25: '_mfr',
        15: '_mfr',
        26: '_mfr'
    }
    compalgs_order = list(groups_legendnames.keys())
    filter__ = copy.deepcopy(filter_)
    for mc_scenario in mc_scenarios:
        path_tlx = os.path.join(dir_in, f"tlx--{mc_scenario}.csv")
        df_tlx = pd.read_csv(path_tlx)
        dir_out_mc = os.path.join(dir_out_tlx, mc_scenario)
        os.makedirs(dir_out_mc, exist_ok=True)
        path_out = os.path.join(dir_out_mc, f"filter={str(filter_)}.csv")
        if len(filter_['model-name']) > 1:
            filter__ = None
            algs_colors = {alg: color for alg, color in algs_colors.items() if alg == 'HTM'}
            features = '.'.join([f.replace('modname=', '') for f in sorted(filter_['model-name'])]) + ' - MEGA'
        compalgs_wlalgs_wllevelsdata = get_compalgs_wlalgs_wllevelsdata(runs_comps, df_tlx, algs_colors, hz, features,
                                                                        mc_scenario, filter__, feat_groupby,
                                                                        compalgs_order, delays_order, dir_out)
        wlalgs_comps_wlvariations = {alg: {} for alg in wl_algs}
        for comp, wlalgs_wllevelsdata in compalgs_wlalgs_wllevelsdata.items():
            for wl_alg, wllevelsdata in wlalgs_wllevelsdata.items():
                var_cols = wllevelsdata.var(axis='columns').mean()
                wlalgs_comps_wlvariations[wl_alg][comp] = var_cols
        df_overlaps = pd.DataFrame(wlalgs_comps_wlvariations).round(3)
        df_overlaps.to_csv(path_out)


#
# if __name__ == "__main__":
#     main()
