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
from scipy.stats import ttest_ind, ttest_rel, kstest, mannwhitneyu

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DIR_OUT = os.path.join(_SOURCE_DIR, 'results')
sys.path.append(_SOURCE_DIR)

from source.analyze.anomaly import get_subjects_wldiffs
from source.pipeline.run_pipeline import get_scores
from source.analyze.plot import plot_outputs_bars
from source.analyze.tlx import make_boxplots

# MAKE PLOTS

make_master_results = False
make_table_1_fold = False
make_table_2_fold = False
make_table_2_foldsavg = False
make_boxplots_table23 = False

print_mean_scores = True

convert_chrs = False
convert_tlxs = False
eval_tlxs = False
make_tlx_overlaps = False
make_table_3 = False
make_boxplots_groups = False
make_boxplots_algs = False
make_plots_violin = False

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


def convert_chrfiles_toformat(dir_chrs, dir_out, runs_modes):
    paths_chr = [os.path.join(dir_chrs, f) for f in os.listdir(dir_chrs) if 'CHR' in f]
    rows_offset = []
    rows_straightin = []

    for path in paths_chr:
        # get subj & load df
        subj = path.split('CHR TS')[-1].replace('.xls', '')
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

        for _, row_ in df_offset.iterrows():
            row = {'Subject': f"subj{subj}", 'Run #': row_['RUN #'], 'Run Mode': runs_modes[str(row_['RUN #'])],
                   'GSE': row_['GLIDESLOPE'], 'TDE': row_['TD POINT']}
            rows_offset.append(row)

        for _, row_ in df_straightin.iterrows():
            row = {'Subject': f"subj{subj}", 'Run #': row_['RUN #'], 'Run Mode': runs_modes[str(row_['RUN #'])],
                   'GSE': row_['GLIDESLOPE'], 'TDE': row_['TD POINT']}
            rows_straightin.append(row)

    df_offset = pd.DataFrame(rows_offset)
    df_straight = pd.DataFrame(rows_straightin)

    path_out_offset = os.path.join(dir_out, 'chr--offset.csv')
    path_out_straight = os.path.join(dir_out, 'chr--straight-in.csv')

    df_offset.to_csv(path_out_offset, index=False)
    df_straight.to_csv(path_out_straight, index=False)


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


def eval_tlx(df_tlx, path_out_csv, dir_out_plt, levels_colors):
    # get subjects_wllevels_totalascores, wllevels_totalascores
    subjects_wllevels_totalascores = {}
    for subj, df_subj in df_tlx.groupby('Subject'):
        subjects_wllevels_totalascores[subj] = {}
        for wllevel, df_subj_level in df_subj.groupby('Run Mode'):
            totalascore = df_subj_level['Raw TLX'].sum()
            subjects_wllevels_totalascores[subj][wllevel] = totalascore
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
    scores.to_csv(path_out_csv)
    # make barplots
    for subj, wllevels_totalascores in subjects_wllevels_totalascores.items():
        path_out_plt = os.path.join(dir_out_plt, f"{subj}.png")
        plot_outputs_bars(mydict=wllevels_totalascores,
                          levels_colors=levels_colors,
                          title="TLX WL Scores by Level",
                          xlabel='WL Levels',
                          ylabel="TLX WL",
                          path_out=path_out_plt,
                          xtickrotation=0)


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


def paired_ttest_2samp(sample1, sample2):
    """
    Conducts a paired sample t-test for two small samples.

    Parameters:
        sample1 (list): First sample data (should be a list of numerical values).
        sample2 (list): Second sample data (should be a list of numerical values).

    Returns:
        t_stat (float): T-statistic value.
        p_value (float): Two-tailed p-value.
    """
    if len(sample1) != len(sample2):
        raise ValueError("Both samples should have the same length.")

    # Conduct paired t-test
    t_stat, p_value = ttest_rel(sample1, sample2)

    return t_stat, p_value


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

if convert_chrs:
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
    dir_chrs = "/Users/samheiserman/Desktop/PhD/paper2 - guo&cardullo/CHR"
    dir_out = "/Users/samheiserman/Desktop/repos/workload_assessor/data"
    convert_chrfiles_toformat(dir_chrs, dir_out, runs_modes)

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
    runs = None  # [20, 33, 8, 37, 7, 23, 13, 27]
    filter_ = None  # {'Run #': runs}  # None
    mc_scenarios = ['offset', 'straight-in']
    levels_colors = {'baseline': 'grey', 'delay=0.048': 'yellow', 'delay=0.096': 'orange', 'delay=0.192': 'red'}
    dir_out_tlx = os.path.join(dir_out, 'tlx-scores')
    os.makedirs(dir_out_tlx, exist_ok=True)
    for scenario in mc_scenarios:
        dir_out_mc = os.path.join(dir_out_tlx, scenario)
        os.makedirs(dir_out_mc, exist_ok=True)
        path_tlx = os.path.join(dir_in, f"tlx--{scenario}.csv")
        df_tlx = pd.read_csv(path_tlx)
        path_out_csv = os.path.join(dir_out_mc, f'filter={filter_}.csv')
        dir_out_plt = os.path.join(dir_out_mc, f'filter={filter_}--wllevels--aScoreTotals--bar')
        os.makedirs(dir_out_plt, exist_ok=True)
        if filter_:
            df_tlx = filter_df(df_tlx, filter_)
        eval_tlx(df_tlx, path_out_csv, dir_out_plt, levels_colors)

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

if make_master_results:
    training = "training=40%"  # training=40%-1, training=40%-2, training=100%
    fold = '2'  # 1,2
    dir_results = f"/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/{training}/{fold}"
    os.makedirs(dir_results, exist_ok=True)
    dir_in = "/Users/samheiserman/Desktop/repos/workload_assessor/data"
    fns_delays = {
        'run10_ap.csv': 'baseline',
        'run14_mfr.csv': 'baseline',
        'run20_nc.csv': 'baseline',
        'run1_ss.csv': 'baseline',
        'run3_mc.csv': 'baseline',
        'run2_ss.csv': 'delay=0.048',
        'run6_mc.csv': 'delay=0.048',
        'run8_nc.csv': 'delay=0.048',
        'run9_mfr.csv': 'delay=0.048',
        'run16_ap.csv': 'delay=0.048',
        'run5_ap.csv': 'delay=0.096',
        'run7_nc.csv': 'delay=0.096',
        'run11_mfr.csv': 'delay=0.096',
        'run17_mc.csv': 'delay=0.096',
        'run19_ss.csv': 'delay=0.096',
        'run4_ss.csv': 'delay=0.192',
        'run12_mc.csv': 'delay=0.192',
        'run13_nc.csv': 'delay=0.192',
        'run15_mfr.csv': 'delay=0.192',
        'run18_ap.csv': 'delay=0.192',
        'run24_mfr.csv': 'baseline',
        'run30_mc.csv': 'baseline',
        'run31_ap.csv': 'baseline',
        'run33_nc.csv': 'baseline',
        'run34_ss.csv': 'baseline',
        'run22_mc.csv': 'delay=0.048',
        'run29_ap.csv': 'delay=0.048',
        'run32_mfr.csv': 'delay=0.048',
        'run37_nc.csv': 'delay=0.048',
        'run39_ss.csv': 'delay=0.048',
        'run21_ss.csv': 'delay=0.096',
        'run23_nc.csv': 'delay=0.096',
        'run25_mfr.csv': 'delay=0.096',
        'run28_ap.csv': 'delay=0.096',
        'run40_mc.csv': 'delay=0.096',
        'run26_mfr.csv': 'delay=0.192',
        'run27_nc.csv': 'delay=0.192',
        'run35_ap.csv': 'delay=0.192',
        'run36_mc.csv': 'delay=0.192',
        'run38_ss.csv': 'delay=0.192'
    }
    wlalgs_scenarios_featuresets = {
        'HTM-roll': {
            'offset': ['roll_stick'],
            'straight-in': ['roll_stick'],
        },
        'HTM-pitch': {
            'offset': ['pitch_stick'],
            'straight-in': ['pitch_stick'],
        },
        'HTM-rudder': {
            'offset': ['rudder_pedals'],
            'straight-in': ['rudder_pedals'],
        },
        'HTM-rollpitch': {
            'offset': ['pitch_stick', 'roll_stick'],
            'straight-in': ['pitch_stick', 'roll_stick'],
        },
        # 'HTM': {
        #     'offset': ['roll_stick'],
        #     'straight-in': ['pitch_stick', 'roll_stick'],
        # },
        'SteeringEntropy-roll': {
            'offset': ['roll_stick'],
            'straight-in': ['roll_stick'],
        },
        'SteeringEntropy-pitch': {
            'offset': ['pitch_stick'],
            'straight-in': ['pitch_stick'],
        },
        'SteeringEntropy-rudder': {
            'offset': ['rudder_pedals'],
            'straight-in': ['rudder_pedals'],
        },
        # 'SteeringEntropy': {
        #     'offset': ['rudder_pedals'],
        #     'straight-in': ['throttle'],
        # },
        'IPSD-roll': {
            'offset': ['roll_stick'],
            'straight-in': ['roll_stick'],
        },
        'IPSD-pitch': {
            'offset': ['pitch_stick'],
            'straight-in': ['pitch_stick'],
        },
        'IPSD-rudder': {
            'offset': ['rudder_pedals'],
            'straight-in': ['rudder_pedals'],
        },
        'FPSD-roll': {
            'offset': ['roll_stick'],
            'straight-in': ['roll_stick'],
        },
        'FPSD-pitch': {
            'offset': ['pitch_stick'],
            'straight-in': ['pitch_stick'],
        },
        'FPSD-rudder': {
            'offset': ['rudder_pedals'],
            'straight-in': ['rudder_pedals'],
        },
    }
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
    rows = []
    for wl_alg, scenarios_featuresets in wlalgs_scenarios_featuresets.items():
        wl_alg_ = wl_alg.split('-')[0]
        print(f"\n{wl_alg}")
        for scenario, featureset in scenarios_featuresets.items():
            features = '.'.join(featureset)
            print(f"  {scenario} --> {features}")
            modname = f"modname={features}" if len(featureset) == 1 else f"modname=megamodel_features={len(featureset)}"
            dir_ = os.path.join(dir_results,
                                wl_alg_,
                                f"hz=16.67; features={features}",
                                scenario,
                                'total',
                                modname)
            subjs_fns_wls = pd.read_csv(os.path.join(dir_, "subjects_filenames_totalascores.csv"))
            comp_algs = []
            delay_conditions = []
            for fn in subjs_fns_wls['Unnamed: 0']:
                comp_algs.append(fn.split('_')[1].replace('.csv', ''))
                delay_conditions.append(fns_delays[fn])
            subjs_fns_wls.insert(loc=0, column='delay conditions', value=delay_conditions)
            subjs_fns_wls.insert(loc=0, column='comp alg', value=comp_algs)
            subjs = [c for c in subjs_fns_wls if 'subj' in c]
            for subj in subjs:
                for _, row in subjs_fns_wls.iterrows():
                    row_ = {
                        'wl alg': f"{wl_alg}",  # - {features}
                        'flight scenario': scenario,
                        'delay comp alg': row['comp alg'],
                        'delay condition': row['delay conditions'],
                        'subject': subj,
                        'wl perceived': row[subj]
                    }
                    rows.append(row_)
    df_master = pd.DataFrame(rows)
    # Get df_master for TLX
    rows = []
    for scenario in scenarios_featuresets:
        path_tlx = path_tlx = os.path.join(dir_in, f"tlx--{scenario}.csv")
        df_tlx = pd.read_csv(path_tlx)
        tlx_cols = [c for c in df_tlx if c not in ['Subject', 'Run #', 'Run Mode']]
        for _, row in df_tlx.iterrows():
            for c in tlx_cols:
                row_ = {
                    'wl alg': f"TLX-{c.split(' ')[0]}",
                    'flight scenario': scenario,
                    'delay comp alg': runs_comps[row['Run #']].replace('_', ''),
                    'delay condition': row['Run Mode'],
                    'subject': row['Subject'],
                    'wl perceived': row[c]
                }
                rows.append(row_)
    df_master_tlx = pd.DataFrame(rows)
    # Get df_master for CHR
    rows = []
    for scenario in scenarios_featuresets:
        path_chr = os.path.join(dir_in, f"chr--{scenario}.csv")
        df_chr = pd.read_csv(path_chr)
        chr_cols = [c for c in df_chr if c not in ['Subject', 'Run #', 'Run Mode']]
        for _, row in df_chr.iterrows():
            for c in chr_cols:
                row_ = {
                    'wl alg': f"CHR-{c.split(' ')[0]}",
                    'flight scenario': scenario,
                    'delay comp alg': runs_comps[row['Run #']].replace('_', ''),
                    'delay condition': row['Run Mode'],
                    'subject': row['Subject'],
                    'wl perceived': row[c]
                }
                rows.append(row_)
    df_master_chr = pd.DataFrame(rows)
    # Get df_master_total
    df_master_total = pd.concat([df_master, df_master_tlx, df_master_chr], axis=0)
    path_out = os.path.join(dir_results, 'results-master.csv')
    df_master_total.to_csv(path_out)

if make_table_1_fold:
    """ How often does WL drop from comp. to no comp.? """
    training = "training=40%"  # training=40%-1, training=40%-2, training=100%
    fold = '2'  # 1,2
    dir_results = f"/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/{training}"
    os.makedirs(dir_results, exist_ok=True)
    dir_results = os.path.join(dir_results, fold)
    os.makedirs(dir_results, exist_ok=True)
    path_in = os.path.join(dir_results, 'results-master.csv')
    path_out = os.path.join(dir_results, 'table1-wldrop-from-nc.csv')
    df_master_total = pd.read_csv(path_in)
    compalgs = ['mc', 'mfr', 'ap', 'ss']
    scenarios = ['straight-in', 'offset']
    delays = ['baseline', 'delay=0.048', 'delay=0.096', 'delay=0.192']
    wl_algs = ['HTM-roll', 'HTM-pitch', 'HTM-rudder', 'HTM-rollpitch',
               'SteeringEntropy-roll', 'SteeringEntropy-pitch', 'SteeringEntropy-rudder',
               'CHR-GSE', 'CHR-TDE',
               'TLX-Raw', 'TLX-Mental', 'TLX-Physical', 'TLX-Temporal', 'TLX-Performance', 'TLX-Effort',
               'TLX-Frustration',
               'IPSD-roll', 'IPSD-pitch', 'IPSD-rudder', 'FPSD-roll', 'FPSD-pitch', 'FPSD-rudder']
    rows = []
    for wl_alg in wl_algs:
        print(f"\n{wl_alg}")
        vals_wlalg = [c for c in df_master_total['wl alg'].unique() if wl_alg in c]
        df = df_master_total[df_master_total['wl alg'].isin(vals_wlalg)]
        for compalg in compalgs:
            for scenario in scenarios:
                for delay in delays:
                    print(f"      {compalg}--{scenario}--{delay}")
                    filter_dict = {'delay comp alg': [compalg, 'nc'], 'flight scenario': [scenario],
                                   'delay condition': [delay]}
                    df_ = filter_df(df, filter_dict)
                    # report if WL(comp) < WL(no comp)
                    if len(df_['delay comp alg'].unique()) == 1:
                        print("          **EMPTY")
                        row = {'wl alg': wl_alg, 'delay comp alg': compalg,
                               'flight approach': scenario, 'delay (ms)': delay, 'WL drop from no comp': 'N/A'}
                    else:
                        wl_nc = round(df_[df_['delay comp alg'] == 'nc']['wl perceived'].sum(), 2)
                        wl_comp = round(df_[df_['delay comp alg'] == compalg]['wl perceived'].sum(), 2)
                        print(f"        comp/nc = {wl_comp}/{wl_nc}")
                        drop = 'no'
                        if wl_nc > wl_comp:
                            drop = 'YES'
                            print("          GOOD")
                        row = {'wl alg': wl_alg, 'delay comp alg': compalg,
                               'flight approach': scenario, 'delay (ms)': delay, 'WL drop from no comp': drop}
                    rows.append(row)
    df_table = pd.DataFrame(rows)
    df_table.to_csv(path_out)

if make_table_2_fold:
    """ How clear are WL drops from comp. to no comp.? """
    training = "training=40%"
    fold = '2'  # 1,2
    dir_results = f"/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/{training}"
    os.makedirs(dir_results, exist_ok=True)
    dir_results = os.path.join(dir_results, fold)
    os.makedirs(dir_results, exist_ok=True)
    path_in = os.path.join(dir_results, 'results-master.csv')
    path_out = os.path.join(dir_results, 'table2-wldrop-from-nc.csv')
    df_master_total = pd.read_csv(path_in)
    compalgs = ['mc', 'mfr', 'ap', 'ss']
    scenarios = ['straight-in', 'offset']
    delays = ['baseline', 'delay=0.048', 'delay=0.096', 'delay=0.192']
    wl_algs = ['HTM-roll', 'HTM-pitch', 'HTM-rudder', 'HTM-rollpitch',
               'SteeringEntropy-roll', 'SteeringEntropy-pitch', 'SteeringEntropy-rudder',
               'CHR-GSE', 'CHR-TDE',
               'TLX-Raw', 'TLX-Mental', 'TLX-Physical', 'TLX-Temporal', 'TLX-Performance', 'TLX-Effort',
               'TLX-Frustration',
               'IPSD-roll', 'IPSD-pitch', 'IPSD-rudder', 'FPSD-roll', 'FPSD-pitch', 'FPSD-rudder']
    rows = []
    for wl_alg in wl_algs:
        print(f"\n{wl_alg}")
        vals_wlalg = [c for c in df_master_total['wl alg'].unique() if wl_alg in c]
        df = df_master_total[df_master_total['wl alg'].isin(vals_wlalg)]
        for compalg in compalgs:
            for scenario in scenarios:
                for delay in delays:
                    print(f"      {compalg}--{scenario}--{delay}")
                    filter_dict = {'delay comp alg': [compalg, 'nc'], 'flight scenario': [scenario],
                                   'delay condition': [delay]}
                    df_ = filter_df(df, filter_dict)
                    # get %change in WL from comp to no-comp & coefficient of variation of comp
                    if len(df_['delay comp alg'].unique()) == 1:
                        print("          **EMPTY")
                        row = {'wl alg': wl_alg, 'delay comp alg': compalg, 'flight approach': scenario,
                               'delay (ms)': delay, '%WL drop from no comp': 'N/A', 'coeff of variance': 'N/A',
                               'paired ttest p-value': 'NA'}
                    else:
                        wl_nc = df_[df_['delay comp alg'] == 'nc']['wl perceived']
                        wl_comp = df_[df_['delay comp alg'] == compalg]['wl perceived']
                        pct_drop = round((wl_nc.mean() - wl_comp.mean()) / wl_nc.mean() * 100, 2)
                        coeff_var = round(100 * wl_comp.std() / wl_comp.mean(), 2)
                        t_stat, p_value = paired_ttest_2samp(wl_nc, wl_comp)
                        row = {'wl alg': wl_alg, 'delay comp alg': compalg, 'flight approach': scenario,
                               'delay (ms)': delay, '%WL drop from no comp': pct_drop, 'coeff of variance': coeff_var,
                               'paired ttest p-value': round(p_value, 4)}
                    rows.append(row)
    df_table = pd.DataFrame(rows)
    df_table.to_csv(path_out)

if make_table_2_foldsavg:
    """ How clear are WL drops from comp. to no comp.? """
    table2_1 = pd.read_csv(
        "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/training=40%/1/table2-wldrop-from-nc.csv")
    table2_2 = pd.read_csv(
        "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/training=40%/2/table2-wldrop-from-nc.csv")
    path_out = "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/training=40%/table2-wldrop-from-nc.csv"
    cols_score = ['%WL drop from no comp', 'coeff of variance', 'paired ttest p-value']
    gpby = table2_1.groupby('wl alg')
    rows = []
    for wl, df_wl_1 in gpby:
        print(f"\n{wl}")
        df_wl_1 = df_wl_1[cols_score].reset_index().drop(columns=['index'])
        df_wl_2 = table2_2[table2_2['wl alg'] == wl][cols_score].reset_index().drop(columns=['index'])
        for _, row_1 in df_wl_1.iterrows():
            row_2 = df_wl_2.iloc[_]
            nans_1 = [v for v in row_1 if np.isnan(v)]
            nans_2 = [v for v in row_2 if np.isnan(v)]
            isnan_1 = True if len(nans_1) > 0 else False
            isnan_2 = True if len(nans_2) > 0 else False
            if not isnan_1 and not isnan_2:
                row_3 = (row_1 + row_2) / 2
            else:
                if isnan_1:
                    row_3 = row_2
                else:  # isnan_2
                    row_3 = row_1
            row_3 = {k: round(v, 3) for k, v in dict(row_3).items()}
            row_3['wl alg'] = wl
            rows.append(row_3)
    df = pd.DataFrame(rows)
    df.to_csv(path_out)

if make_table_3:
    """ Are the same high perfoming subjects found by HTM """
    training = "training=40%"
    fold = '1'  # 1,2
    dir_results = f"/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/{training}/{fold}"
    path_in = os.path.join(dir_results, 'results-master.csv')
    path_out = os.path.join(dir_results, 'table3-high-performing-subjects.csv')
    df_master_total = pd.read_csv(path_in)
    compalgs = ['mc', 'mfr', 'ap', 'ss']
    scenarios = ['straight-in', 'offset']
    delays = ['baseline', 'delay=0.048', 'delay=0.096', 'delay=0.192']
    wl_alg = 'HTM'
    wlalgs_scenarios_featuresets = {
        'HTM': {
            'offset': ['roll_stick'],
            'straight-in': ['pitch_stick', 'roll_stick'],
        },
        'SteeringEntropy': {
            'offset': ['rudder_pedals'],
            'straight-in': ['throttle'],
        },
        'PSD-1': {
            'offset': ['roll_stick'],
            'straight-in': ['roll_stick'],
        },
        'PSD-2': {
            'offset': ['pitch_stick'],
            'straight-in': ['pitch_stick'],
        },
        'PSD-3': {
            'offset': ['rudder_pedals'],
            'straight-in': ['rudder_pedals'],
        },
    }
    rows = []
    for compalg in compalgs:
        for scenario in scenarios:
            for delay in delays:
                print(f"      {compalg}--{scenario}--{delay}")
                filter_dict = {'delay comp alg': [compalg, 'nc'], 'flight scenario': [scenario],
                               'delay condition': [delay],
                               'wl alg': [f"{wl_alg} - {v}" for v in wlalgs_scenarios_featuresets['HTM'][scenario]]}
                df_ = filter_df(df_master_total, filter_dict)
                if len(df_['delay comp alg'].unique()) == 1:
                    print("          **EMPTY")
                    row = {'delay comp alg': compalg, 'flight approach': scenario,
                           'delay (ms)': delay, 'subj': 'N/A', '%WL drop from no comp': 'N/A'}
                else:
                    gpby = df_.groupby('subject')
                    for subj, df_subj in gpby:
                        wl_comp = df_subj[df_subj['delay comp alg'] == compalg]['wl perceived'].values[0]
                        wl_nc = df_subj[df_subj['delay comp alg'] == 'nc']['wl perceived'].values[0]
                        pct_drop = round((wl_nc - wl_comp) / wl_nc * 100, 2)
                        row = {'delay comp alg': compalg, 'flight approach': scenario,
                               'delay (ms)': delay, 'subj': subj, '%WL drop from no comp': pct_drop}
                        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path_out)

if make_boxplots_table23:
    dir_out = "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/training=40%"
    path_t2 = "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/training=40%/table2-wldrop-from-nc.csv"
    table2 = pd.read_csv(path_t2)
    algs_colors = {
        'TLX-Raw': 'lightgrey',
        'TLX-Mental': 'lightgrey',
        'TLX-Physical': 'lightgrey',
        'TLX-Performance': 'lightgrey',
        'TLX-Frustration': 'lightgrey',
        'TLX-Effort': 'lightgrey',
        'TLX-Temporal': 'lightgrey',
        'CHR GSE': 'lightgrey',
        'CHR TDE': 'lightgrey',
        'IPSD RS': 'lightgrey',
        'IPSD PS': 'lightgrey',
        'IPSD RP': 'lightgrey',
        'FPSD RS': 'lightgrey',
        'FPSD PS': 'lightgrey',
        # 'FPSD RP': 'lightgrey',
        'SE RS': 'yellow',
        'SE PS': 'yellow',
        'SE RP': 'yellow',
        # 'HTM': 'lightblue',
        'HTM RS': 'lightblue',
        'HTM PS': 'lightblue',
        'HTM RP': 'lightblue',
        'HTM RS+RP': 'lightblue',
    }
    wls_convert = {
        'HTM RS': 'HTM-roll',
        'HTM PS': 'HTM-pitch',
        'HTM RP': 'HTM-rudder',
        'HTM RS+RP': 'HTM-rollpitch',
        'TLX-Raw': 'TLX-Raw',
        'TLX-Temporal': 'TLX-Temporal',
        'TLX-Physical': 'TLX-Physical',
        'TLX-Mental': 'TLX-Mental',
        'TLX-Effort': 'TLX-Effort',
        'TLX-Frustration': 'TLX-Frustration',
        'TLX-Performance': 'TLX-Performance',
        'CHR GSE': 'CHR-GSE',
        'CHR TDE': 'CHR-TDE',
        'SE RS': 'SteeringEntropy-roll',
        'SE PS': 'SteeringEntropy-pitch',
        'SE RP': 'SteeringEntropy-rudder',
        'IPSD RS': 'IPSD-roll',
        'IPSD PS': 'IPSD-pitch',
        'IPSD RP': 'IPSD-rudder',
        'FPSD RS': 'FPSD-roll',
        'FPSD PS': 'FPSD-pitch',
        # 'FPSD RP': 'FPSD-rudder'
    }

    algs_percentdrops = {}
    algs_pvalues = {}
    for wl in algs_colors:
        wl_ = wls_convert[wl]
        df_wl = table2[table2['wl alg'] == wl_]
        algs_percentdrops[wl] = df_wl['%WL drop from no comp'].values
        algs_pvalues[wl] = df_wl['paired ttest p-value'].values

    make_boxplots(data_dict={k: v for k, v in algs_percentdrops.items() if k != 'FPSD RP'},
                  levels_colors={k: v for k, v in algs_colors.items() if k != 'FPSD RP'},
                  ylabel="% Drop in Workload",
                  title="from Compensated to Non-Compensated Behavior",
                  suptitle="% Drop in Workload",
                  xtickrotation=90,
                  path_out=os.path.join(dir_out, "percentdrops--box.png"),
                  ylim=(-125, 75),
                  showmeans=True)

    make_boxplots(data_dict=algs_pvalues,
                  levels_colors=algs_colors,
                  ylabel="P-values of Paried T-Tests",
                  title="for Significant Difference between Compensated & Non-Compensated Behavior",
                  suptitle="P-values of Paried T-Tests",
                  xtickrotation=90,
                  path_out=os.path.join(dir_out, "pvalues--box.png"),
                  ylim=None,
                  showmeans=True)

if print_mean_scores:
    table2 = pd.read_csv(
        "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/training=40%/table2-wldrop-from-nc.csv")
    cols_score = ['%WL drop from no comp', 'paired ttest p-value']
    gpby = table2.groupby('wl alg')
    for wl, df_wl in gpby:
        print(f"\n{wl}")
        for col in cols_score:
            print(f"  {col} --> {round(df_wl[col].mean(), 3)}")

# if __name__ == "__main__":
#     main()
