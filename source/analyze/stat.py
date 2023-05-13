import os
import sys

import pandas as pd
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
    t_pval2 = ttest_ind(algs_data['HTM'], algs_data['TLX'])[1]
    t_pval3 = ttest_ind(algs_data['SE'], algs_data['TLX'])[1]
    # KS-Test
    ks_pval1 = kstest(algs_data['HTM'], algs_data['SE'])[1]
    ks_pval2 = kstest(algs_data['HTM'], algs_data['TLX'])[1]
    ks_pval3 = kstest(algs_data['SE'], algs_data['TLX'])[1]
    # MW-Test
    mw_pval1 = mannwhitneyu(algs_data['HTM'], algs_data['SE'])[1]
    mw_pval2 = mannwhitneyu(algs_data['HTM'], algs_data['TLX'])[1]
    mw_pval3 = mannwhitneyu(algs_data['SE'], algs_data['TLX'])[1]
    print("  T")
    print(f"    HTM vs SE --> {t_pval1}")
    print(f"    HTM vs TLX --> {t_pval2}")
    print(f"    SE vs TLX --> {t_pval3}")
    print("  KS")
    print(f"    HTM vs SE --> {ks_pval1}")
    print(f"    HTM vs TLX --> {ks_pval2}")
    print(f"    SE vs TLX --> {ks_pval3}")
    print("  MW")
    print(f"    HTM vs SE --> {mw_pval1}")
    print(f"    HTM vs TLX --> {mw_pval2}")
    print(f"    SE vs TLX --> {mw_pval3}\n\n")


"""
Test for statistically significant performance differences between Algs
"""

"""
1) % Increase from Baseline to all other WL levels
    --> 1 value per subject
"""
title = "% Increase from Baseline to ALL other WL levels"
algs_paths = {
    'HTM': os.path.join(ALGS_DIRS_IN['HTM'], "subjects_wldiffs.csv"),
    'SE': os.path.join(ALGS_DIRS_IN['SE'], "subjects_wldiffs.csv"),
    'TLX': os.path.join(ALGS_DIRS_IN['TLX'], "scores_subjects.csv")
}
algs_data = {}
for alg, alg_path in algs_paths.items():
    data = pd.read_csv(alg_path)
    algs_data[alg] = data['%Change from Baseline'].values
print("TEST 1")
run_stat_tests(algs_data)


"""
2) % Increase from Baseline to EACH other WL level
    --> 3 values per subject
"""
title = "% Increase from Baseline to each other WL level"
algs_paths = {
    'HTM': os.path.join(ALGS_DIRS_IN['HTM'], "subjects_levels_wldiffs.csv"),
    'SE': os.path.join(ALGS_DIRS_IN['SE'], "subjects_levels_wldiffs.csv"),
    'TLX': os.path.join(ALGS_DIRS_IN['SE'], "subjects_levels_wldiffs.csv")
}
algs_data = {}
for alg, alg_path in algs_paths.items():
    data = pd.read_csv(alg_path)
    values = []
    cols = [c for c in data if c!='Unnamed: 0']
    for c in cols:
        values += list(data[c].values)
    algs_data[alg] = values
    print(alg)
print("TEST 3")
run_stat_tests(algs_data)


"""
3) Degree of overlap with TLX
    --> 1 value per subject
"""
title = "Degree of overlap with TLX"
algs_paths = {
    'HTM': os.path.join(ALGS_DIRS_IN['HTM'], "subjects_overlaps.csv"),
    'SE': os.path.join(ALGS_DIRS_IN['SE'], "subjects_overlaps.csv"),
    'TLX': os.path.join(ALGS_DIRS_IN['SE'], "subjects_overlaps.csv")
}
algs_data = {}
for alg, alg_path in algs_paths.items():
    data = pd.read_csv(alg_path)
    algs_data[alg] = data['overlaps'].values
plot_hists(algs_data, DIR_OUT, title)
print("TEST 3")
run_stat_tests(algs_data)


#
# if __name__ == "__main__":
#     main()
