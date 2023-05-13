import os
import sys

import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DIR_OUT = os.path.join(_SOURCE_DIR, 'results')
ALGS_DIRS_IN = {
    'HTM': "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/HTM/preproc--autocorr_thresh=5; hz=5",
    'SE': "/Users/samheiserman/Desktop/repos/workload_assessor/results/post-hoc/SteeringEntropy/preproc--autocorr_thresh=5; hz=5",
    'TLX': "/Users/samheiserman/Desktop/repos/workload_assessor/results/tlx"
}

sys.path.append(_SOURCE_DIR)
from source.analyze.plot import plot_hists

"""
Test for statistically significant performance differences between Algs
"""

"""
1) % Increase from Baseline to all other WL levels
    --> 1 value per subject
"""
title = "% Increase from Baseline to all other WL levels"
algs_paths = {
    'HTM': os.path.join(ALGS_DIRS_IN['HTM'], "subjects_wldiffs.csv"),
    'SE': os.path.join(ALGS_DIRS_IN['SE'], "subjects_wldiffs.csv"),
    'TLX': os.path.join(ALGS_DIRS_IN['TLX'], "scores_subjects.csv")
}
algs_data = {}
for alg, alg_path in algs_paths.items():
    data = pd.read_csv(alg_path)
    algs_data[alg] = data['%Change from Baseline'].values
# plot_hists(algs_data, DIR_OUT, title)


"""
2) % Increase from Baseline to each other WL level
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
    algs_data[alg] = data.values  ## NEED TO SPLIT BY WLLEVEL

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

#
# if __name__ == "__main__":
#     main()
