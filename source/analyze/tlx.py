import os

import pandas
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


def make_boxplots(data_dict, title, path_out):
    plt.cla()
    fig, ax = plt.subplots()
    bplot = ax.boxplot(data_dict.values(), patch_artist=True)
    ax.set_xticklabels(data_dict.keys())
    colors = ['grey', 'orange', 'blue', 'green']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    plt.ylim(0, 1.0)
    plt.ylabel('Raw TLX')
    plt.suptitle('TLX Scores by Run Mode')
    plt.title(title)
    plt.savefig(path_out)


def main():
    """

    Takes the csv file and extracts subject, run mode, raw tlx
    Newframe is the data frame to do anything specific
    Check matplotlib/pandas documentation for more information

    """
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    columns = ["Subject", "Run #", "Run Mode", "Mental Demand", "Physical Demand", "Temporal Demand", "Performance",
               "Effort", "Frustration", "Raw TLX"]
    dataframe = pandas.read_csv(PATH_TLX, usecols=columns)
    newframe = dataframe.loc[1:, ["Subject", "Run Mode", "Raw TLX"]]
    nftlx = newframe.loc[1:, ["Raw TLX"]]
    nftlx.apply(pandas.to_numeric)

    """ box plots for run mode - all subjects combined """
    path_out = os.path.join(DIR_OUT, f"*modes_compared.png")
    modes_scores = {}
    gpby_mode = newframe.groupby('Run Mode')
    for mode, df_mode in gpby_mode:
        modes_scores[MODES_CONVERT[mode]] = df_mode['Raw TLX'].values
    make_boxplots(modes_scores, "All Subjects", path_out)

    """ box plots for run mode score by subject """
    gpby_subj = newframe.groupby('Subject')
    for subj, df_subj in gpby_subj:
        path_out = os.path.join(DIR_OUT, f"{subj}.png")
        modes_scores = {}
        gpby_mode = df_subj.groupby('Run Mode')
        for mode, df_mode in gpby_mode:
            modes_scores[MODES_CONVERT[mode]] = df_mode['Raw TLX'].values
        make_boxplots(modes_scores, f"{subj}", path_out)


if __name__ == "__main__":
    main()
