import os
import pandas
from matplotlib import pyplot as plt

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
PATH_TLX = os.path.join(_SOURCE_DIR, 'data', 'tlx.csv')
DIR_OUT = os.path.join(_SOURCE_DIR, 'results', 'tlx')

# fulldata = []
# namestack = []

def make_boxplots(data_dict, title, path_out):
    plt.cla()
    fig, ax = plt.subplots()
    ax.boxplot(data_dict.values())
    ax.set_xticklabels(data_dict.keys())
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
    columns = ["Subject", "Run #", "Run Mode", "Mental Demand", "Physical Demand","Temporal Demand", "Performance", "Effort", "Frustration", "Raw TLX"]
    dataframe = pandas.read_csv(PATH_TLX, usecols=columns)
    newframe = dataframe.loc[1:,["Subject", "Run Mode", "Raw TLX"]]
    nftlx = newframe.loc[1:,["Raw TLX"]]
    nftlx.apply(pandas.to_numeric)
    print(newframe)

    means_newframe = newframe.groupby(["Subject", "Run Mode"])[["Raw TLX"]].mean()
    #columnsnew = ["Subject", "Run Mode", "Raw TLX"]

    print(means_newframe)

    """ box plots for run mode - all subjects combined """
    path_out = os.path.join(DIR_OUT, f"*modes_compared.png")
    modes_scores = {}
    gpby_mode = newframe.groupby('Run Mode')
    for mode, df_mode in gpby_mode:
        modes_scores[mode] = df_mode['Raw TLX'].values
    make_boxplots(modes_scores, "All Subjects", path_out)

    """ box plots for run mode score by subject """
    gpby_subj = newframe.groupby('Subject')
    for subj, df_subj in gpby_subj:
        path_out = os.path.join(DIR_OUT, f"{subj}.png")
        modes_scores = {}
        gpby_mode = df_subj.groupby('Run Mode')
        for mode, df_mode in gpby_mode:
            modes_scores[mode] = df_mode['Raw TLX'].values
        make_boxplots(modes_scores, f"{subj}", path_out)

    # means_newframe.columns = ["Subject", "Run Mode", "Raw TLX"]

    # multi_index = pandas.MultiIndex(levels = [['first', 'second'], ['a', 'b']], codes = [[0, 0, 1, 1], [0, 1, 0, 1]])
    # df = pandas.DataFrame(columns=multi_index)
    
    # means_newframe = means_newframe.set_index(['Subject', 'Run Mode']).value()
    #
    # means_newframe.unstack().plot(kind='bar')


# def extract_names(path_tlx):
#     """
#     args: file - path to csv file
#
#     extracts the names of the csv file, appends them to the list 'namestack'
#     can sort by association after extraction.
#     if you want to get run mode canbe gotten
#
#     rerturn: none
#     """
#     i = 0
#     stringindex = 0
#     namestr = ""
#     with open(path_tlx,'r') as reader:
#         for line in reader.readlines():
#             fulldata.append(line)
#             if i > 1:
#                 while line[stringindex] != ',':
#                     print(line[stringindex])
#                     namestr += line[stringindex]
#                     print('namestr: ' + str(namestr))
#                     print('stringindex: ' +str(stringindex))
#                     stringindex+=1
#                 namestack.append(namestr)
#                 #print(namestack)
#                 stringindex = 0
#                 namestr = ""
#             i+=1


if __name__ == "__main__":
    main()
