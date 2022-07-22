import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.utils.utils import make_dir


def plot_data(filenames_data: dict, file_type: str, dir_output: str):
    dir_out = os.path.join(dir_output, 'data_plots')
    make_dir(dir_out)
    for fname, data in filenames_data.items():
        fname = fname.replace(f".{file_type}", "")
        path_out = os.path.join(dir_out, f"{fname}.png")
        plt.cla()
        for c in data:
            plt.plot(data[c], label=c)
        plt.title(fname)
        plt.legend()
        plt.savefig(path_out)


def plot_boxplot(data_plot, title, outpath, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.boxplot(data_plot.values())
    ax.set_xticklabels(data_plot.keys(), rotation=90)
    ax.yaxis.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.axhline(0, label="Min Possible WL", color='red', linestyle='--', linewidth=1)
    # plt.axhline(1, label="Max Possible WL", color='red', linestyle='--', linewidth=1)
    if title is not None:
        plt.title(title)
    # plt.legend()
    plt.savefig(outpath, bbox_inches="tight")
