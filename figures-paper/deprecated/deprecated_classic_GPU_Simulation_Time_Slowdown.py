"""
======================
Style sheets reference
======================

This script demonstrates the different available style sheets on a
common set of example plots: scatter plot, image, bar graph, patches,
line plot and histogram.

Any of these style sheets can be imported (i.e. activated) by its name.
For example for the ggplot style:

>>> plt.style.use('ggplot')

The names of the available style sheets can be found
in the list `matplotlib.style.available`
(they are also printed in the corner of each plot below).

See more details in :ref:`Customizing Matplotlib
using style sheets<customizing-with-style-sheets>`.
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import matplotlib.ticker as ticker

import pandas as pd

np.random.seed(19680801)


def plot_scatter_x_HW_Cycles_y_Simulated_Cycles(ax, indexes, cycles, marker="", markersize=1, color="", label=""):
    """Scatter plot."""
    
    indexes_keys = [_ for _ in indexes.keys()]
    x = [indexes[_] for _ in indexes_keys]
    y = [cycles[_] for _ in indexes_keys]

    x_numpy = np.array(x)
    y_numpy = np.array(y)

    def log(x, base=10):
        return np.log(x)/np.log(base)

    max_error = max(log(y_numpy) - log(x_numpy))
    min_error = min(log(y_numpy) - log(x_numpy))

    print(max_error, min_error)
    
    ax.plot(x, y, ls='none', marker=marker, color=color, markersize=markersize, label=label)
    
    ax.fill_between(x_numpy, 10**(log(x_numpy) + min_error), 10**(log(x_numpy) + max_error), 
                    facecolor=color, alpha=0.3, label="")
    

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlim(min(x_numpy),max(x_numpy))
    ax.set_xlabel('HW Execution Time (s)', fontsize=30)
    ax.set_ylabel('Simulation Time (s)', fontsize=30)
    ax.set_title('')
    return ax

def read_xlsx_GPU_simulation_time(file_name="", NCU_sheet_name="", PPT_sheet_name="", ASIM_sheet_name="", OURS_sheet_name=""):

    NCU_simulation_time = {}
    PPT_simulation_time = {}
    ASIM_simulation_time = {}
    OURS_simulation_time = {}

    data_NCU = pd.read_excel(file_name, sheet_name=NCU_sheet_name)
    data_PPT = pd.read_excel(file_name, sheet_name=PPT_sheet_name)
    data_ASIM = pd.read_excel(file_name, sheet_name=ASIM_sheet_name)
    

    for idx, row in data_ASIM.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_simulation_time = row["Unnamed: 34"]
        if not kernel_key in ASIM_simulation_time.keys():
            if isinstance(kernel_simulation_time, (int, float)):
                ASIM_simulation_time[kernel_key] = kernel_simulation_time
        else:
            if isinstance(kernel_simulation_time, (int, float)):
                ASIM_simulation_time[kernel_key] += kernel_simulation_time

    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_simulation_time = row["Kernel execution time (ns)"] * 1e-9
        if kernel_key in ASIM_simulation_time.keys():
            if not kernel_key in NCU_simulation_time.keys():
                if isinstance(kernel_simulation_time, (int, float)):
                    NCU_simulation_time[kernel_key] = kernel_simulation_time
            else:
                if isinstance(kernel_simulation_time, (int, float)):
                    NCU_simulation_time[kernel_key] += kernel_simulation_time

    for idx, row in data_PPT.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_simulation_time = row["Unnamed: 34"] + row["Unnamed: 35"]
        if kernel_key in ASIM_simulation_time.keys():
            if not kernel_key in PPT_simulation_time.keys():
                if isinstance(kernel_simulation_time, (int, float)):
                    PPT_simulation_time[kernel_key] = kernel_simulation_time
            else:
                if isinstance(kernel_simulation_time, (int, float)):
                    PPT_simulation_time[kernel_key] += kernel_simulation_time

    
    return NCU_simulation_time, PPT_simulation_time, ASIM_simulation_time, OURS_simulation_time



def plot_figure_GPU_Simulation_Time_Slowdown(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    prng = read_xlsx_GPU_simulation_time(
                "compare.xlsx",
                NCU_sheet_name="NCU",
                PPT_sheet_name="PPT",
                ASIM_sheet_name="ASIM",
                OURS_sheet_name="")
    
    NCU_simulation_time, PPT_simulation_time, \
    ASIM_simulation_time, OURS_simulation_time = prng[0], prng[1], prng[2], prng[3]

    NCU_simulation_time = {k: v \
                           for k, v in sorted(NCU_simulation_time.items(), key=lambda item: item[1], reverse=False)}
    PPT_simulation_time = {k: PPT_simulation_time[k] \
                           for k, v in sorted(NCU_simulation_time.items(), key=lambda item: item[1], reverse=False)}
    ASIM_simulation_time = {k: ASIM_simulation_time[k] \
                           for k, v in sorted(NCU_simulation_time.items(), key=lambda item: item[1], reverse=False)}


    indexes = list(NCU_simulation_time.keys())

    fig, axs = plt.subplots(ncols=1, nrows=1, num=style_label,
                            figsize=(7.8, 7.8), layout='constrained')

    background_color = mcolors.rgb_to_hsv(
        mcolors.to_rgb(plt.rcParams['figure.facecolor']))[2]
    if background_color < 0.5:
        title_color = [0.8, 0.8, 1]
    else:
        title_color = np.array([19, 6, 84]) / 256
    
    
    axs.plot(NCU_simulation_time.values(), NCU_simulation_time.values(), \
             ls='--', color='#949494', linewidth=5, label="")
    plot_scatter_x_HW_Cycles_y_Simulated_Cycles(axs, NCU_simulation_time, NCU_simulation_time, \
                                                marker='^', markersize=20, color="#c387c3", label="NCU")
    plot_scatter_x_HW_Cycles_y_Simulated_Cycles(axs, NCU_simulation_time, PPT_simulation_time, \
                                                marker='s', markersize=20, color="#fcca99", label="PPT")
    plot_scatter_x_HW_Cycles_y_Simulated_Cycles(axs, NCU_simulation_time, ASIM_simulation_time, \
                                                marker='o', markersize=20, color="#8ad9f8", label="ASIM")
    axs.legend(loc='best', fontsize=25, frameon=True, shadow=True, fancybox=False, framealpha=1.0, borderpad=0.3,
           ncol=1, markerfirst=True, markerscale=1.3, numpoints=1, handlelength=2.0)

    

    axs.grid(True, which='major', axis='both', linestyle='--', color='gray', linewidth=1)


if __name__ == "__main__":

    print(plt.style.available)
    """
    ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
    """
    style_list = ['default', 'classic'] + sorted(
        style for style in plt.style.available
        if style != 'classic' and not style.startswith('_'))
    style_list = ['classic']

    for style_label in style_list:
        with plt.rc_context({"figure.max_open_warning": len(style_list)}):
            with plt.style.context(style_label):
                plot_figure_GPU_Simulation_Time_Slowdown(style_label=style_label)
                plt.savefig('figs/'+style_label+'_GPU_simulation_time_slowdown.pdf', format='pdf')
    