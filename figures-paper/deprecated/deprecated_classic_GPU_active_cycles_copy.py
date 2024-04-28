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

    y_bigger_than_x_indexs = []

    for i in range(len(x_numpy)):
        if y_numpy[i] > x_numpy[i]:
            y_bigger_than_x_indexs.append(i)
    
    y_smaller_than_x_indexs = []

    for i in range(len(x_numpy)):
        if y_numpy[i] < x_numpy[i]:
            y_smaller_than_x_indexs.append(i)

    y_numpy_bigger_than_x = y_numpy[y_bigger_than_x_indexs]
    x_numpy_bigger_than_x = x_numpy[y_bigger_than_x_indexs]

    y_numpy_smaller_than_x = y_numpy[y_smaller_than_x_indexs]
    x_numpy_smaller_than_x = x_numpy[y_smaller_than_x_indexs]

    def log(x, base=10):
        return np.log(x)/np.log(base)

    if len(y_numpy_bigger_than_x) > 0:
        max_error = max(log(y_numpy_bigger_than_x) - log(x_numpy_bigger_than_x))
    else:
        max_error = 0
    
    if len(y_numpy_smaller_than_x) > 0:
        min_error = max(log(x_numpy_smaller_than_x) - log(y_numpy_smaller_than_x))
    else:
        min_error = 0

    
    ax.plot(x, y, ls='none', marker=marker, color=color, markersize=markersize, label=label)
    
    ax.fill_between(x_numpy, 10**(log(x_numpy) - min_error), 10**(log(x_numpy) + max_error), 
                    facecolor=color, alpha=0.2, label="")
    

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlim(min(x_numpy),max(x_numpy))
    ax.set_ylim(min(x_numpy),max(x_numpy))
    ax.set_xlabel('HW Cycles', fontsize=30)
    ax.set_ylabel('Simulated Cycles', fontsize=30)
    ax.set_title('')
    return ax

def read_xlsx_GPU_active_cycles(file_name="", NCU_sheet_name="", PPT_sheet_name="", ASIM_sheet_name="", OURS_sheet_name=""):

    NCU_cycles = {}
    PPT_cycles = {}
    ASIM_cycles = {}
    OURS_cycles = {}

    None_of_OURS_Cycle = []

    data_NCU = pd.read_excel(file_name, sheet_name=NCU_sheet_name)
    data_PPT = pd.read_excel(file_name, sheet_name=PPT_sheet_name)
    data_ASIM = pd.read_excel(file_name, sheet_name=ASIM_sheet_name)
    data_OURS = pd.read_excel(file_name, sheet_name=OURS_sheet_name)
    

    for idx, row in data_OURS.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_cycles = row["GPU active cycles"]
        if 1:
            if not kernel_key in OURS_cycles.keys():
                if isinstance(kernel_cycles, (int, float)) and not pd.isna(kernel_cycles):
                    OURS_cycles[kernel_key] = kernel_cycles
                else:
                    None_of_OURS_Cycle.append({kernel_key: kernel_id})
            else:
                if isinstance(kernel_cycles, (int, float)) and not pd.isna(kernel_cycles):
                    OURS_cycles[kernel_key] += kernel_cycles
                else:
                    None_of_OURS_Cycle.append({kernel_key: kernel_id})

    for idx, row in data_ASIM.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_cycles = row["GPU active cycles"]
        if not {kernel_key: kernel_id} in None_of_OURS_Cycle:
            if not kernel_key in ASIM_cycles.keys():
                if isinstance(kernel_cycles, (int, float)):
                    ASIM_cycles[kernel_key] = kernel_cycles
            else:
                if isinstance(kernel_cycles, (int, float)):
                    ASIM_cycles[kernel_key] += kernel_cycles

    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_cycles = row["GPU active cycles"]
        if not {kernel_key: kernel_id} in None_of_OURS_Cycle:
            if kernel_key in ASIM_cycles.keys():
                if not kernel_key in NCU_cycles.keys():
                    if isinstance(kernel_cycles, (int, float)):
                        NCU_cycles[kernel_key] = kernel_cycles
                else:
                    if isinstance(kernel_cycles, (int, float)):
                        NCU_cycles[kernel_key] += kernel_cycles

    for idx, row in data_PPT.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_cycles = row["GPU active cycles"]
        if not {kernel_key: kernel_id} in None_of_OURS_Cycle:
            if kernel_key in ASIM_cycles.keys():
                if not kernel_key in PPT_cycles.keys():
                    if isinstance(kernel_cycles, (int, float)):
                        PPT_cycles[kernel_key] = kernel_cycles
                else:
                    if isinstance(kernel_cycles, (int, float)):
                        PPT_cycles[kernel_key] += kernel_cycles

    
    
    return NCU_cycles, PPT_cycles, ASIM_cycles, OURS_cycles



def plot_figure_GPU_active_cycles(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    prng = read_xlsx_GPU_active_cycles(
                "compare.xlsx", 
                NCU_sheet_name="NCU",
                PPT_sheet_name="PPT",
                ASIM_sheet_name="ASIM",
                OURS_sheet_name="OURS")
    
    NCU_cycles, PPT_cycles, ASIM_cycles, OURS_cycles = prng[0], prng[1], prng[2], prng[3]

    NCU_cycles = {k: v for k, v in sorted(NCU_cycles.items(), key=lambda item: item[1], reverse=False)}
    PPT_cycles = {k: PPT_cycles[k] for k, v in sorted(NCU_cycles.items(), key=lambda item: item[1], reverse=False)}
    ASIM_cycles = {k: ASIM_cycles[k] for k, v in sorted(NCU_cycles.items(), key=lambda item: item[1], reverse=False)}
    OURS_cycles = {k: OURS_cycles[k] for k, v in sorted(NCU_cycles.items(), key=lambda item: item[1], reverse=False)}


    indexes = list(NCU_cycles.keys())

    fig, axs = plt.subplots(ncols=1, nrows=1, num=style_label,
                            figsize=(7.8, 7.8), layout='constrained')

    background_color = mcolors.rgb_to_hsv(
        mcolors.to_rgb(plt.rcParams['figure.facecolor']))[2]
    if background_color < 0.5:
        title_color = [0.8, 0.8, 1]
    else:
        title_color = np.array([19, 6, 84]) / 256
    
    
        


    
    axs.plot(NCU_cycles.values(), NCU_cycles.values(), ls='--', color='#949494', linewidth=5, label="")
    plot_scatter_x_HW_Cycles_y_Simulated_Cycles(axs, NCU_cycles, NCU_cycles, marker='^', markersize=20, color="#c387c3", label="NCU")
    plot_scatter_x_HW_Cycles_y_Simulated_Cycles(axs, NCU_cycles, PPT_cycles, marker='s', markersize=20, color="#fcca99", label="PPT")
    plot_scatter_x_HW_Cycles_y_Simulated_Cycles(axs, NCU_cycles, ASIM_cycles, marker='o', markersize=20, color="#8ad9f8", label="ASIM")
    plot_scatter_x_HW_Cycles_y_Simulated_Cycles(axs, NCU_cycles, OURS_cycles, marker='H', markersize=20, color="pink", label="OURS")
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
                plot_figure_GPU_active_cycles(style_label=style_label)
                plt.savefig('figs/'+style_label+'_GPU_active_cycles_bak.eps', format='eps')
                plt.savefig('figs/'+style_label+'_GPU_active_cycles_bak.png', format='png')
    