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

def plot_scatter_x_HW_achieved_occupancy_y_Simulated_achieved_occupancy(ax, indexes, cycles, marker="", markersize=1, color="", label=""):
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
        max_error = max(y_numpy_bigger_than_x - x_numpy_bigger_than_x)
    else:
        max_error = 0
    
    if len(y_numpy_smaller_than_x) > 0:
        min_error = max(x_numpy_smaller_than_x - y_numpy_smaller_than_x)
    else:
        min_error = 0

    print(max_error, min_error)
    
    ax.plot(x, y, ls='none', marker=marker, color=color, markersize=markersize, label=label)
    
    ax.fill_between(x_numpy, x_numpy - min_error, x_numpy + max_error, 
                    facecolor=color, alpha=0.3, label="")
    

    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlim(min(x_numpy),max(x_numpy))
    ax.set_ylim(min(x_numpy),100)
    ax.set_xlabel('HW Occupancy (%)', fontsize=30)
    ax.set_ylabel('Simulated Occupancy (%)', fontsize=30)
    ax.set_title('')
    return ax

def read_xlsx_GPU_achieved_occupancy(file_name="", NCU_sheet_name="", PPT_sheet_name="", ASIM_sheet_name="", OURS_sheet_name=""):

    NCU_achieved_occupancy = {}
    PPT_achieved_occupancy = {}
    ASIM_achieved_occupancy = {}
    OURS_achieved_occupancy = {}

    data_NCU = pd.read_excel(file_name, sheet_name=NCU_sheet_name)
    data_PPT = pd.read_excel(file_name, sheet_name=PPT_sheet_name)
    data_ASIM = pd.read_excel(file_name, sheet_name=ASIM_sheet_name)
    

    for idx, row in data_ASIM.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_achieved_occupancy = row["achieved occupancy"]
        if not kernel_key in ASIM_achieved_occupancy.keys():
            if isinstance(kernel_achieved_occupancy, (int, float)):
                ASIM_achieved_occupancy[kernel_key] = kernel_achieved_occupancy
        else:
            if isinstance(kernel_achieved_occupancy, (int, float)):
                ASIM_achieved_occupancy[kernel_key] = max(kernel_achieved_occupancy, \
                                                          ASIM_achieved_occupancy[kernel_key])

    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_achieved_occupancy = row["achieved occupancy"]
        if kernel_key in ASIM_achieved_occupancy.keys():
            if not kernel_key in NCU_achieved_occupancy.keys():
                if isinstance(kernel_achieved_occupancy, (int, float)):
                    NCU_achieved_occupancy[kernel_key] = kernel_achieved_occupancy
            else:
                if isinstance(kernel_achieved_occupancy, (int, float)):
                    NCU_achieved_occupancy[kernel_key] = max(kernel_achieved_occupancy, \
                                                             NCU_achieved_occupancy[kernel_key])

    for idx, row in data_PPT.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_achieved_occupancy = row["achieved occupancy"]
        if kernel_key in ASIM_achieved_occupancy.keys():
            if not kernel_key in PPT_achieved_occupancy.keys():
                if isinstance(kernel_achieved_occupancy, (int, float)) and kernel_achieved_occupancy < 100:
                    PPT_achieved_occupancy[kernel_key] = kernel_achieved_occupancy
            else:
                if isinstance(kernel_achieved_occupancy, (int, float)) and kernel_achieved_occupancy < 100:
                    PPT_achieved_occupancy[kernel_key] = max(kernel_achieved_occupancy, \
                                                             PPT_achieved_occupancy[kernel_key])

    return NCU_achieved_occupancy, PPT_achieved_occupancy, \
           ASIM_achieved_occupancy, OURS_achieved_occupancy



def plot_figure_GPU_achieved_occupancy(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    prng = read_xlsx_GPU_achieved_occupancy(
                "compare.xlsx", 
                NCU_sheet_name="NCU",
                PPT_sheet_name="PPT",
                ASIM_sheet_name="ASIM",
                OURS_sheet_name="")
    
    NCU_achieved_occupancy, PPT_achieved_occupancy, \
    ASIM_achieved_occupancy, OURS_achieved_occupancy = prng[0], prng[1], prng[2], prng[3]

    NCU_achieved_occupancy = {k: v for k, v in sorted(NCU_achieved_occupancy.items(), key=lambda item: item[1], reverse=False)}
    PPT_achieved_occupancy = {k: PPT_achieved_occupancy[k] for k, v in sorted(NCU_achieved_occupancy.items(), key=lambda item: item[1], reverse=False)}
    ASIM_achieved_occupancy = {k: ASIM_achieved_occupancy[k] for k, v in sorted(NCU_achieved_occupancy.items(), key=lambda item: item[1], reverse=False)}

    print(NCU_achieved_occupancy)
    print(PPT_achieved_occupancy)
    print(ASIM_achieved_occupancy)

    indexes = list(NCU_achieved_occupancy.keys())

    fig, axs = plt.subplots(ncols=1, nrows=1, num=style_label,
                            figsize=(7.8, 7.8), layout='constrained')

    background_color = mcolors.rgb_to_hsv(
        mcolors.to_rgb(plt.rcParams['figure.facecolor']))[2]
    if background_color < 0.5:
        title_color = [0.8, 0.8, 1]
    else:
        title_color = np.array([19, 6, 84]) / 256
    
    
    axs.plot(NCU_achieved_occupancy.values(), NCU_achieved_occupancy.values(), \
             ls='--', color='#949494', linewidth=5, label="")
    plot_scatter_x_HW_achieved_occupancy_y_Simulated_achieved_occupancy(\
                                                axs, NCU_achieved_occupancy, \
                                                NCU_achieved_occupancy, marker='^', \
                                                markersize=20, color="#c387c3", label="NCU")
    plot_scatter_x_HW_achieved_occupancy_y_Simulated_achieved_occupancy(\
                                                axs, NCU_achieved_occupancy, \
                                                PPT_achieved_occupancy, marker='s', \
                                                markersize=20, color="#fcca99", label="PPT")
    plot_scatter_x_HW_achieved_occupancy_y_Simulated_achieved_occupancy(\
                                                axs, NCU_achieved_occupancy, \
                                                ASIM_achieved_occupancy, marker='o', \
                                                markersize=20, color="#8ad9f8", label="ASIM")
    axs.legend(loc='best', fontsize=25, frameon=True, shadow=True, \
               fancybox=False, framealpha=1.0, borderpad=0.3,\
               ncol=1, markerfirst=True, markerscale=1.3, \
               numpoints=1, handlelength=2.0)

    

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
                plot_figure_GPU_achieved_occupancy(style_label=style_label)
                plt.savefig('figs/'+style_label+'_GPU_achieved_occupancy.pdf', format='pdf')
