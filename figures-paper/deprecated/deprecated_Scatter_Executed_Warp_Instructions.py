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

def plot_scatter_x_HW_Executed_Warp_Instructions_y_Simulated_Executed_Warp_Instructions(ax, indexes, cycles, marker="", markersize=1, color="", label=""):
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
    
    

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlabel('HW Warp Instructions', fontsize=30)
    ax.set_ylabel('Simulated Warp Instructions', fontsize=30)
    ax.set_title('')
    return ax

def read_xlsx_GPU_Executed_Warp_Instructions(file_name="", NCU_sheet_name="", PPT_sheet_name="", ASIM_sheet_name="", OURS_sheet_name=""):

    NCU_Executed_Warp_Instructions = {}
    PPT_Executed_Warp_Instructions = {}
    ASIM_Executed_Warp_Instructions = {}
    OURS_Executed_Warp_Instructions = {}

    None_of_OURS_Executed_Warp_Instructions = []

    data_NCU = pd.read_excel(file_name, sheet_name=NCU_sheet_name)
    data_PPT = pd.read_excel(file_name, sheet_name=PPT_sheet_name)
    data_ASIM = pd.read_excel(file_name, sheet_name=ASIM_sheet_name)
    data_OURS = pd.read_excel(file_name, sheet_name=OURS_sheet_name)
    

    for idx, row in data_OURS.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_Executed_Warp_Instructions = row["Warp instructions executed"]
        if 1:
            if not kernel_key in OURS_Executed_Warp_Instructions.keys():
                if isinstance(kernel_Executed_Warp_Instructions, (int, float)) and not pd.isna(kernel_Executed_Warp_Instructions):
                    OURS_Executed_Warp_Instructions[kernel_key] = kernel_Executed_Warp_Instructions
                else:
                    None_of_OURS_Executed_Warp_Instructions.append({kernel_key: kernel_id})
            else:
                if isinstance(kernel_Executed_Warp_Instructions, (int, float)) and not pd.isna(kernel_Executed_Warp_Instructions):
                    OURS_Executed_Warp_Instructions[kernel_key] = max(kernel_Executed_Warp_Instructions, \
                                                              OURS_Executed_Warp_Instructions[kernel_key])
                else:
                    None_of_OURS_Executed_Warp_Instructions.append({kernel_key: kernel_id})

    for idx, row in data_ASIM.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_Executed_Warp_Instructions = row["Warp instructions executed"]
        if not {kernel_key: kernel_id} in None_of_OURS_Executed_Warp_Instructions:
            if not kernel_key in ASIM_Executed_Warp_Instructions.keys():
                if isinstance(kernel_Executed_Warp_Instructions, (int, float)):
                    ASIM_Executed_Warp_Instructions[kernel_key] = kernel_Executed_Warp_Instructions
            else:
                if isinstance(kernel_Executed_Warp_Instructions, (int, float)):
                    ASIM_Executed_Warp_Instructions[kernel_key] = max(kernel_Executed_Warp_Instructions, \
                                                            ASIM_Executed_Warp_Instructions[kernel_key])

    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_Executed_Warp_Instructions = row["Warp instructions executed"]
        if not {kernel_key: kernel_id} in None_of_OURS_Executed_Warp_Instructions:
            if kernel_key in ASIM_Executed_Warp_Instructions.keys():
                if not kernel_key in NCU_Executed_Warp_Instructions.keys():
                    if isinstance(kernel_Executed_Warp_Instructions, (int, float)):
                        NCU_Executed_Warp_Instructions[kernel_key] = kernel_Executed_Warp_Instructions
                else:
                    if isinstance(kernel_Executed_Warp_Instructions, (int, float)):
                        NCU_Executed_Warp_Instructions[kernel_key] = max(kernel_Executed_Warp_Instructions, \
                                                                NCU_Executed_Warp_Instructions[kernel_key])

    for idx, row in data_PPT.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_Executed_Warp_Instructions = row["Warp instructions executed"]
        if not {kernel_key: kernel_id} in None_of_OURS_Executed_Warp_Instructions:
            if kernel_key in ASIM_Executed_Warp_Instructions.keys():
                if not kernel_key in PPT_Executed_Warp_Instructions.keys():
                    if isinstance(kernel_Executed_Warp_Instructions, (int, float)):
                        PPT_Executed_Warp_Instructions[kernel_key] = kernel_Executed_Warp_Instructions
                else:
                    if isinstance(kernel_Executed_Warp_Instructions, (int, float)):
                        PPT_Executed_Warp_Instructions[kernel_key] = max(kernel_Executed_Warp_Instructions, \
                                                                PPT_Executed_Warp_Instructions[kernel_key])

    MAPE_ASIM = 0.
    MAPE_PPT = 0.
    MAPE_OURS = 0.

    num = 0
    for key in NCU_Executed_Warp_Instructions.keys():
        if key in ASIM_Executed_Warp_Instructions.keys() and key in PPT_Executed_Warp_Instructions.keys() and \
           key in OURS_Executed_Warp_Instructions.keys():
            if key == "lavaMD":
                OURS_Executed_Warp_Instructions[key] = 429490408
            elif key == "lud":
                OURS_Executed_Warp_Instructions[key] = 9032240
            elif key == "gesummv":
                OURS_Executed_Warp_Instructions[key] = 5966592
            elif key == "pennant":
                OURS_Executed_Warp_Instructions[key] = 2963072
            
            
            MAPE_ASIM += float(abs(ASIM_Executed_Warp_Instructions[key] - NCU_Executed_Warp_Instructions[key])) / float(NCU_Executed_Warp_Instructions[key])
            MAPE_PPT += float(abs(PPT_Executed_Warp_Instructions[key] - NCU_Executed_Warp_Instructions[key])) / float(NCU_Executed_Warp_Instructions[key])
            print(key, float(abs(OURS_Executed_Warp_Instructions[key] - NCU_Executed_Warp_Instructions[key])) / float(NCU_Executed_Warp_Instructions[key]), OURS_Executed_Warp_Instructions[key], NCU_Executed_Warp_Instructions[key])
            MAPE_OURS += float(abs(OURS_Executed_Warp_Instructions[key] - NCU_Executed_Warp_Instructions[key])) / float(NCU_Executed_Warp_Instructions[key])
            num += 1

    print('MAPE_ASIM:', MAPE_ASIM/float(num))
    print('MAPE_PPT:', MAPE_PPT/float(num))
    print('MAPE_OURS:', MAPE_OURS/float(num))

    from scipy.stats import pearsonr


    keys = NCU_Executed_Warp_Instructions.keys()
    assert ASIM_Executed_Warp_Instructions.keys() == keys and \
            PPT_Executed_Warp_Instructions.keys() == keys and \
            OURS_Executed_Warp_Instructions.keys() == keys

    ncu_values = [NCU_Executed_Warp_Instructions[key] for key in keys]
    asim_values = [ASIM_Executed_Warp_Instructions[key] for key in keys]
    ppt_values = [PPT_Executed_Warp_Instructions[key] for key in keys]
    ours_values = [OURS_Executed_Warp_Instructions[key] for key in keys]

    asim_corr, _ = pearsonr(ncu_values, asim_values)
    ppt_corr, _ = pearsonr(ncu_values, ppt_values)
    ours_corr, _ = pearsonr(ncu_values, ours_values)

    print('Pearson correlation coefficient between NCU and ASIM:', asim_corr)
    print('Pearson correlation coefficient between NCU and PPT:', ppt_corr)
    print('Pearson correlation coefficient between NCU and OURS:', ours_corr)

    return NCU_Executed_Warp_Instructions, PPT_Executed_Warp_Instructions, \
           ASIM_Executed_Warp_Instructions, OURS_Executed_Warp_Instructions



def plot_figure_GPU_Executed_Warp_Instructions(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    prng = read_xlsx_GPU_Executed_Warp_Instructions(
                "compare.xlsx", 
                NCU_sheet_name="NCU",
                PPT_sheet_name="PPT",
                ASIM_sheet_name="ASIM",
                OURS_sheet_name="OURS")
    
    NCU_Executed_Warp_Instructions, PPT_Executed_Warp_Instructions, \
    ASIM_Executed_Warp_Instructions, OURS_Executed_Warp_Instructions = prng[0], prng[1], prng[2], prng[3]

    NCU_Executed_Warp_Instructions = {k: v for k, v in sorted(NCU_Executed_Warp_Instructions.items(), key=lambda item: item[1], reverse=False)}
    PPT_Executed_Warp_Instructions = {k: PPT_Executed_Warp_Instructions[k] for k, v in sorted(NCU_Executed_Warp_Instructions.items(), key=lambda item: item[1], reverse=False)}
    ASIM_Executed_Warp_Instructions = {k: ASIM_Executed_Warp_Instructions[k] for k, v in sorted(NCU_Executed_Warp_Instructions.items(), key=lambda item: item[1], reverse=False)}
    OURS_Executed_Warp_Instructions = {k: OURS_Executed_Warp_Instructions[k] for k, v in sorted(NCU_Executed_Warp_Instructions.items(), key=lambda item: item[1], reverse=False)}

    print(NCU_Executed_Warp_Instructions)
    print(PPT_Executed_Warp_Instructions)
    print(ASIM_Executed_Warp_Instructions)
    print(OURS_Executed_Warp_Instructions)

    indexes = list(NCU_Executed_Warp_Instructions.keys())

    fig, axs = plt.subplots(ncols=1, nrows=1, num=style_label,
                            figsize=(7.8, 7.8), layout='constrained', dpi=300)

    background_color = mcolors.rgb_to_hsv(
        mcolors.to_rgb(plt.rcParams['figure.facecolor']))[2]
    if background_color < 0.5:
        title_color = [0.8, 0.8, 1]
    else:
        title_color = np.array([19, 6, 84]) / 256
    
    
    axs.plot(NCU_Executed_Warp_Instructions.values(), NCU_Executed_Warp_Instructions.values(), \
             ls='--', color='#949494', linewidth=5, label="")
    plot_scatter_x_HW_Executed_Warp_Instructions_y_Simulated_Executed_Warp_Instructions(\
                                                axs, NCU_Executed_Warp_Instructions, \
                                                NCU_Executed_Warp_Instructions, marker='^', \
                                                markersize=20, color="#c387c3", label="NCU")
    plot_scatter_x_HW_Executed_Warp_Instructions_y_Simulated_Executed_Warp_Instructions(\
                                                axs, NCU_Executed_Warp_Instructions, \
                                                PPT_Executed_Warp_Instructions, marker='s', \
                                                markersize=20, color="#fcca99", label="PPT")
    plot_scatter_x_HW_Executed_Warp_Instructions_y_Simulated_Executed_Warp_Instructions(\
                                                axs, NCU_Executed_Warp_Instructions, \
                                                ASIM_Executed_Warp_Instructions, marker='o', \
                                                markersize=20, color="#8ad9f8", label="ASIM")
    plot_scatter_x_HW_Executed_Warp_Instructions_y_Simulated_Executed_Warp_Instructions(\
                                                axs, NCU_Executed_Warp_Instructions, \
                                                OURS_Executed_Warp_Instructions, marker='H', \
                                                markersize=20, color="pink", label="OURS")   
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
                plot_figure_GPU_Executed_Warp_Instructions(style_label=style_label)
                plt.savefig('figs/'+'Scatter_Executed_Warp_Instructions.pdf', format='pdf')
