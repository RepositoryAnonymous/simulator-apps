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

def plot_scatter_x_HW_IPC_y_Simulated_IPC(ax, indexes, cycles, marker="", markersize=1, color="", label=""):
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
    
    

    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlim(min(x_numpy),max(x_numpy))
    max_ylim = 0
    if len(y_numpy_bigger_than_x) == 0:
        max_ylim = max(x_numpy)
    else:
        max_ylim = max(max(y_numpy_bigger_than_x), max(x_numpy))
    ax.set_ylim(min(x_numpy), 5)
    ax.set_xlabel('HW IPC per SM', fontsize=30)
    ax.set_ylabel('Simulated IPC per SM', fontsize=30)
    ax.set_title('')
    return ax

def read_xlsx_GPU_IPC(file_name="", NCU_sheet_name="", PPT_sheet_name="", ASIM_sheet_name="", OURS_sheet_name=""):

    NCU_IPC = {}
    PPT_IPC = {}
    ASIM_IPC = {}
    OURS_IPC = {}

    NCU_IPC_avg = {}
    PPT_IPC_avg = {}
    ASIM_IPC_avg = {}
    OURS_IPC_avg = {}

    None_of_OURS_IPC = []

    data_NCU = pd.read_excel(file_name, sheet_name=NCU_sheet_name)
    data_PPT = pd.read_excel(file_name, sheet_name=PPT_sheet_name)
    data_ASIM = pd.read_excel(file_name, sheet_name=ASIM_sheet_name)
    data_OURS = pd.read_excel(file_name, sheet_name=OURS_sheet_name)
    

    for idx, row in data_OURS.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_IPC = row["Instructions executed per clock cycle (IPC)"]
        if 1:
            if not kernel_key in OURS_IPC.keys():
                if isinstance(kernel_IPC, (int, float)) and not pd.isna(kernel_IPC):
                    OURS_IPC[kernel_key] = kernel_IPC
                    OURS_IPC_avg[kernel_key] = [kernel_IPC]
                else:
                    None_of_OURS_IPC.append({kernel_key: kernel_id})
            else:
                if isinstance(kernel_IPC, (int, float)) and not pd.isna(kernel_IPC):
                    OURS_IPC[kernel_key]  = max(kernel_IPC, OURS_IPC[kernel_key])
                    OURS_IPC_avg[kernel_key].append(kernel_IPC)
                else:
                    None_of_OURS_IPC.append({kernel_key: kernel_id})

    for idx, row in data_ASIM.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_IPC = row["Instructions executed per clock cycle (IPC)"]
        if not {kernel_key: kernel_id} in None_of_OURS_IPC:
            if not kernel_key in ASIM_IPC.keys():
                if isinstance(kernel_IPC, (int, float)):
                    ASIM_IPC[kernel_key] = kernel_IPC
                    ASIM_IPC_avg[kernel_key] = [kernel_IPC]
            else:
                if isinstance(kernel_IPC, (int, float)):
                    ASIM_IPC[kernel_key] = max(kernel_IPC, ASIM_IPC[kernel_key])
                    ASIM_IPC_avg[kernel_key].append(kernel_IPC)

    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_IPC = row["Instructions executed per clock cycle (IPC)"]
        if not {kernel_key: kernel_id} in None_of_OURS_IPC:
            if kernel_key in ASIM_IPC.keys():
                if not kernel_key in NCU_IPC.keys():
                    if isinstance(kernel_IPC, (int, float)):
                        NCU_IPC[kernel_key] = kernel_IPC
                        NCU_IPC_avg[kernel_key] = [kernel_IPC]
                else:
                    if isinstance(kernel_IPC, (int, float)):
                        NCU_IPC[kernel_key] = max(kernel_IPC, NCU_IPC[kernel_key])
                        NCU_IPC_avg[kernel_key].append(kernel_IPC)

    for idx, row in data_PPT.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_IPC = row["Instructions executed per clock cycle (IPC)"]
        if not {kernel_key: kernel_id} in None_of_OURS_IPC:
            if kernel_key in ASIM_IPC.keys():
                if not kernel_key in PPT_IPC.keys():
                    if isinstance(kernel_IPC, (int, float)) and kernel_IPC < 100:
                        PPT_IPC[kernel_key] = kernel_IPC
                        PPT_IPC_avg[kernel_key] = [kernel_IPC]
                else:
                    if isinstance(kernel_IPC, (int, float)) and kernel_IPC < 100:
                        PPT_IPC[kernel_key] = max(kernel_IPC, PPT_IPC[kernel_key])
                        PPT_IPC_avg[kernel_key].append(kernel_IPC)

    MAPE_ASIM = 0.
    MAPE_PPT = 0.
    MAPE_OURS = 0.

    num = 0
    for key in NCU_IPC.keys():
        if key in ASIM_IPC.keys() and key in PPT_IPC.keys() and \
           key in OURS_IPC.keys():

            MAPE_ASIM += float(abs(ASIM_IPC[key] - NCU_IPC[key])) / float(NCU_IPC[key])
            MAPE_PPT += float(abs(PPT_IPC[key] - NCU_IPC[key])) / float(NCU_IPC[key])
            MAPE_OURS += float(abs(OURS_IPC[key] - NCU_IPC[key])) / float(NCU_IPC[key])
            MAPE_ASIM += float(abs(sum(ASIM_IPC_avg[key])/float(len(ASIM_IPC_avg[key])) - \
                                    sum(NCU_IPC_avg[key])/float(len(NCU_IPC_avg[key])))) / \
                                        float(sum(NCU_IPC_avg[key])/float(len(NCU_IPC_avg[key]))
                                        )
            MAPE_PPT += float(abs(sum(PPT_IPC_avg[key])/float(len(PPT_IPC_avg[key])) - \
                                    sum(NCU_IPC_avg[key])/float(len(NCU_IPC_avg[key])))) / \
                                        float(sum(NCU_IPC_avg[key])/float(len(NCU_IPC_avg[key]))
                                        )
            MAPE_OURS += float(abs(sum(OURS_IPC_avg[key])/float(len(OURS_IPC_avg[key])) - \
                                    sum(NCU_IPC_avg[key])/float(len(NCU_IPC_avg[key])))) / \
                                        float(sum(NCU_IPC_avg[key])/float(len(NCU_IPC_avg[key]))
                                        )
            
            num += 1

    print('MAPE_ASIM:', MAPE_ASIM/float(num))
    print('MAPE_PPT:', MAPE_PPT/float(num))
    print('MAPE_OURS:', MAPE_OURS/float(num))

    from scipy.stats import pearsonr


    keys = NCU_IPC.keys()
    assert ASIM_IPC.keys() == keys and \
            PPT_IPC.keys() == keys and \
            OURS_IPC.keys() == keys

    ncu_values = [NCU_IPC[key] for key in keys]
    asim_values = [ASIM_IPC[key] for key in keys]
    ppt_values = [PPT_IPC[key] for key in keys]
    ours_values = [OURS_IPC[key] for key in keys]

    asim_corr, _ = pearsonr(ncu_values, asim_values)
    ppt_corr, _ = pearsonr(ncu_values, ppt_values)
    ours_corr, _ = pearsonr(ncu_values, ours_values)

    print('Pearson correlation coefficient between NCU and ASIM:', asim_corr)
    print('Pearson correlation coefficient between NCU and PPT:', ppt_corr)
    print('Pearson correlation coefficient between NCU and OURS:', ours_corr)

    return NCU_IPC, PPT_IPC, ASIM_IPC, OURS_IPC



def plot_figure_GPU_IPC(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    prng = read_xlsx_GPU_IPC(
                "compare.xlsx", 
                NCU_sheet_name="NCU",
                PPT_sheet_name="PPT",
                ASIM_sheet_name="ASIM",
                OURS_sheet_name="OURS")
    
    NCU_IPC, PPT_IPC, ASIM_IPC, OURS_IPC = prng[0], prng[1], prng[2], prng[3]

    NCU_IPC = {k: v for k, v in sorted(NCU_IPC.items(), key=lambda item: item[1], reverse=False)}
    PPT_IPC = {k: PPT_IPC[k] for k, v in sorted(NCU_IPC.items(), key=lambda item: item[1], reverse=False)}
    ASIM_IPC = {k: ASIM_IPC[k] for k, v in sorted(NCU_IPC.items(), key=lambda item: item[1], reverse=False)}
    OURS_IPC = {k: OURS_IPC[k] for k, v in sorted(NCU_IPC.items(), key=lambda item: item[1], reverse=False)}

    print(NCU_IPC)
    print(PPT_IPC)
    print(ASIM_IPC)

    indexes = list(NCU_IPC.keys())

    fig, axs = plt.subplots(ncols=1, nrows=1, num=style_label,
                            figsize=(7.8, 7.8), layout='constrained')

    background_color = mcolors.rgb_to_hsv(
        mcolors.to_rgb(plt.rcParams['figure.facecolor']))[2]
    if background_color < 0.5:
        title_color = [0.8, 0.8, 1]
    else:
        title_color = np.array([19, 6, 84]) / 256
    
    
    axs.plot(NCU_IPC.values(), NCU_IPC.values(), \
             ls='--', color='#949494', linewidth=5, label="")
    plot_scatter_x_HW_IPC_y_Simulated_IPC(axs, NCU_IPC, \
                                          NCU_IPC, marker='^', \
                                          markersize=20, color="#c387c3", label="NCU")
    plot_scatter_x_HW_IPC_y_Simulated_IPC(axs, NCU_IPC, \
                                          PPT_IPC, marker='s', \
                                          markersize=20, color="#fcca99", label="PPT")
    plot_scatter_x_HW_IPC_y_Simulated_IPC(axs, NCU_IPC, \
                                          ASIM_IPC, marker='o', \
                                          markersize=20, color="#8ad9f8", label="ASIM")
    plot_scatter_x_HW_IPC_y_Simulated_IPC(axs, NCU_IPC, \
                                          OURS_IPC, marker='H', \
                                          markersize=20, color="pink", label="OURS")
    axs.legend(loc='best', fontsize=25, frameon=True, shadow=True, fancybox=False, framealpha=1.0, borderpad=0.3,
               ncol=1, markerfirst=True, markerscale=1.3, numpoints=1, handlelength=2.0)

    

    axs.grid(True, which='major', axis='y', linestyle='--', color='gray', linewidth=1)


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
                plot_figure_GPU_IPC(style_label=style_label)
                plt.savefig('figs/'+'Scatter_IPC.pdf', format='pdf')
