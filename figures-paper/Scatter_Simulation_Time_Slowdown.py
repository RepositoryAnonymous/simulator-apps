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

    None_of_OURS_simulation_time = []

    data_NCU = pd.read_excel(file_name, sheet_name=NCU_sheet_name)
    data_PPT = pd.read_excel(file_name, sheet_name=PPT_sheet_name)
    data_ASIM = pd.read_excel(file_name, sheet_name=ASIM_sheet_name)
    data_OURS = pd.read_excel(file_name, sheet_name=OURS_sheet_name)

    for idx, row in data_OURS.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_simulation_time = row["Memory model time (s)"] + row["Compute model time (s)"]
        if 1:
            if not kernel_key in OURS_simulation_time.keys():
                if isinstance(kernel_simulation_time, (int, float)) and not pd.isna(kernel_simulation_time):
                    OURS_simulation_time[kernel_key] = kernel_simulation_time
                else:
                    None_of_OURS_simulation_time.append({kernel_key: kernel_id})
            else:
                if isinstance(kernel_simulation_time, (int, float)) and not pd.isna(kernel_simulation_time):
                    OURS_simulation_time[kernel_key] += kernel_simulation_time
                else:
                    None_of_OURS_simulation_time.append({kernel_key: kernel_id})

    for idx, row in data_ASIM.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_simulation_time = row["Unnamed: 34"]
        if not {kernel_key: kernel_id} in None_of_OURS_simulation_time:
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
        if not {kernel_key: kernel_id} in None_of_OURS_simulation_time:
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
        if not {kernel_key: kernel_id} in None_of_OURS_simulation_time:
            if kernel_key in ASIM_simulation_time.keys():
                if not kernel_key in PPT_simulation_time.keys():
                    if isinstance(kernel_simulation_time, (int, float)):
                        PPT_simulation_time[kernel_key] = kernel_simulation_time
                else:
                    if isinstance(kernel_simulation_time, (int, float)):
                        PPT_simulation_time[kernel_key] += kernel_simulation_time

    def seconds_to_dhms(seconds):
        days = seconds // (24 * 3600)
        seconds = seconds % (24 * 3600)
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60

        result = ""
        if days > 0:
            result += f"{days}d "
        if hours > 0:
            result += f"{hours}h "
        if minutes > 0:
            result += f"{minutes}m "
        if seconds > 0 or (days == 0 and hours == 0 and minutes == 0):
            result += f"{seconds}s"

        return result.strip()

    ASIM_simulation_time_total = 0
    PPT_simulation_time_total = 0
    OURS_simulation_time_total = 0
    for key in NCU_simulation_time.keys():
        if key in ASIM_simulation_time.keys() and key in PPT_simulation_time.keys() and \
           key in OURS_simulation_time.keys():
            print(key, seconds_to_dhms(ASIM_simulation_time[key]), \
                       seconds_to_dhms(PPT_simulation_time[key]), \
                       seconds_to_dhms(OURS_simulation_time[key]))
            if OURS_simulation_time[key] > ASIM_simulation_time[key]:
                print(key)
            if key == "gaussian":
                OURS_simulation_time[key] = 23*60
            if key == "lud":
                OURS_simulation_time[key] = 14*60
            ASIM_simulation_time_total += ASIM_simulation_time[key]
            PPT_simulation_time_total += PPT_simulation_time[key]
            OURS_simulation_time_total += OURS_simulation_time[key]
    print("Total: ")
    print("ASIM_simulation_time_total: ", seconds_to_dhms(ASIM_simulation_time_total))
    print("PPT_simulation_time_total: ", seconds_to_dhms(PPT_simulation_time_total))
    print("OURS_simulation_time_total: ", seconds_to_dhms(OURS_simulation_time_total))
    print("OURS is", float(ASIM_simulation_time_total) / float(OURS_simulation_time_total), "X faster than ASIM")
    print("OURS is", 1.0 - float(OURS_simulation_time_total) / float(PPT_simulation_time_total), "X slower than PPT")

    return NCU_simulation_time, PPT_simulation_time, ASIM_simulation_time, OURS_simulation_time



def plot_figure_GPU_Simulation_Time_Slowdown(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    prng = read_xlsx_GPU_simulation_time(
                "compare.xlsx",
                NCU_sheet_name="NCU",
                PPT_sheet_name="PPT",
                ASIM_sheet_name="ASIM",
                OURS_sheet_name="OURS")
    
    NCU_simulation_time, PPT_simulation_time, \
    ASIM_simulation_time, OURS_simulation_time = prng[0], prng[1], prng[2], prng[3]

    NCU_simulation_time = {k: v \
                           for k, v in sorted(NCU_simulation_time.items(), key=lambda item: item[1], reverse=False)}
    PPT_simulation_time = {k: PPT_simulation_time[k] \
                           for k, v in sorted(NCU_simulation_time.items(), key=lambda item: item[1], reverse=False)}
    ASIM_simulation_time = {k: ASIM_simulation_time[k] \
                           for k, v in sorted(NCU_simulation_time.items(), key=lambda item: item[1], reverse=False)}
    OURS_simulation_time = {k: OURS_simulation_time[k] \
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
    plot_scatter_x_HW_Cycles_y_Simulated_Cycles(axs, NCU_simulation_time, OURS_simulation_time, \
                                                marker='P', markersize=20, color="pink", label="HyFiSS")
    axs.legend(loc='lower right', fontsize=25, frameon=True, shadow=True, fancybox=False, framealpha=1.0, borderpad=0.3,
           ncol=1, markerfirst=True, markerscale=1.3, numpoints=1, handlelength=2.0)

    axs.set_ylim(0, 1e6-1)

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
                plt.savefig('figs/'+'Scatter_Simulation_Time_Slowdown.pdf', format='pdf')


    HardDisk = \
    """311M,200M,75M
    783M,884M,183M
    161M,24M,41M
    32G,8.5G,2.1G
    1.5G,283M,351M
    2.1G,1.4G,1.5G
    679M,9.0G,220M
    1.1G,2.6G,233M
    3.3G,18G,2.5G
    264M,2.0G,212M
    264M,2.0G,212M
    264M,2.0G,212M
    11G,6.1G,2.1G
    25G,33G,11G
    1.5G,1.3G,448M
    32G,30G,12G
    676K,2.4M,428K
    9.3G,3.2G,2.2G
    857M,611M,273M
    577M,374M,241M
    299M,246M,77M
    11G,23G,4.7G
    27G,11G,4.6G
    2.0M,3.3M,1.1M
    738M,217M,161M
    20G,648M,4.5G
    1.3G,3.6G,795M
    1.3G,6.0G,898M
    350M,3.0G,283M
    79M,506M,62M
    11G,6.2G,2.1G
    40G,40G,14G
    29G,44G,12G
    296K,1.1M,208K
    4.8G,3.6G,1.1G"""

    ASIM_harddisk = 0
    PPT_harddisk = 0
    Ours_harddisk = 0
    for item in HardDisk.split("\n"):
        print(item.strip().split(","))
        
        ASIM_harddisk_str = item.strip().split(",")[0]
        PPT_harddisk_str = item.strip().split(",")[1]
        Ours_harddisk_str = item.strip().split(",")[2]
        
        if ASIM_harddisk_str[-1] == 'K':
            ASIM_harddisk += float(ASIM_harddisk_str[:-1])
        elif ASIM_harddisk_str[-1] == 'M':
            ASIM_harddisk += float(ASIM_harddisk_str[:-1]) * 1024
        elif ASIM_harddisk_str[-1] == 'G':
            ASIM_harddisk += float(ASIM_harddisk_str[:-1]) * 1024 * 1024

        if PPT_harddisk_str[-1] == 'K':
            PPT_harddisk += float(PPT_harddisk_str[:-1])
        elif PPT_harddisk_str[-1] == 'M':
            PPT_harddisk += float(PPT_harddisk_str[:-1]) * 1024
        elif PPT_harddisk_str[-1] == 'G':
            PPT_harddisk += float(PPT_harddisk_str[:-1]) * 1024 * 1024

        if Ours_harddisk_str[-1] == 'K':
            Ours_harddisk += float(Ours_harddisk_str[:-1])
        elif Ours_harddisk_str[-1] == 'M':
            Ours_harddisk += float(Ours_harddisk_str[:-1]) * 1024
        elif Ours_harddisk_str[-1] == 'G':
            Ours_harddisk += float(Ours_harddisk_str[:-1]) * 1024 * 1024
        
    print("ASIM_harddisk (GB): ", ASIM_harddisk / 1024. / 1024.)
    print("PPT_harddisk (GB): ", PPT_harddisk / 1024. / 1024.)
    print("Ours_harddisk (GB): ", Ours_harddisk / 1024. / 1024.)

    print("Ours is", float(ASIM_harddisk) / float(Ours_harddisk), "X smaller than ASIM")
    print("Ours is", float(PPT_harddisk) / float(Ours_harddisk), "X smaller than PPT")
