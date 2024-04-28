"""
======================
Style sheets reference
======================

This script demonstrates the different available style sheets on a
common set of example plots: bar plot, image, bar graph, patches,
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

indexes_short_name_dict = {
    "2DConvolution": "2DConv",
    "3DConvolution": "3DConv",
    "cublas_GemmEx_HF_TC_example_128x128x128": "TGEMMx128",
    "cublas_GemmEx_HF_TC_example_256x256x256": "TGEMMx256",
    "cublas_GemmEx_HF_TC_example_512x512x512": "TGEMMx512",
    "cublas_GemmEx_HF_TC_example_1024x1024x1024": "TGEMMx1024",
    "cublas_GemmEx_HF_TC_example_2048x2048x2048": "TGEMMx2048",
    "cublas_GemmEx_HF_TC_example_4096x4096x4096": "TGEMMx4096",
    "cublas_GemmEx_HF_CC_example_128x128x128": "CGEMMx128",
    "cublas_GemmEx_HF_CC_example_256x256x256": "CGEMMx256",
    "cublas_GemmEx_HF_CC_example_512x512x512": "CGEMMx512",
    "cublas_GemmEx_HF_CC_example_1024x1024x1024": "CGEMMx1024",
    "cublas_GemmEx_HF_CC_example_2048x2048x2048": "GemmEx",
    "cublas_GemmEx_HF_CC_example_4096x4096x4096": "CGEMMx4096",
    "conv_bench_inference_halfx700x161x1x1x32x20x5x0x0x2x2": "conv_inf",
    "conv_bench_train_halfx700x161x1x1x32x20x5x0x0x2x2": "conv_train",
    "gemm_bench_inference_halfx1760x7000x1760x0x0": "gemm_inf",
    "gemm_bench_train_halfx1760x7000x1760x0x0": "gemm_train",
    "rnn_bench_inference_halfx1024x1x25xlstm": "rnn_inf",
    "rnn_bench_train_halfx1024x1x25xlstm": "rnn_train",
    "lulesh": "Lulesh",
    "pennant": "Pennant",
    "b+tree": "b+tree",
    "backprop": "backprop",
    "bfs": "bfs",
    "dwt2d": "dwt2d",
    "gaussian": "gaussian",
    "hotspot": "hotspot",
    "huffman": "huffman",
    "lavaMD": "lavaMD",
    "nn": "nn",
    "pathfinder": "pathfinder",
    "2DConvolution": "2DConv",
    "3DConvolution": "3DConv",
    "3mm": "3mm",
    "atax": "atax",
    "bicg": "bicg",
    "gemm": "gemm",
    "gesummv": "gesummv",
    "gramschmidt": "gramsch",
    "mvt": "mvt",
    "cfd": "cfd",
    "hotspot3D": "hotspot3D",
    "lud": "lud",
    "nw": "nw",
    "AN_32": "AlexNet",
    "GRU": "GRU",
    "LSTM_32": "LSTM",
    "SN_32": "SqueezeNet",
}

Except_keys = [
    "cublas_GemmEx_HF_TC_example_128x128x128",
    "cublas_GemmEx_HF_TC_example_256x256x256",
    "cublas_GemmEx_HF_TC_example_512x512x512",
    "cublas_GemmEx_HF_CC_example_1024x1024x1024",
    "cublas_GemmEx_HF_CC_example_128x128x128",
    "cublas_GemmEx_HF_CC_example_256x256x256",
    "cublas_GemmEx_HF_CC_example_512x512x512",
]

def plot_bar_x_application_y_L1_Hit_Rate_Error_Rate(ax, indexes, width, cycles, bar_position=0, color="", label=""):
    """bar plot."""
    '''
    plt.bar(x, height, width=width, color=colors, edgecolor='black')
    '''
    indexes_keys = [_ for _ in indexes]


    rects = ax.bar(x=bar_position, height=cycles, width=width, label=label)

    ax.set_ylim(0, 1.0)
    return rects

def read_xlsx_L1_Hit_Rate(file_name="", NCU_sheet_name="", PPT_sheet_name="", ASIM_sheet_name="", OURS_sheet_name=""):

    NCU_L1_Hit_Rate_Error_Rate = {}
    PPT_L1_Hit_Rate_Error_Rate = {}
    ASIM_L1_Hit_Rate_Error_Rate = {}
    OURS_L1_Hit_Rate_Error_Rate = {}

    NCU_L1_Hit_Rate = {}
    PPT_L1_Hit_Rate = {}
    ASIM_L1_Hit_Rate = {}
    OURS_L1_Hit_Rate = {}

    NCU_L1_total_requests = {}
    NCU_L1_total_requests_merge = {}

    NCU_L1_hit_requests = {}
    PPT_L1_hit_requests = {}
    ASIM_L1_hit_requests = {}
    OURS_L1_hit_requests = {}
    
    None_of_OURS_L1_Hit_Rate = []

    data_NCU = pd.read_excel(file_name, sheet_name=NCU_sheet_name)
    data_PPT = pd.read_excel(file_name, sheet_name=PPT_sheet_name)
    data_ASIM = pd.read_excel(file_name, sheet_name=ASIM_sheet_name)
    data_OURS = pd.read_excel(file_name, sheet_name=OURS_sheet_name)

    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_L1_requests = row["unified L1 cache total requests"]
        if not kernel_key in Except_keys:
            if not (kernel_key, kernel_id) in NCU_L1_total_requests.keys():
                if isinstance(kernel_L1_requests, (int, float)) and not pd.isna(kernel_L1_requests):
                    NCU_L1_total_requests[(kernel_key, kernel_id)] = kernel_L1_requests
            else:
                print("Error of processing NCU_L1_total_requests...")
                exit()
    
    print("NCU_L1_total_requests: ", NCU_L1_total_requests, "\n")

    for idx, row in data_OURS.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_L1_Hit_Rate = row["unified L1 cache hit rate"]
        
        if not kernel_key in Except_keys:
            kernel_L1_total_requests = NCU_L1_total_requests[(kernel_key, kernel_id)]
            
            if not kernel_key in OURS_L1_hit_requests.keys():
                if isinstance(kernel_L1_Hit_Rate, (int, float)) and not pd.isna(kernel_L1_Hit_Rate):
                    OURS_L1_hit_requests[kernel_key] = kernel_L1_Hit_Rate / 100.0 * kernel_L1_total_requests
                else:
                    None_of_OURS_L1_Hit_Rate.append({kernel_key: kernel_id})
            else:
                if isinstance(kernel_L1_Hit_Rate, (int, float)) and not pd.isna(kernel_L1_Hit_Rate):
                    OURS_L1_hit_requests[kernel_key] += kernel_L1_Hit_Rate / 100.0 * kernel_L1_total_requests
                else:
                    None_of_OURS_L1_Hit_Rate.append({kernel_key: kernel_id})
    
    print("OURS_L1_hit_requests: ", OURS_L1_hit_requests, "\n")
    
    for idx, row in data_PPT.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_L1_Hit_Rate = row["unified L1 cache hit rate"]
        
        if not kernel_key in Except_keys:
            kernel_L1_total_requests = NCU_L1_total_requests[(kernel_key, kernel_id)]
            
            if not kernel_key in PPT_L1_hit_requests.keys():
                if isinstance(kernel_L1_Hit_Rate, (int, float)) and not pd.isna(kernel_L1_Hit_Rate) and \
                    not {kernel_key: kernel_id} in None_of_OURS_L1_Hit_Rate:
                    PPT_L1_hit_requests[kernel_key] = kernel_L1_Hit_Rate / 100.0 * kernel_L1_total_requests
            else:
                if isinstance(kernel_L1_Hit_Rate, (int, float)) and not pd.isna(kernel_L1_Hit_Rate) and \
                    not {kernel_key: kernel_id} in None_of_OURS_L1_Hit_Rate:
                    PPT_L1_hit_requests[kernel_key] += kernel_L1_Hit_Rate / 100.0 * kernel_L1_total_requests
    
    print("PPT_L1_hit_requests: ", PPT_L1_hit_requests, "\n")

    for idx, row in data_ASIM.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_L1_Hit_Rate = row["unified L1 cache hit rate"]
        
        if not kernel_key in Except_keys:
            kernel_L1_total_requests = NCU_L1_total_requests[(kernel_key, kernel_id)]
            
            if not kernel_key in ASIM_L1_hit_requests.keys():
                if isinstance(kernel_L1_Hit_Rate, (int, float)) and not pd.isna(kernel_L1_Hit_Rate) and \
                    not {kernel_key: kernel_id} in None_of_OURS_L1_Hit_Rate:
                    ASIM_L1_hit_requests[kernel_key] = kernel_L1_Hit_Rate / 100.0 * kernel_L1_total_requests
            else:
                if isinstance(kernel_L1_Hit_Rate, (int, float)) and not pd.isna(kernel_L1_Hit_Rate) and \
                    not {kernel_key: kernel_id} in None_of_OURS_L1_Hit_Rate:
                    ASIM_L1_hit_requests[kernel_key] += kernel_L1_Hit_Rate / 100.0 * kernel_L1_total_requests
    
    print("ASIM_L1_hit_requests: ", ASIM_L1_hit_requests, "\n")

    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_L1_Hit_Rate = row["unified L1 cache hit rate"]
        
        if not kernel_key in Except_keys:
            kernel_L1_total_requests = NCU_L1_total_requests[(kernel_key, kernel_id)]
            
            if not kernel_key in NCU_L1_hit_requests.keys():
                if isinstance(kernel_L1_Hit_Rate, (int, float)) and not pd.isna(kernel_L1_Hit_Rate) and \
                    not {kernel_key: kernel_id} in None_of_OURS_L1_Hit_Rate:
                    NCU_L1_hit_requests[kernel_key] = kernel_L1_Hit_Rate / 100.0 * kernel_L1_total_requests
                    NCU_L1_total_requests_merge[kernel_key] = kernel_L1_total_requests
            else:
                if isinstance(kernel_L1_Hit_Rate, (int, float)) and not pd.isna(kernel_L1_Hit_Rate) and \
                    not {kernel_key: kernel_id} in None_of_OURS_L1_Hit_Rate:
                    NCU_L1_hit_requests[kernel_key] += kernel_L1_Hit_Rate / 100.0 * kernel_L1_total_requests
                    NCU_L1_total_requests_merge[kernel_key] += kernel_L1_total_requests

    print("NCU_L1_hit_requests: ", NCU_L1_hit_requests, "\n")

    print("NCU_L1_total_requests_merge: ", NCU_L1_total_requests_merge, "\n")

    for kernel_key in NCU_L1_hit_requests.keys():
        NCU_L1_Hit_Rate[kernel_key] = NCU_L1_hit_requests[kernel_key] / NCU_L1_total_requests_merge[kernel_key]
        if kernel_key in PPT_L1_hit_requests.keys():
            PPT_L1_Hit_Rate[kernel_key] = PPT_L1_hit_requests[kernel_key] / NCU_L1_total_requests_merge[kernel_key]
        if kernel_key in ASIM_L1_hit_requests.keys():
            ASIM_L1_Hit_Rate[kernel_key] = ASIM_L1_hit_requests[kernel_key] / NCU_L1_total_requests_merge[kernel_key]
        if kernel_key in OURS_L1_hit_requests.keys():
            OURS_L1_Hit_Rate[kernel_key] = OURS_L1_hit_requests[kernel_key] / NCU_L1_total_requests_merge[kernel_key]
    
    print("NCU_L1_Hit_Rate: ", NCU_L1_Hit_Rate, "\n")
    print("PPT_L1_Hit_Rate: ", PPT_L1_Hit_Rate, "\n")
    print("ASIM_L1_Hit_Rate: ", ASIM_L1_Hit_Rate, "\n")
    print("OURS_L1_Hit_Rate: ", OURS_L1_Hit_Rate, "\n")

    for kernel_key in NCU_L1_Hit_Rate.keys():
        NCU_L1_Hit_Rate_Error_Rate[kernel_key] = abs(NCU_L1_Hit_Rate[kernel_key] - NCU_L1_Hit_Rate[kernel_key]) / NCU_L1_Hit_Rate[kernel_key]
        if kernel_key in PPT_L1_Hit_Rate.keys():
            PPT_L1_Hit_Rate_Error_Rate[kernel_key] = abs(PPT_L1_Hit_Rate[kernel_key] - NCU_L1_Hit_Rate[kernel_key]) / NCU_L1_Hit_Rate[kernel_key]
        if kernel_key in ASIM_L1_Hit_Rate.keys():
            ASIM_L1_Hit_Rate_Error_Rate[kernel_key] = abs(ASIM_L1_Hit_Rate[kernel_key] - NCU_L1_Hit_Rate[kernel_key]) / NCU_L1_Hit_Rate[kernel_key]
        if kernel_key in OURS_L1_Hit_Rate.keys():
            OURS_L1_Hit_Rate_Error_Rate[kernel_key] = abs(OURS_L1_Hit_Rate[kernel_key] - NCU_L1_Hit_Rate[kernel_key]) / NCU_L1_Hit_Rate[kernel_key]
    
    print("NCU_L1_Hit_Rate_Error_Rate: ", NCU_L1_Hit_Rate_Error_Rate, "\n")
    print("PPT_L1_Hit_Rate_Error_Rate: ", PPT_L1_Hit_Rate_Error_Rate, "\n")
    print("ASIM_L1_Hit_Rate_Error_Rate: ", ASIM_L1_Hit_Rate_Error_Rate, "\n")
    print("OURS_L1_Hit_Rate_Error_Rate: ", OURS_L1_Hit_Rate_Error_Rate, "\n")

    return NCU_L1_Hit_Rate_Error_Rate, PPT_L1_Hit_Rate_Error_Rate, ASIM_L1_Hit_Rate_Error_Rate, OURS_L1_Hit_Rate_Error_Rate


def plot_figure_L1_Hit_Rate(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    prng = read_xlsx_L1_Hit_Rate(
                "compare.xlsx", 
                NCU_sheet_name="NCU",
                PPT_sheet_name="PPT",
                ASIM_sheet_name="ASIM",
                OURS_sheet_name="OURS")
    
    NCU_L1_Hit_Rate_Error_Rate, PPT_L1_Hit_Rate_Error_Rate, ASIM_L1_Hit_Rate_Error_Rate, OURS_L1_Hit_Rate_Error_Rate = prng[0], prng[1], prng[2], prng[3]


    indexes = list(NCU_L1_Hit_Rate_Error_Rate.keys())
    

    fig, axs = plt.subplots(ncols=1, nrows=1, num=style_label,
                            figsize=(15, 5.8), layout='constrained')


    background_color = mcolors.rgb_to_hsv(
        mcolors.to_rgb(plt.rcParams['figure.facecolor']))[2]
    if background_color < 0.5:
        title_color = [0.8, 0.8, 1]
    else:
        title_color = np.array([19, 6, 84]) / 256
    
    ASIM_L1_Hit_Rate_Error_Rate_list = [ASIM_L1_Hit_Rate_Error_Rate[_] for _ in indexes]
    PPT_L1_Hit_Rate_Error_Rate_list = [PPT_L1_Hit_Rate_Error_Rate[_] for _ in indexes]
    OURS_L1_Hit_Rate_Error_Rate_list = [OURS_L1_Hit_Rate_Error_Rate[_] for _ in indexes]

    species = [_ for _ in indexes]
    penguin_means = {}
    penguin_means['ASIM'] = tuple(float("%.2f" % float(ASIM_L1_Hit_Rate_Error_Rate_list[_])) for _ in range(len(ASIM_L1_Hit_Rate_Error_Rate_list)))
    penguin_means['PPT'] = tuple(float("%.2f" % float(PPT_L1_Hit_Rate_Error_Rate_list[_])) for _ in range(len(PPT_L1_Hit_Rate_Error_Rate_list)))
    penguin_means['OURS'] = tuple(float("%.2f" % float(OURS_L1_Hit_Rate_Error_Rate_list[_])) for _ in range(len(OURS_L1_Hit_Rate_Error_Rate_list)))

    cluster_name = ['ASIM', 'PPT', 'OURS']

    x = np.arange(len(species))
    width = 0.25
    multiplier = 0
    color = ["#c387c3", "#fcca99", "#8ad9f8"]
    cluster_name = ['ASIM', 'PPT', 'OURS']
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = plot_bar_x_application_y_L1_Hit_Rate_Error_Rate(axs, indexes, width, \
                                                                penguin_means[cluster_name[multiplier]], bar_position=x + offset, \
                                                                label=cluster_name[multiplier], color=color[multiplier])
        labels = axs.bar_label(rects, padding=3)

        for label in labels:
            label.set_rotation(45)
        multiplier += 1

    from matplotlib.ticker import MultipleLocator
    axs.yaxis.set_major_locator(MultipleLocator(0.5))
    
    axs.tick_params(axis='x', labelsize=20)
    axs.tick_params(axis='y', labelsize=10)
    axs.set_ylabel('Simulated L1 Hit Rate Err.', fontsize=20)
    axs.set_title('')

    axs.legend(loc='best', fontsize=15, frameon=True, shadow=True, fancybox=False, framealpha=1.0, borderpad=0.3,
               ncol=1, markerfirst=True, markerscale=1.3, numpoints=1, handlelength=2.0)
    
    indexes_short_name = [indexes_short_name_dict[_] for _ in indexes]

    axs.set_xticks(x)
    axs.set_xticklabels(indexes_short_name, rotation=30, fontsize=15)
    

    axs.grid(True, which='major', axis='both', linestyle='--', color='gray', linewidth=1)


if __name__ == "__main__":

    print(plt.style.available)
    """
    ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
    """
    style_list = ['default', 'classic'] + sorted(
        style for style in plt.style.available
        if style != 'classic' and not style.startswith('_'))
    style_list = ['fast']

    for style_label in style_list:
        style_label_name = "classic"
        with plt.rc_context({"figure.max_open_warning": len(style_list)}):
            with plt.style.context(style_label):
                plot_figure_L1_Hit_Rate(style_label=style_label)
                plt.savefig('figs/'+style_label_name+'_GPU_L1_Hit_Rate_Error_Rate.eps', format='eps')
                plt.savefig('figs/'+style_label_name+'_GPU_L1_Hit_Rate_Error_Rate.png', format='png')
    