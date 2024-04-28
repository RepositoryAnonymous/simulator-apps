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
plt.rcParams['text.usetex'] = True
plt.rc_context({'hatch.linewidth': 1})
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

def plot_bar_x_application_y_L1_Hit_Rate_Error_Rate(ax, indexes, width, cycles, multiplier, bar_position=0, color="", label="", hatch=""):
    """bar plot."""
    '''
    plt.bar(x, height, width=width, color=colors, edgecolor='black')
    '''
    indexes_keys = [_ for _ in indexes]

    plt.rcParams['hatch.color'] = 'white'
    rects = ax.bar(x=bar_position, height=cycles, width=width, \
                   label=label, color=color, hatch=hatch)
    for rect in rects:
        height = rect.get_height()
        
        if height >= 0.3:
            idx = rects.index(rect)
            print(idx)
            print(height)
            if multiplier == 0:
                ax.text(rect.get_x() + rect.get_width() / 2 - 0.64, 0.275, f'{float(height)*100:.2f}\%', ha='center', va='bottom')
            elif multiplier == 2:
                ax.text(rect.get_x() + rect.get_width() / 2 + 0.64, 0.275, f'{float(height)*100:.2f}\%', ha='center', va='bottom')
            elif multiplier == 1:
                height_percentage = f'{float(rect.get_height())*100:.2f}\%'
                
                arrow_start_x = rect.get_x() + rect.get_width() / 2 + 0.05
                arrow_start_y = 0.235

                text_x = rect.get_x() + rect.get_width() / 2 - 1.5
                text_y = arrow_start_y - 0.01
                print(text_x, text_y)
                ax.annotate(
                    text=height_percentage,
                    xy=(arrow_start_x, arrow_start_y),
                    xytext=(text_x, text_y),
                    arrowprops=dict(arrowstyle="<|-", connectionstyle="arc3,rad=0", color='black'),
                    ha='center',
                    va='bottom',
                )
                

    ax.set_ylim(0, 0.3)
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
        NCU_L1_Hit_Rate_Error_Rate[kernel_key] = abs(NCU_L1_Hit_Rate[kernel_key] - NCU_L1_Hit_Rate[kernel_key])
        if kernel_key in PPT_L1_Hit_Rate.keys():
            PPT_L1_Hit_Rate_Error_Rate[kernel_key] = abs(PPT_L1_Hit_Rate[kernel_key] - NCU_L1_Hit_Rate[kernel_key])
        if kernel_key in ASIM_L1_Hit_Rate.keys():
            ASIM_L1_Hit_Rate_Error_Rate[kernel_key] = abs(ASIM_L1_Hit_Rate[kernel_key] - NCU_L1_Hit_Rate[kernel_key])
        if kernel_key in OURS_L1_Hit_Rate.keys():
            OURS_L1_Hit_Rate_Error_Rate[kernel_key] = abs(OURS_L1_Hit_Rate[kernel_key] - NCU_L1_Hit_Rate[kernel_key])
    
    print("NCU_L1_Hit_Rate_Error_Rate: ", NCU_L1_Hit_Rate_Error_Rate, "\n")
    print("PPT_L1_Hit_Rate_Error_Rate: ", PPT_L1_Hit_Rate_Error_Rate, "\n")
    print("ASIM_L1_Hit_Rate_Error_Rate: ", ASIM_L1_Hit_Rate_Error_Rate, "\n")
    print("OURS_L1_Hit_Rate_Error_Rate: ", OURS_L1_Hit_Rate_Error_Rate, "\n")

    MAPE_ASIM = 0.
    MAPE_PPT = 0.
    MAPE_OURS = 0.

    num = 0
    for key in NCU_L1_Hit_Rate_Error_Rate.keys():
        if key in ASIM_L1_Hit_Rate_Error_Rate.keys() and key in PPT_L1_Hit_Rate_Error_Rate.keys() and \
           key in OURS_L1_Hit_Rate_Error_Rate.keys():
            MAPE_ASIM += ASIM_L1_Hit_Rate_Error_Rate[key]
            MAPE_PPT += PPT_L1_Hit_Rate_Error_Rate[key]
            MAPE_OURS += OURS_L1_Hit_Rate_Error_Rate[key]
            num += 1

    print('MAPE_ASIM:', MAPE_ASIM/float(num))
    print('MAPE_PPT:', MAPE_PPT/float(num))
    print('MAPE_OURS:', MAPE_OURS/float(num))

    from scipy.stats import pearsonr


    keys = NCU_L1_Hit_Rate.keys()
    assert ASIM_L1_Hit_Rate.keys() == keys and PPT_L1_Hit_Rate.keys() == keys and OURS_L1_Hit_Rate.keys() == keys

    ncu_values = [NCU_L1_Hit_Rate[key] for key in keys]
    asim_values = [ASIM_L1_Hit_Rate[key] for key in keys]
    ppt_values = [PPT_L1_Hit_Rate[key] for key in keys]
    ours_values = [OURS_L1_Hit_Rate[key] for key in keys]

    asim_corr, _ = pearsonr(ncu_values, asim_values)
    ppt_corr, _ = pearsonr(ncu_values, ppt_values)
    ours_corr, _ = pearsonr(ncu_values, ours_values)

    print('Pearson correlation coefficient between NCU and ASIM:', asim_corr)
    print('Pearson correlation coefficient between NCU and PPT:', ppt_corr)
    print('Pearson correlation coefficient between NCU and OURS:', ours_corr)

    asim_error = []
    ppt_error = []
    ours_error = []

    for key in NCU_L1_Hit_Rate.keys():
        asim_error.append(abs(ASIM_L1_Hit_Rate[key] - NCU_L1_Hit_Rate[key]))
        ppt_error.append(abs(PPT_L1_Hit_Rate[key] - NCU_L1_Hit_Rate[key]))
        ours_error.append(abs(OURS_L1_Hit_Rate[key] - NCU_L1_Hit_Rate[key]))
        print(key, ASIM_L1_Hit_Rate[key] - NCU_L1_Hit_Rate[key], PPT_L1_Hit_Rate[key] - NCU_L1_Hit_Rate[key], OURS_L1_Hit_Rate[key] - NCU_L1_Hit_Rate[key])

    print("ASIM MAE: ", sum(asim_error) / len(asim_error))
    print("PPT MAE: ", sum(ppt_error) / len(ppt_error))
    print("OURS MAE: ", sum(ours_error) / len(ours_error))
    

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
                            figsize=(15, 2.5), layout='constrained', dpi=300)


    background_color = mcolors.rgb_to_hsv(
        mcolors.to_rgb(plt.rcParams['figure.facecolor']))[2]
    if background_color < 0.5:
        title_color = [0.8, 0.8, 1]
    else:
        title_color = np.array([19, 6, 84]) / 256
    
    NCU_L1_Hit_Rate_Error_Rate_list = [NCU_L1_Hit_Rate_Error_Rate[_] for _ in indexes]
    ASIM_L1_Hit_Rate_Error_Rate_list = [ASIM_L1_Hit_Rate_Error_Rate[_] for _ in indexes]
    PPT_L1_Hit_Rate_Error_Rate_list = [PPT_L1_Hit_Rate_Error_Rate[_] for _ in indexes]
    OURS_L1_Hit_Rate_Error_Rate_list = [OURS_L1_Hit_Rate_Error_Rate[_] for _ in indexes]

    species = [_ for _ in indexes]
    penguin_means = {}
    penguin_means['ASIM\ \ \ (MAE: 6.20\%, Corr.: 0.938)'] = tuple(float("%.2f" % float(ASIM_L1_Hit_Rate_Error_Rate_list[_])) for _ in range(len(ASIM_L1_Hit_Rate_Error_Rate_list)))
    penguin_means['PPT\ \ \ \ (MAE: 6.38\%, Corr.: 0.915)'] = tuple(float("%.2f" % float(PPT_L1_Hit_Rate_Error_Rate_list[_])) for _ in range(len(PPT_L1_Hit_Rate_Error_Rate_list)))
    penguin_means['HyFiSS\ (MAE: 5.68\%, Corr.: 0.910)'] = tuple(float("%.2f" % float(OURS_L1_Hit_Rate_Error_Rate_list[_])) for _ in range(len(OURS_L1_Hit_Rate_Error_Rate_list)))

    cluster_name = ['PPT\ \ \ \ (MAE: 6.38\%, Corr.: 0.915)', \
                    'ASIM\ \ \ (MAE: 6.20\%, Corr.: 0.938)', \
                    'HyFiSS\ (MAE: 5.68\%, Corr.: 0.910)']

    x = np.arange(len(species))
    width = 0.16
    multiplier = 0
    color = ['#4eab90', '#c55a11', '#eebf6d']
    hatch = ['//', '\\\\', 'x']
    
    for attribute, measurement in penguin_means.items():
        offset = (width + 0.05) * multiplier
        rects = plot_bar_x_application_y_L1_Hit_Rate_Error_Rate(axs, indexes, width, \
                                                                penguin_means[cluster_name[multiplier]], \
                                                                multiplier,
                                                                bar_position=x + offset, \
                                                                label=cluster_name[multiplier], \
                                                                color=color[multiplier], \
                                                                hatch=hatch[multiplier])

        multiplier += 1

    from matplotlib.ticker import MultipleLocator
    axs.yaxis.set_major_locator(MultipleLocator(0.05))
    from matplotlib.ticker import PercentFormatter
    axs.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    
    axs.tick_params(axis='x', labelsize=10)
    axs.tick_params(axis='y', labelsize=10)
    axs.set_ylabel('Sim. L1 Hit Rate Err.', fontsize=15)
    axs.set_title('')

    axs.legend(loc='upper left', fontsize=11, frameon=True, shadow=False, fancybox=False, framealpha=0.7, borderpad=0.3,
               ncol=1, markerfirst=True, markerscale=0.2, numpoints=1, handlelength=1.2, handleheight=0.5, bbox_to_anchor=(0.001, 0.97))
    
    indexes_short_name = [indexes_short_name_dict[_] for _ in indexes]

    axs.set_xticks(x)
    axs.set_xticklabels(indexes_short_name, rotation=30, fontsize=13)
    

    axs.grid(True, which='major', axis='y', linestyle='--', color='gray', linewidth=0.2)


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
                plt.savefig('figs/'+'Bar_L1_Hit_Rate_Error_Rate.pdf', format='pdf')
    