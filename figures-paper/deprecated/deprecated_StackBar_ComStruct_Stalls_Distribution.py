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
plt.rcParams['hatch.color'] = 'white'
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
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
    "LSTM_32",
    "GRU"
]

def read_xlsx_ComStruct_Stalls(file_name="", OURS_sheet_name=""):
    
    FunctionalUnitPipelineSaturationCycles = {}
    FunctionalUnitIssuingMutualExclusionCycles = {}
    ResultBusSaturationCycles = {}
    DispatchQueueSaturationCycles = {}
    BankConflictCycles = {}
    NoFreeOperandsCollectorUnitCycles = {}
    
    None_of_OURS_System_Stalls = []

    data_OURS = pd.read_excel(file_name, sheet_name=OURS_sheet_name)

    for idx, row in data_OURS.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        
        kernel_FunctionalUnitPipelineSaturationCycles = row["ComputeStructuralStall_Issue_out_has_no_free_slot_Cycles"]
        kernel_FunctionalUnitIssuingMutualExclusionCycles = row["ComputeStructuralStall_Issue_previous_issued_inst_exec_type_is_compute_Cycles"]
        kernel_ResultBusSaturationCycles = row["ComputeStructuralStall_Execute_result_bus_has_no_slot_for_latency_Cycles"]
        kernel_DispatchQueueSaturationCycles = row["ComputeStructuralStall_Execute_m_dispatch_reg_of_fu_is_not_empty_Cycles"]
        kernel_BankConflictCycles = row["ComputeStructuralStall_Writeback_bank_of_reg_is_not_idle_Cycles"]
        kernel_BankConflict1Cycles = row["ComputeStructuralStall_ReadOperands_bank_reg_belonged_to_was_allocated_Cycles"]
        kernel_NoFreeOperandsCollectorUnitCycles = row["ComputeStructuralStall_ReadOperands_port_num_m_in_ports_m_in_fails_as_not_found_free_cu_Cycles"]
        
        if not kernel_key in Except_keys:
            if not kernel_key in FunctionalUnitPipelineSaturationCycles.keys():
                if isinstance(kernel_FunctionalUnitPipelineSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_FunctionalUnitPipelineSaturationCycles) and \
                    isinstance(kernel_FunctionalUnitIssuingMutualExclusionCycles, (int, float)) and \
                    not pd.isna(kernel_FunctionalUnitIssuingMutualExclusionCycles) and \
                    isinstance(kernel_ResultBusSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_ResultBusSaturationCycles) and \
                    isinstance(kernel_DispatchQueueSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_DispatchQueueSaturationCycles) and \
                    isinstance(kernel_BankConflictCycles, (int, float)) and \
                    not pd.isna(kernel_BankConflictCycles) and \
                    isinstance(kernel_NoFreeOperandsCollectorUnitCycles, (int, float)) and \
                    not pd.isna(kernel_NoFreeOperandsCollectorUnitCycles):
                    
                    FunctionalUnitPipelineSaturationCycles[kernel_key] = kernel_FunctionalUnitPipelineSaturationCycles
                    FunctionalUnitIssuingMutualExclusionCycles[kernel_key] = kernel_FunctionalUnitIssuingMutualExclusionCycles
                    ResultBusSaturationCycles[kernel_key] = kernel_ResultBusSaturationCycles
                    DispatchQueueSaturationCycles[kernel_key] = kernel_DispatchQueueSaturationCycles
                    BankConflictCycles[kernel_key] = kernel_BankConflictCycles + kernel_BankConflict1Cycles
                    NoFreeOperandsCollectorUnitCycles[kernel_key] = kernel_NoFreeOperandsCollectorUnitCycles
                else:
                    continue
            else:
                if isinstance(kernel_FunctionalUnitPipelineSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_FunctionalUnitPipelineSaturationCycles) and \
                    isinstance(kernel_FunctionalUnitIssuingMutualExclusionCycles, (int, float)) and \
                    not pd.isna(kernel_FunctionalUnitIssuingMutualExclusionCycles) and \
                    isinstance(kernel_ResultBusSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_ResultBusSaturationCycles) and \
                    isinstance(kernel_DispatchQueueSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_DispatchQueueSaturationCycles) and \
                    isinstance(kernel_BankConflictCycles, (int, float)) and \
                    not pd.isna(kernel_BankConflictCycles) and \
                    isinstance(kernel_NoFreeOperandsCollectorUnitCycles, (int, float)) and \
                    not pd.isna(kernel_NoFreeOperandsCollectorUnitCycles):

                    FunctionalUnitPipelineSaturationCycles[kernel_key] += kernel_FunctionalUnitPipelineSaturationCycles
                    FunctionalUnitIssuingMutualExclusionCycles[kernel_key] += kernel_FunctionalUnitIssuingMutualExclusionCycles
                    ResultBusSaturationCycles[kernel_key] += kernel_ResultBusSaturationCycles
                    DispatchQueueSaturationCycles[kernel_key] += kernel_DispatchQueueSaturationCycles
                    BankConflictCycles[kernel_key] += kernel_BankConflictCycles + kernel_BankConflict1Cycles
                    NoFreeOperandsCollectorUnitCycles[kernel_key] += kernel_NoFreeOperandsCollectorUnitCycles
                else:
                    continue
        else:
            continue
    
    FunctionalUnitPipelineSaturationCycles_rate = {}
    FunctionalUnitIssuingMutualExclusionCycles_rate = {}
    ResultBusSaturationCycles_rate = {}
    DispatchQueueSaturationCycles_rate = {}
    BankConflictCycles_rate = {}
    NoFreeOperandsCollectorUnitCycles_rate = {}

    for key in FunctionalUnitPipelineSaturationCycles.keys():
        print("KEY: ", key)
        print(FunctionalUnitPipelineSaturationCycles[key], FunctionalUnitIssuingMutualExclusionCycles[key], \
            ResultBusSaturationCycles[key], DispatchQueueSaturationCycles[key], \
            BankConflictCycles[key], \
            NoFreeOperandsCollectorUnitCycles[key])
        all_cycles = float(FunctionalUnitPipelineSaturationCycles[key] + FunctionalUnitIssuingMutualExclusionCycles[key] + \
            ResultBusSaturationCycles[key] + DispatchQueueSaturationCycles[key] + \
            BankConflictCycles[key] + \
            NoFreeOperandsCollectorUnitCycles[key])
        FunctionalUnitPipelineSaturationCycles_rate[key] = \
            float(FunctionalUnitPipelineSaturationCycles[key]) / all_cycles
        FunctionalUnitIssuingMutualExclusionCycles_rate[key] = \
            float(FunctionalUnitIssuingMutualExclusionCycles[key]) / all_cycles
        ResultBusSaturationCycles_rate[key] = \
            float(ResultBusSaturationCycles[key]) / all_cycles
        DispatchQueueSaturationCycles_rate[key] = \
            float(DispatchQueueSaturationCycles[key]) / all_cycles
        BankConflictCycles_rate[key] = \
            float(BankConflictCycles[key]) / all_cycles
        NoFreeOperandsCollectorUnitCycles_rate[key] = \
            float(NoFreeOperandsCollectorUnitCycles[key]) / all_cycles


    return FunctionalUnitPipelineSaturationCycles_rate, FunctionalUnitIssuingMutualExclusionCycles_rate, \
           ResultBusSaturationCycles_rate, DispatchQueueSaturationCycles_rate, \
           BankConflictCycles_rate, \
           NoFreeOperandsCollectorUnitCycles_rate


def plot_figure_StackBar_System_Stalls(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    prng = read_xlsx_ComStruct_Stalls(
                "compare.xlsx", 
                OURS_sheet_name="OURS")
    
    OURS_DATA = None

    FunctionalUnitPipelineSaturationCycles_rate, FunctionalUnitIssuingMutualExclusionCycles_rate, \
    ResultBusSaturationCycles_rate, DispatchQueueSaturationCycles_rate, \
    BankConflictCycles_rate, \
    NoFreeOperandsCollectorUnitCycles_rate = \
    prng[0], prng[1], prng[2], prng[3], prng[4], prng[5]

    indexes = list(FunctionalUnitPipelineSaturationCycles_rate.keys())
    
    background_color = mcolors.rgb_to_hsv(
        mcolors.to_rgb(plt.rcParams['figure.facecolor']))[2]
    if background_color < 0.5:
        title_color = [0.8, 0.8, 1]
    else:
        title_color = np.array([19, 6, 84]) / 256

    species = [_ for _ in indexes]
    penguin_means = {}

    for key in FunctionalUnitPipelineSaturationCycles_rate.keys():
        penguin_means[key] =      ( float("%.6f" % float(FunctionalUnitPipelineSaturationCycles_rate[key])), \
                                    float("%.6f" % float(FunctionalUnitIssuingMutualExclusionCycles_rate[key])), \
                                    float("%.6f" % float(ResultBusSaturationCycles_rate[key])), \
                                    float("%.6f" % float(DispatchQueueSaturationCycles_rate[key])), \
                                    float("%.6f" % float(BankConflictCycles_rate[key])), \
                                    float("%.6f" % float(NoFreeOperandsCollectorUnitCycles_rate[key]))
                                  )
    
    x = np.arange(len(species))
    width = 0.6
    
    labels = list(penguin_means.keys())
    data = list(penguin_means.values())

    colors = ['#7fabd4', '#ff9f4e', '#7fc97f', \
              '#e377c2', '#7f7f7f', '#bcbd22']
    
    hatch = ['-', '/', '\\', 'x', '+', '/|']

    fig, ax = plt.subplots(ncols=1, nrows=1, num=style_label,
                           figsize=(15, 2.6), layout='constrained', dpi=300)

    bottom = [0] * len(labels)
    for i in range(len(data[0])):
        ax.bar(labels, [d[i] for d in data], width=width, bottom=bottom, color=colors[i], hatch=hatch[i])
        bottom = [sum(x) for x in zip(bottom, [d[i] for d in data])]

    ax.set_ylabel('Percentage')
    ax.set_title('Stacked Bar Chart')

    indexes_short_name = [indexes_short_name_dict[_] for _ in indexes]
    
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylabel('ComStruct Distribution', fontsize=14.5, y=0.4)
    ax.set_title('')
    
    ax.legend(['Functional Unit Pipeline Saturation', \
               'Functional Unit Issuing Mutual Exclusion', \
               'Result Bus Saturation', \
               'Dispatch Queue Saturation', \
               'Bank Conflict', \
               'No Free Operands Collector Unit'], \
              fontsize=11, frameon=False, shadow=False, fancybox=False, \
              framealpha=1.0, borderpad=0.3, markerfirst=True, \
              markerscale=0.2, numpoints=1, handlelength=2.0, handleheight=0.5, \
              loc='upper center', bbox_to_anchor=(0.5, -0.33), ncol=4)
    
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(indexes_short_name, rotation=30, fontsize=12)
    
    ax.grid(True, which='major', axis='y', linestyle='--', color='gray', linewidth=0.2)


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
                plot_figure_StackBar_System_Stalls(style_label=style_label)
                plt.savefig('figs/'+'StackBar_ComStruct_Stalls_Distribution.pdf', format='pdf')
    