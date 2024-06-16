#!/bin/bash

function exit_script() {                                                      
	exit 1                                                                    
}

apps_root=$(cd $(dirname $0); pwd)

cd $apps_root

CONFIG1=$apps_root/../gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM8_AA100/gpgpusim.config
CONFIG2=$apps_root/../gpu-simulator/configs/tested-cfgs/SM8_AA100/trace.config
ACCSIM=$apps_root/../gpu-simulator/bin/release/accel-sim.out
TRACESPROCESSING=$apps_root/../util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing

##################################################################################
###                                                                            ###
###                            cublas_GemmEx_HF_CC                             ###
###                                                                            ###
##################################################################################

# trace_dir=./ASIMTracesCollection_A100/cublas_GemmEx_HF_CC/128x128x128/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/cublas_GemmEx_HF_CC/256x256x256/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/cublas_GemmEx_HF_CC/512x512x512/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/cublas_GemmEx_HF_CC/1024x1024x1024/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/cublas_GemmEx_HF_CC/2048x2048x2048/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

##################################################################################
###                                                                            ###
###                            cublas_GemmEx_HF_TC                             ###
###                                                                            ###
##################################################################################
# trace_dir=./ASIMTracesCollection_A100/cublas_GemmEx_HF_TC/128x128x128/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/cublas_GemmEx_HF_TC/256x256x256/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/cublas_GemmEx_HF_TC/512x512x512/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/cublas_GemmEx_HF_TC/1024x1024x1024/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/cublas_GemmEx_HF_TC/2048x2048x2048/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

##################################################################################
###                                                                            ###
###                          cusparse_spmm_csr_HF_CC                           ###
###                                                                            ###
##################################################################################

# trace_dir=./ASIMTracesCollection_A100/cusparse_spmm_csr_HF_CC/512x512x13107x512/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/cusparse_spmm_csr_HF_CC/1024x1024x52428x1024/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/cusparse_spmm_csr_HF_CC/2048x2048x209716x2048/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/cusparse_spmm_csr_HF_CC/4096x4096x838860x4096/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

##################################################################################
###                                                                            ###
###                                 PolyBench                                  ###
###                                                                            ###
##################################################################################

trace_dir=./ASIMTracesCollection_A100/PolyBench/2DCONV/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/PolyBench/3DCONV/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/PolyBench/3MM/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/PolyBench/ATAX/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/PolyBench/BICG/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/PolyBench/GEMM/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/PolyBench/GESUMMV/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/PolyBench/GRAMSCHM/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/PolyBench/MVT/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

##################################################################################
###                                                                            ###
###                                  Rodinia                                   ###
###                                                                            ###
##################################################################################

trace_dir=./ASIMTracesCollection_A100/Rodinia/b+tree/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Rodinia/backprop/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Rodinia/bfs/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Rodinia/cfd/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Rodinia/dwt2d/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Rodinia/gaussian/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Rodinia/hotspot/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Rodinia/hotspot3D/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Rodinia/huffman/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Rodinia/lavaMD/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Rodinia/lud/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Rodinia/nn/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Rodinia/nw/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Rodinia/pathfinder/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

##################################################################################
###                                                                            ###
###                                  sputnik                                   ###
###                                                                            ###
##################################################################################

# trace_dir=./ASIMTracesCollection_A100/sputnik_spmm_csr_HF_CC/512x512x13107x512/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/sputnik_spmm_csr_HF_CC/1024x1024x52428x1024/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/sputnik_spmm_csr_HF_CC/2048x2048x209716x2048/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/sputnik_spmm_csr_HF_CC/4096x4096x838860x4096/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log


##################################################################################
###                                                                            ###
###                                vectorSparse                                ###
###                                                                            ###
##################################################################################

# trace_dir=./ASIMTracesCollection_A100/vectorSparse_spmm_csr_HF_TC/512x512x13107x512/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/vectorSparse_spmm_csr_HF_TC/1024x1024x52428x1024/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/vectorSparse_spmm_csr_HF_TC/2048x2048x209716x2048/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/vectorSparse_spmm_csr_HF_TC/4096x4096x838860x4096/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log


##################################################################################
###                                                                            ###
###                                 DeepBench                                  ###
###                                                                            ###
##################################################################################

trace_dir=./ASIMTracesCollection_A100/DeepBench/conv_bench_inference_halfx700x161x1x1x32x20x5x0x0x2x2/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/DeepBench/conv_bench_train_halfx700x161x1x1x32x20x5x0x0x2x2/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/DeepBench/gemm_bench_inference_halfx1760x7000x1760x0x0/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/DeepBench/gemm_bench_train_halfx1760x7000x1760x0x0/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/DeepBench/rnn_bench_inference_halfx1024x1x25xlstm/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/DeepBench/rnn_bench_train_halfx1024x1x25xlstm/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log


##################################################################################
###                                                                            ###
###                                   Tango                                    ###
###                                                                            ###
##################################################################################

trace_dir=./ASIMTracesCollection_A100/Tango/AlexNet/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/Tango/CifarNet/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Tango/GRU/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Tango/LSTM/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection_A100/Tango/ResNet/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/Tango/SqueezeNet/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log



##################################################################################
###                                                                            ###
###                                   Lulesh                                   ###
###                                                                            ###
##################################################################################

trace_dir=./ASIMTracesCollection_A100/LULESH/cuda/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log


##################################################################################
###                                                                            ###
###                                   Pennant                                  ###
###                                                                            ###
##################################################################################

trace_dir=./ASIMTracesCollection_A100/PENNANT
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log


##################################################################################
###                                                                            ###
###                                  pannotia                                  ###
###                                                                            ###
##################################################################################

trace_dir=./ASIMTracesCollection_A100/pannotia/bc
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/pannotia/color_max
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/pannotia/color_maxmin
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/pannotia/fw
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/pannotia/mis
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/pannotia/pagerank
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/pannotia/pagerank_spmv
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/pannotia/sssp
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection_A100/pannotia/sssp_ell
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log
