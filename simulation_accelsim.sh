#!/bin/bash

function exit_script() {                                                      
	exit 1                                                                    
}

apps_root=$(cd $(dirname $0); pwd)

cd $apps_root

CONFIG1=$apps_root/../gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config
CONFIG2=$apps_root/../gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config
ACCSIM=$apps_root/../gpu-simulator/bin/release/accel-sim.out
TRACESPROCESSING=$apps_root/../util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing

##################################################################################
###                                                                            ###
###                            cublas_GemmEx_HF_CC                             ###
###                                                                            ###
##################################################################################

# trace_dir=./ASIMTracesCollection/cublas_GemmEx_HF_CC/128x128x128/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/cublas_GemmEx_HF_CC/256x256x256/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/cublas_GemmEx_HF_CC/512x512x512/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/cublas_GemmEx_HF_CC/1024x1024x1024/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/cublas_GemmEx_HF_CC/2048x2048x2048/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

##################################################################################
###                                                                            ###
###                            cublas_GemmEx_HF_TC                             ###
###                                                                            ###
##################################################################################
# trace_dir=./ASIMTracesCollection/cublas_GemmEx_HF_TC/128x128x128/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/cublas_GemmEx_HF_TC/256x256x256/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/cublas_GemmEx_HF_TC/512x512x512/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/cublas_GemmEx_HF_TC/1024x1024x1024/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/cublas_GemmEx_HF_TC/2048x2048x2048/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

##################################################################################
###                                                                            ###
###                          cusparse_spmm_csr_HF_CC                           ###
###                                                                            ###
##################################################################################

# trace_dir=./ASIMTracesCollection/cusparse_spmm_csr_HF_CC/512x512x13107x512/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/cusparse_spmm_csr_HF_CC/1024x1024x52428x1024/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/cusparse_spmm_csr_HF_CC/2048x2048x209716x2048/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/cusparse_spmm_csr_HF_CC/4096x4096x838860x4096/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

##################################################################################
###                                                                            ###
###                                 PolyBench                                  ###
###                                                                            ###
##################################################################################

trace_dir=./ASIMTracesCollection/PolyBench/2DCONV/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/PolyBench/3DCONV/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/PolyBench/3MM/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/PolyBench/ATAX/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/PolyBench/BICG/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/PolyBench/GEMM/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/PolyBench/GESUMMV/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/PolyBench/GRAMSCHM/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/PolyBench/MVT/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

##################################################################################
###                                                                            ###
###                                  Rodinia                                   ###
###                                                                            ###
##################################################################################

trace_dir=./ASIMTracesCollection/Rodinia/b+tree/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Rodinia/backprop/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Rodinia/bfs/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Rodinia/cfd/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Rodinia/dwt2d/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Rodinia/gaussian/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Rodinia/hotspot/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Rodinia/hotspot3D/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Rodinia/huffman/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Rodinia/lavaMD/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Rodinia/lud/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Rodinia/nn/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Rodinia/nw/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Rodinia/pathfinder/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

##################################################################################
###                                                                            ###
###                                  sputnik                                   ###
###                                                                            ###
##################################################################################

# trace_dir=./ASIMTracesCollection/sputnik_spmm_csr_HF_CC/512x512x13107x512/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/sputnik_spmm_csr_HF_CC/1024x1024x52428x1024/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/sputnik_spmm_csr_HF_CC/2048x2048x209716x2048/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/sputnik_spmm_csr_HF_CC/4096x4096x838860x4096/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log


##################################################################################
###                                                                            ###
###                                vectorSparse                                ###
###                                                                            ###
##################################################################################

# trace_dir=./ASIMTracesCollection/vectorSparse_spmm_csr_HF_TC/512x512x13107x512/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/vectorSparse_spmm_csr_HF_TC/1024x1024x52428x1024/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/vectorSparse_spmm_csr_HF_TC/2048x2048x209716x2048/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/vectorSparse_spmm_csr_HF_TC/4096x4096x838860x4096/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log


##################################################################################
###                                                                            ###
###                                 DeepBench                                  ###
###                                                                            ###
##################################################################################

trace_dir=./ASIMTracesCollection/DeepBench/conv_bench_inference_halfx700x161x1x1x32x20x5x0x0x2x2/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/DeepBench/conv_bench_train_halfx700x161x1x1x32x20x5x0x0x2x2/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/DeepBench/gemm_bench_inference_halfx1760x7000x1760x0x0/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/DeepBench/gemm_bench_train_halfx1760x7000x1760x0x0/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/DeepBench/rnn_bench_inference_halfx1024x1x25xlstm/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/DeepBench/rnn_bench_train_halfx1024x1x25xlstm/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log


##################################################################################
###                                                                            ###
###                                   Tango                                    ###
###                                                                            ###
##################################################################################

trace_dir=./ASIMTracesCollection/Tango/AlexNet/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/Tango/CifarNet/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Tango/GRU/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Tango/LSTM/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

# trace_dir=./ASIMTracesCollection/Tango/ResNet/kernelslist
# $TRACESPROCESSING $trace_dir
# $ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log

trace_dir=./ASIMTracesCollection/Tango/SqueezeNet/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log



##################################################################################
###                                                                            ###
###                                   Lulesh                                   ###
###                                                                            ###
##################################################################################

trace_dir=./apps/ASIMTracesCollection/LULESH/cuda/kernelslist
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log


##################################################################################
###                                                                            ###
###                                   Pennant                                  ###
###                                                                            ###
##################################################################################

trace_dir=./apps/ASIMTracesCollection/PENNANT
$TRACESPROCESSING $trace_dir
$ACCSIM -trace $trace_dir.g -config $CONFIG1 -config $CONFIG2 > $(dirname "$trace_dir")/simulation.log
