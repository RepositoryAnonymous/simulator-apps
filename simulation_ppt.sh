#!/bin/bash

function exit_script() {                                                      
	exit 1                                                                    
}

apps_root=$(cd $(dirname $0); pwd)

cd $apps_root/..

##################################################################################
###                                                                            ###
###                            cublas_GemmEx_HF_CC                             ###
###                                                                            ###
##################################################################################

# mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/cublas_GemmEx_HF_CC/128x128x128/ --sass --config QV100 --granularity 1
# mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/cublas_GemmEx_HF_CC/256x256x256/ --sass --config QV100 --granularity 1
# mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/cublas_GemmEx_HF_CC/512x512x512/ --sass --config QV100 --granularity 1
# mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/cublas_GemmEx_HF_CC/1024x1024x1024/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/cublas_GemmEx_HF_CC/2048x2048x2048/ --sass --config QV100 --granularity 1
# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/cublas_GemmEx_HF_CC/4096x4096x4096/ --sass --config QV100 --granularity 1
# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/cublas_GemmEx_HF_CC/8192x8192x8192/ --sass --config QV100 --granularity 1

##################################################################################
###                                                                            ###
###                            cublas_GemmEx_HF_TC                             ###
###                                                                            ###
##################################################################################

# mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/cublas_GemmEx_HF_TC/128x128x128/ --sass --config QV100 --granularity 1
# mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/cublas_GemmEx_HF_TC/256x256x256/ --sass --config QV100 --granularity 1
# mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/cublas_GemmEx_HF_TC/512x512x512/ --sass --config QV100 --granularity 1
# mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/cublas_GemmEx_HF_TC/1024x1024x1024/ --sass --config QV100 --granularity 1
# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/cublas_GemmEx_HF_TC/2048x2048x2048/ --sass --config QV100 --granularity 1
# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/cublas_GemmEx_HF_TC/4096x4096x4096/ --sass --config QV100 --granularity 1
# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/cublas_GemmEx_HF_TC/8192x8192x8192/ --sass --config QV100 --granularity 1

##################################################################################
###                                                                            ###
###                          cusparse_spmm_csr_HF_CC                           ###
###                                                                            ###
##################################################################################

# mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/cusparse_spmm_csr_HF_CC/512x512x13107x512/ --sass --config QV100 --granularity 1
# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/cusparse_spmm_csr_HF_CC/1024x1024x52428x1024/ --sass --config QV100 --granularity 1
# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/cusparse_spmm_csr_HF_CC/2048x2048x209716x2048/ --sass --config QV100 --granularity 1
# mpiexec -n 40 python3 ppt.py --app $apps_root/PPTTracesCollection/cusparse_spmm_csr_HF_CC/4096x4096x838860x4096/ --sass --config QV100 --granularity 1

##################################################################################
###                                                                            ###
###                                 PolyBench                                  ###
###                                                                            ###
##################################################################################

mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/PolyBench/2DCONV/ --sass --config QV100 --granularity 1
mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/PolyBench/3DCONV/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/PolyBench/3MM/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/PolyBench/ATAX/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/PolyBench/BICG/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/PolyBench/GEMM/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/PolyBench/GESUMMV/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/PolyBench/GRAMSCHM/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/PolyBench/MVT/ --sass --config QV100 --granularity 1

##################################################################################
###                                                                            ###
###                                  Rodinia                                   ###
###                                                                            ###
##################################################################################

mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/Rodinia/b+tree/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/Rodinia/backprop/ --sass --config QV100 --granularity 1
mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/Rodinia/bfs/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/Rodinia/cfd/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/Rodinia/dwt2d/ --sass --config QV100 --granularity 1
mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/Rodinia/gaussian/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/Rodinia/hotspot/ --sass --config QV100 --granularity 1
mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/Rodinia/hotspot3D/ --sass --config QV100 --granularity 1
mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/Rodinia/huffman/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/Rodinia/lavaMD/ --sass --config QV100 --granularity 1
mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/Rodinia/lud/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/Rodinia/nn/ --sass --config QV100 --granularity 1
mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/Rodinia/nw/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/Rodinia/pathfinder/ --sass --config QV100 --granularity 1


##################################################################################
###                                                                            ###
###                                  sputnik                                   ###
###                                                                            ###
##################################################################################

# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/sputnik_spmm_csr_HF_CC/512x512x13107x512/ --sass --config QV100 --granularity 1
# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/sputnik_spmm_csr_HF_CC/1024x1024x52428x1024/ --sass --config QV100 --granularity 1
# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/sputnik_spmm_csr_HF_CC/2048x2048x209716x2048/ --sass --config QV100 --granularity 1
# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/sputnik_spmm_csr_HF_CC/4096x4096x838860x4096/ --sass --config QV100 --granularity 1

##################################################################################
###                                                                            ###
###                                vectorSparse                                ###
###                                                                            ###
##################################################################################

# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/vectorSparse_spmm_csr_HF_TC/512x512x13107x512/ --sass --config QV100 --granularity 1
# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/vectorSparse_spmm_csr_HF_TC/1024x1024x52428x1024/ --sass --config QV100 --granularity 1
# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/vectorSparse_spmm_csr_HF_TC/2048x2048x209716x2048/ --sass --config QV100 --granularity 1
# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/vectorSparse_spmm_csr_HF_TC/4096x4096x838860x4096/ --sass --config QV100 --granularity 1

##################################################################################
###                                                                            ###
###                                 DeepBench                                  ###
###                                                                            ###
##################################################################################

mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/DeepBench/conv_bench_inference_halfx700x161x1x1x32x20x5x0x0x2x2/ --sass --config QV100 --granularity 1
# mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/DeepBench/conv_bench_train_halfx700x161x1x1x32x20x5x0x0x2x2/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/DeepBench/gemm_bench_inference_halfx1760x7000x1760x0x0/ --sass --config QV100 --granularity 1
mpiexec -n 20 python3 ppt.py --app $apps_root/PPTTracesCollection/DeepBench/gemm_bench_train_halfx1760x7000x1760x0x0/ --sass --config QV100 --granularity 1
mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/DeepBench/rnn_bench_inference_halfx1024x1x25xlstm/ --sass --config QV100 --granularity 1
mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/DeepBench/rnn_bench_train_halfx1024x1x25xlstm/ --sass --config QV100 --granularity 1

##################################################################################
###                                                                            ###
###                                   Tango                                    ###
###                                                                            ###
##################################################################################

mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/Tango/AlexNet/ --sass --config QV100 --granularity 1
# mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/Tango/CifarNet/ --sass --config QV100 --granularity 1
mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/Tango/GRU/ --sass --config QV100 --granularity 1
mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/Tango/LSTM/ --sass --config QV100 --granularity 1
# mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/Tango/ResNet/ --sass --config QV100 --granularity 1
mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/Tango/SqueezeNet/ --sass --config QV100 --granularity 1



##################################################################################
###                                                                            ###
###                                   Lulesh                                   ###
###                                                                            ###
##################################################################################

mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/LULESH/cuda --sass --config QV100 --granularity 1


##################################################################################
###                                                                            ###
###                                   Pennant                                  ###
###                                                                            ###
##################################################################################

mpiexec -n 10 python3 ppt.py --app $apps_root/PPTTracesCollection/PENNANT --sass --config QV100 --granularity 1
