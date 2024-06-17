#!/bin/bash

# Stable Version, 20240617

function exit_script() {                                                      
	exit 1                                                                    
}

apps_root=$(cd $(dirname $0); pwd)

cd $apps_root
rm -rf ./OursTracesCollection_A100
mkdir ./OursTracesCollection_A100

mkdir ./OursTracesCollection_A100/cublas_GemmEx_HF_CC
# mkdir ./OursTracesCollection_A100/cublas_GemmEx_HF_TC
# mkdir ./OursTracesCollection_A100/cusparse_spmm_csr_HF_CC
mkdir ./OursTracesCollection_A100/PolyBench
mkdir ./OursTracesCollection_A100/Rodinia
# mkdir ./OursTracesCollection_A100/sputnik_spmm_csr_HF_CC
# mkdir ./OursTracesCollection_A100/vectorSparse_spmm_csr_HF_TC
mkdir ./OursTracesCollection_A100/DeepBench
mkdir ./OursTracesCollection_A100/Tango
mkdir ./OursTracesCollection_A100/LULESH
mkdir ./OursTracesCollection_A100/PENNANT
mkdir ./OursTracesCollection_A100/pannotia

##################################################################################
###                                                                            ###
###                            cublas_GemmEx_HF_CC                             ###
###                                                                            ###
##################################################################################

cd $apps_root/cublas_GemmEx_HF_CC
make clean && make

# LD_PRELOAD=../../tracing-tool/tracer.so ./cublas_GemmEx_HF_CC_example -m 128 -n 128 -k 128
# mkdir ../OursTracesCollection_A100/cublas_GemmEx_HF_CC/128x128x128
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cublas_GemmEx_HF_CC/128x128x128

# LD_PRELOAD=../../tracing-tool/tracer.so ./cublas_GemmEx_HF_CC_example -m 256 -n 256 -k 256
# mkdir ../OursTracesCollection_A100/cublas_GemmEx_HF_CC/256x256x256
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cublas_GemmEx_HF_CC/256x256x256

# LD_PRELOAD=../../tracing-tool/tracer.so ./cublas_GemmEx_HF_CC_example -m 512 -n 512 -k 512
# mkdir ../OursTracesCollection_A100/cublas_GemmEx_HF_CC/512x512x512
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cublas_GemmEx_HF_CC/512x512x512

# LD_PRELOAD=../../tracing-tool/tracer.so ./cublas_GemmEx_HF_CC_example -m 1024 -n 1024 -k 1024
# mkdir ../OursTracesCollection_A100/cublas_GemmEx_HF_CC/1024x1024x1024
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cublas_GemmEx_HF_CC/1024x1024x1024

LD_PRELOAD=../../tracing-tool/tracer.so ./cublas_GemmEx_HF_CC_example -m 2048 -n 2048 -k 2048
mkdir ../OursTracesCollection_A100/cublas_GemmEx_HF_CC/2048x2048x2048
mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cublas_GemmEx_HF_CC/2048x2048x2048

# LD_PRELOAD=../../tracing-tool/tracer.so ./cublas_GemmEx_HF_CC_example -m 4096 -n 4096 -k 4096
# mkdir ../OursTracesCollection_A100/cublas_GemmEx_HF_CC/4096x4096x4096
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cublas_GemmEx_HF_CC/4096x4096x4096

# LD_PRELOAD=../../tracing-tool/tracer.so ./cublas_GemmEx_HF_CC_example -m 8192 -n 8192 -k 8192
# mkdir ../OursTracesCollection_A100/cublas_GemmEx_HF_CC/8192x8192x8192
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cublas_GemmEx_HF_CC/8192x8192x8192
 
##################################################################################
###                                                                            ###
###                            cublas_GemmEx_HF_TC                             ###
###                                                                            ###
##################################################################################

# cd $apps_root/cublas_GemmEx_HF_TC
# make clean && make
# LD_PRELOAD=../../tracing-tool/tracer.so ./cublas_GemmEx_HF_TC_example -m 128 -n 128 -k 128
# mkdir ../OursTracesCollection_A100/cublas_GemmEx_HF_TC/128x128x128
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cublas_GemmEx_HF_TC/128x128x128

# LD_PRELOAD=../../tracing-tool/tracer.so ./cublas_GemmEx_HF_TC_example -m 256 -n 256 -k 256
# mkdir ../OursTracesCollection_A100/cublas_GemmEx_HF_TC/256x256x256
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cublas_GemmEx_HF_TC/256x256x256

# LD_PRELOAD=../../tracing-tool/tracer.so ./cublas_GemmEx_HF_TC_example -m 512 -n 512 -k 512
# mkdir ../OursTracesCollection_A100/cublas_GemmEx_HF_TC/512x512x512
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cublas_GemmEx_HF_TC/512x512x512

# LD_PRELOAD=../../tracing-tool/tracer.so ./cublas_GemmEx_HF_TC_example -m 1024 -n 1024 -k 1024
# mkdir ../OursTracesCollection_A100/cublas_GemmEx_HF_TC/1024x1024x1024
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cublas_GemmEx_HF_TC/1024x1024x1024

# LD_PRELOAD=../../tracing-tool/tracer.so ./cublas_GemmEx_HF_TC_example -m 2048 -n 2048 -k 2048
# mkdir ../OursTracesCollection_A100/cublas_GemmEx_HF_TC/2048x2048x2048
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cublas_GemmEx_HF_TC/2048x2048x2048

# LD_PRELOAD=../../tracing-tool/tracer.so ./cublas_GemmEx_HF_TC_example -m 4096 -n 4096 -k 4096
# mkdir ../OursTracesCollection_A100/cublas_GemmEx_HF_TC/4096x4096x4096
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cublas_GemmEx_HF_TC/4096x4096x4096

# LD_PRELOAD=../../tracing-tool/tracer.so ./cublas_GemmEx_HF_TC_example -m 8192 -n 8192 -k 8192
# mkdir ../OursTracesCollection_A100/cublas_GemmEx_HF_TC/8192x8192x8192
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cublas_GemmEx_HF_TC/8192x8192x8192

##################################################################################
###                                                                            ###
###                          cusparse_spmm_csr_HF_CC                           ###
###                                                                            ###
##################################################################################

# cd $apps_root/cusparse_spmm_csr_HF_CC
# make clean && make

# # 95% zero
# LD_PRELOAD=../../tracing-tool/tracer.so ./cusparse_spmm_csr_HF_CC_example --A_num_rows 512 --A_num_cols 512 --A_nnz 13107 --B_num_cols 512 
# mkdir ../OursTracesCollection_A100/cusparse_spmm_csr_HF_CC/512x512x13107x512
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cusparse_spmm_csr_HF_CC/512x512x13107x512

# # 95% zero
# LD_PRELOAD=../../tracing-tool/tracer.so ./cusparse_spmm_csr_HF_CC_example --A_num_rows 1024 --A_num_cols 1024 --A_nnz 52428 --B_num_cols 1024 
# mkdir ../OursTracesCollection_A100/cusparse_spmm_csr_HF_CC/1024x1024x52428x1024
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cusparse_spmm_csr_HF_CC/1024x1024x52428x1024

# # 95% zero
# LD_PRELOAD=../../tracing-tool/tracer.so ./cusparse_spmm_csr_HF_CC_example --A_num_rows 2048 --A_num_cols 2048 --A_nnz 209716 --B_num_cols 2048 
# mkdir ../OursTracesCollection_A100/cusparse_spmm_csr_HF_CC/2048x2048x209716x2048
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cusparse_spmm_csr_HF_CC/2048x2048x209716x2048

# # 95% zero
# LD_PRELOAD=../../tracing-tool/tracer.so ./cusparse_spmm_csr_HF_CC_example --A_num_rows 4096 --A_num_cols 4096 --A_nnz 838860 --B_num_cols 4096 
# mkdir ../OursTracesCollection_A100/cusparse_spmm_csr_HF_CC/4096x4096x838860x4096
# mv -f memory_traces/ sass_traces/ configs/ ../OursTracesCollection_A100/cusparse_spmm_csr_HF_CC/4096x4096x838860x4096

##################################################################################
###                                                                            ###
###                                 PolyBench                                  ###
###                                                                            ###
##################################################################################

cd $apps_root/PolyBench/
make clean && make

cd $apps_root/PolyBench/CUDA/2DCONV
LD_PRELOAD=../../../../tracing-tool/tracer.so ./2DConvolution.exe 
trace_save_dir=../../../OursTracesCollection_A100/PolyBench/2DCONV
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/PolyBench/CUDA/3DCONV
LD_PRELOAD=../../../../tracing-tool/tracer.so ./3DConvolution.exe
trace_save_dir=../../../OursTracesCollection_A100/PolyBench/3DCONV
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/PolyBench/CUDA/3MM
LD_PRELOAD=../../../../tracing-tool/tracer.so ./3mm.exe
trace_save_dir=../../../OursTracesCollection_A100/PolyBench/3MM
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/PolyBench/CUDA/ATAX
LD_PRELOAD=../../../../tracing-tool/tracer.so ./atax.exe
trace_save_dir=../../../OursTracesCollection_A100/PolyBench/ATAX
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/PolyBench/CUDA/BICG
LD_PRELOAD=../../../../tracing-tool/tracer.so ./bicg.exe
trace_save_dir=../../../OursTracesCollection_A100/PolyBench/BICG
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/PolyBench/CUDA/GEMM
LD_PRELOAD=../../../../tracing-tool/tracer.so ./gemm.exe
trace_save_dir=../../../OursTracesCollection_A100/PolyBench/GEMM
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/PolyBench/CUDA/GESUMMV
LD_PRELOAD=../../../../tracing-tool/tracer.so ./gesummv.exe
trace_save_dir=../../../OursTracesCollection_A100/PolyBench/GESUMMV
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/PolyBench/CUDA/GRAMSCHM
LD_PRELOAD=../../../../tracing-tool/tracer.so ./gramschmidt.exe
trace_save_dir=../../../OursTracesCollection_A100/PolyBench/GRAMSCHM
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/PolyBench/CUDA/MVT
LD_PRELOAD=../../../../tracing-tool/tracer.so ./mvt.exe
trace_save_dir=../../../OursTracesCollection_A100/PolyBench/MVT
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

##################################################################################
###                                                                            ###
###                                  Rodinia                                   ###
###                                                                            ###
##################################################################################

cd $apps_root/Rodinia/
make clean && make

cd $apps_root/Rodinia/src/b+tree
LD_PRELOAD=../../../../tracing-tool/tracer.so ./b+tree file ../../data/b+tree/mil.txt command ../../data/b+tree/command.txt
trace_save_dir=../../../OursTracesCollection_A100/Rodinia/b+tree
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Rodinia/src/backprop
LD_PRELOAD=../../../../tracing-tool/tracer.so ./backprop 65536
trace_save_dir=../../../OursTracesCollection_A100/Rodinia/backprop
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Rodinia/src/bfs
LD_PRELOAD=../../../../tracing-tool/tracer.so ./bfs ../../data/bfs/graph1MW_6.txt
trace_save_dir=../../../OursTracesCollection_A100/Rodinia/bfs
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

# # cd $apps_root/Rodinia/src/cfd
# # LD_PRELOAD=../../../../tracing-tool/tracer.so ./euler3d ../../data/cfd/fvcorr.domn.097K
# # trace_save_dir=../../../OursTracesCollection_A100/Rodinia/cfd
# # mkdir $trace_save_dir
# # mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Rodinia/src/dwt2d
LD_PRELOAD=../../../../tracing-tool/tracer.so ./dwt2d rgb.bmp -d 1024x1024 -f -5 -l 3
trace_save_dir=../../../OursTracesCollection_A100/Rodinia/dwt2d
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Rodinia/src/gaussian
LD_PRELOAD=../../../../tracing-tool/tracer.so ./gaussian -f ../../data/gaussian/matrix1024.txt
trace_save_dir=../../../OursTracesCollection_A100/Rodinia/gaussian
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Rodinia/src/hotspot
LD_PRELOAD=../../../../tracing-tool/tracer.so ./hotspot 512 2 2 ../../data/hotspot/temp_1024 ../../data/hotspot/power_1024 output.out
trace_save_dir=../../../OursTracesCollection_A100/Rodinia/hotspot
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Rodinia/src/hotspot3D
LD_PRELOAD=../../../../tracing-tool/tracer.so ./3D 512 8 100 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.out
trace_save_dir=../../../OursTracesCollection_A100/Rodinia/hotspot3D
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Rodinia/src/huffman
LD_PRELOAD=../../../../tracing-tool/tracer.so ./pavle ../../data/huffman/test1024_H2.206587175259.in 
trace_save_dir=../../../OursTracesCollection_A100/Rodinia/huffman
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Rodinia/src/lavaMD
LD_PRELOAD=../../../../tracing-tool/tracer.so ./lavaMD -boxes1d 10
trace_save_dir=../../../OursTracesCollection_A100/Rodinia/lavaMD
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Rodinia/src/lud
LD_PRELOAD=../../../../tracing-tool/tracer.so ./lud_cuda -i ../../data/lud/2048.dat
trace_save_dir=../../../OursTracesCollection_A100/Rodinia/lud
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Rodinia/src/nn
LD_PRELOAD=../../../../tracing-tool/tracer.so ./nn filelist -r 5 -lat 30 -lng 90
trace_save_dir=../../../OursTracesCollection_A100/Rodinia/nn
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Rodinia/src/nw
LD_PRELOAD=../../../../tracing-tool/tracer.so ./needle 2048 10
trace_save_dir=../../../OursTracesCollection_A100/Rodinia/nw
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Rodinia/src/pathfinder
LD_PRELOAD=../../../../tracing-tool/tracer.so ./pathfinder 100000 100 20
trace_save_dir=../../../OursTracesCollection_A100/Rodinia/pathfinder
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir


##################################################################################
###                                                                            ###
###                                  sputnik                                   ###
###                                                                            ###
##################################################################################

# mkdir $apps_root/sputnik/build

# cd $apps_root/sputnik/build
# cmake .. --fresh -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS="89" -DCMAKE_CXX_STANDARD=14 -DABSL_PROPAGATE_CXX_STD=ON
# make -j12
# cd $apps_root

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$apps_root/sputnik/build/sputnik

# cd $apps_root/vectorSparse
# make clean 
# mkdir bin && make spmm_benchmark

# # 95% zero
# LD_PRELOAD=../../tracing-tool/tracer.so ./spmm_benchmark --A_num_rows 512 --A_num_cols 512 --A_nnz 13107 --B_num_cols 512 --vec_length=1 --kernel=2 --sorted=1 --func=0 --mixed=1
# trace_save_dir=../OursTracesCollection_A100/sputnik_spmm_csr_HF_CC/512x512x13107x512
# mkdir $trace_save_dir
# mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

# # 95% zero
# LD_PRELOAD=../../tracing-tool/tracer.so ./spmm_benchmark --A_num_rows 1024 --A_num_cols 1024 --A_nnz 52428 --B_num_cols 1024 --vec_length=1 --kernel=2 --sorted=1 --func=0 --mixed=1
# trace_save_dir=../OursTracesCollection_A100/sputnik_spmm_csr_HF_CC/1024x1024x52428x1024
# mkdir $trace_save_dir
# mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

# # 95% zero
# LD_PRELOAD=../../tracing-tool/tracer.so ./spmm_benchmark --A_num_rows 2048 --A_num_cols 2048 --A_nnz 209716 --B_num_cols 2048 --vec_length=1 --kernel=2 --sorted=1 --func=0 --mixed=1
# trace_save_dir=../OursTracesCollection_A100/sputnik_spmm_csr_HF_CC/2048x2048x209716x2048
# mkdir $trace_save_dir
# mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

# # 95% zero
# LD_PRELOAD=../../tracing-tool/tracer.so ./spmm_benchmark --A_num_rows 4096 --A_num_cols 4096 --A_nnz 838860 --B_num_cols 4096 --vec_length=1 --kernel=2 --sorted=1 --func=0 --mixed=1
# trace_save_dir=../OursTracesCollection_A100/sputnik_spmm_csr_HF_CC/4096x4096x838860x4096
# mkdir $trace_save_dir
# mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

##################################################################################
###                                                                            ###
###                                vectorSparse                                ###
###                                                                            ###
##################################################################################

# cd $apps_root/sputnik/build
# cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS="89" -DCMAKE_CXX_STANDARD=14 -DABSL_PROPAGATE_CXX_STD=ON
# make -j12
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$apps_root/sputnik/build/sputnik

# cd $apps_root/vectorSparse
# make clean && make spmm_benchmark

# # 95% zero
# LD_PRELOAD=../../tracing-tool/tracer.so ./spmm_benchmark --A_num_rows 512 --A_num_cols 512 --A_nnz 13107 --B_num_cols 512 --vec_length=8 --kernel=0 --sorted=1 --func=0 --mixed=1
# trace_save_dir=../OursTracesCollection_A100/vectorSparse_spmm_csr_HF_TC/512x512x13107x512
# mkdir $trace_save_dir
# mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

# # 95% zero
# LD_PRELOAD=../../tracing-tool/tracer.so ./spmm_benchmark --A_num_rows 1024 --A_num_cols 1024 --A_nnz 52428 --B_num_cols 1024 --vec_length=8 --kernel=0 --sorted=1 --func=0 --mixed=1
# trace_save_dir=../OursTracesCollection_A100/vectorSparse_spmm_csr_HF_TC/1024x1024x52428x1024
# mkdir $trace_save_dir
# mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

# # 95% zero
# LD_PRELOAD=../../tracing-tool/tracer.so ./spmm_benchmark --A_num_rows 2048 --A_num_cols 2048 --A_nnz 209716 --B_num_cols 2048 --vec_length=8 --kernel=0 --sorted=1 --func=0 --mixed=1
# trace_save_dir=../OursTracesCollection_A100/vectorSparse_spmm_csr_HF_TC/2048x2048x209716x2048
# mkdir $trace_save_dir
# mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

# # 95% zero
# LD_PRELOAD=../../tracing-tool/tracer.so ./spmm_benchmark --A_num_rows 4096 --A_num_cols 4096 --A_nnz 838860 --B_num_cols 4096 --vec_length=8 --kernel=0 --sorted=1 --func=0 --mixed=1
# trace_save_dir=../OursTracesCollection_A100/vectorSparse_spmm_csr_HF_TC/4096x4096x838860x4096
# mkdir $trace_save_dir
# mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

##################################################################################
###                                                                            ###
###                                 DeepBench                                  ###
###                                                                            ###
##################################################################################

cd $apps_root/DeepBench/code/nvidia
make clean && make
cd $apps_root/DeepBench/code/nvidia/bin

LD_PRELOAD=../../../../../tracing-tool/tracer.so ./conv_bench inference half 700 161 1 1 32 20 5 0 0 2 2
trace_save_dir=../../../../OursTracesCollection_A100/DeepBench/conv_bench_inference_halfx700x161x1x1x32x20x5x0x0x2x2
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

# LD_PRELOAD=../../../../../tracing-tool/tracer.so ./conv_bench train half 700 161 1 1 32 20 5 0 0 2 2
# trace_save_dir=../../../../OursTracesCollection_A100/DeepBench/conv_bench_train_halfx700x161x1x1x32x20x5x0x0x2x2
# mkdir $trace_save_dir
# mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

LD_PRELOAD=../../../../../tracing-tool/tracer.so ./gemm_bench inference half 1760 7000 1760 0 0
trace_save_dir=../../../../OursTracesCollection_A100/DeepBench/gemm_bench_inference_halfx1760x7000x1760x0x0
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

LD_PRELOAD=../../../../../tracing-tool/tracer.so ./gemm_bench train half 1760 7000 1760 0 0
trace_save_dir=../../../../OursTracesCollection_A100/DeepBench/gemm_bench_train_halfx1760x7000x1760x0x0
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

LD_PRELOAD=../../../../../tracing-tool/tracer.so ./rnn_bench inference half 1024 1 25 lstm
trace_save_dir=../../../../OursTracesCollection_A100/DeepBench/rnn_bench_inference_halfx1024x1x25xlstm
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

LD_PRELOAD=../../../../../tracing-tool/tracer.so ./rnn_bench train half 1024 1 25 lstm
trace_save_dir=../../../../OursTracesCollection_A100/DeepBench/rnn_bench_train_halfx1024x1x25xlstm
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

##################################################################################
###                                                                            ###
###                                   Tango                                    ### ERROR
###                                                                            ###
##################################################################################

cd $apps_root/Tango
make clean && make

cd $apps_root/Tango/GPU/AlexNet
LD_PRELOAD=../../../../tracing-tool/tracer.so ./AN 32
trace_save_dir=../../../OursTracesCollection_A100/Tango/AlexNet
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

# cd $apps_root/Tango/GPU/CifarNet
# LD_PRELOAD=../../../../tracing-tool/tracer.so ./CN 32
# trace_save_dir=../../../OursTracesCollection_A100/Tango/CifarNet
# mkdir $trace_save_dir
# mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Tango/GPU/GRU
LD_PRELOAD=../../../../tracing-tool/tracer.so ./GRU
trace_save_dir=../../../OursTracesCollection_A100/Tango/GRU
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Tango/GPU/LSTM
LD_PRELOAD=../../../../tracing-tool/tracer.so ./LSTM 32
trace_save_dir=../../../OursTracesCollection_A100/Tango/LSTM
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

# cd $apps_root/Tango/GPU/ResNet
# LD_PRELOAD=../../../../tracing-tool/tracer.so ./RN
# trace_save_dir=../../../OursTracesCollection_A100/Tango/ResNet
# mkdir $trace_save_dir
# mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

cd $apps_root/Tango/GPU/SqueezeNet
LD_PRELOAD=../../../../tracing-tool/tracer.so ./SN 32
trace_save_dir=../../../OursTracesCollection_A100/Tango/SqueezeNet
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

##################################################################################
###                                                                            ###
###                                   Lulesh                                   ###
###                                                                            ###
##################################################################################

cd $apps_root/LULESH/cuda/src
make clean && make

LD_PRELOAD=../../../../tracing-tool/tracer.so ./lulesh -s 45
trace_save_dir=../../../OursTracesCollection_A100/LULESH/cuda
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir


##################################################################################
###                                                                            ###
###                                   Pennant                                  ###
###                                                                            ###
##################################################################################

cd $apps_root/PENNANT
make clean && make

LD_PRELOAD=../../tracing-tool/tracer.so ./build/pennant test/sedovbig/sedovbig.pnt
trace_save_dir=../OursTracesCollection_A100/PENNANT
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

##################################################################################
###                                                                            ###
###                                  pannotia                                  ###
###                                                                            ###
##################################################################################

cd $apps_root/pannotia
export NVIDIA_COMPUTE_SDK_LOCATION=/path/to/NVIDIA_COMPUTE_SDK_LOCATION/4.2/
bash cleanall.sh && bash buildall.sh

LD_PRELOAD=../../tracing-tool/tracer.so ./bc/bc data/bc/data/2k_1M.gr
trace_save_dir=../OursTracesCollection_A100/pannotia/bc
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

LD_PRELOAD=../../tracing-tool/tracer.so ./color/color_max data/color_max/data/G3_circuit.graph 1
trace_save_dir=../OursTracesCollection_A100/pannotia/color_max
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

LD_PRELOAD=../../tracing-tool/tracer.so ./color/color_maxmin data/color_maxmin/data/G3_circuit.graph 1
trace_save_dir=../OursTracesCollection_A100/pannotia/color_maxmin
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

LD_PRELOAD=../../tracing-tool/tracer.so ./fw/fw data/fw/data/256_16384.gr
trace_save_dir=../OursTracesCollection_A100/pannotia/fw
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

LD_PRELOAD=../../tracing-tool/tracer.so ./mis/mis data/mis/data/G3_circuit.graph 1
trace_save_dir=../OursTracesCollection_A100/pannotia/mis
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

LD_PRELOAD=../../tracing-tool/tracer.so ./pagerank/pagerank data/pagerank/data/coAuthorsDBLP.graph 1
trace_save_dir=../OursTracesCollection_A100/pannotia/pagerank
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

LD_PRELOAD=../../tracing-tool/tracer.so ./pagerank/pagerank_spmv data/pagerank_spmv/data/coAuthorsDBLP.graph 1
trace_save_dir=../OursTracesCollection_A100/pannotia/pagerank_spmv
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

LD_PRELOAD=../../tracing-tool/tracer.so ./sssp/sssp data/sssp/data/USA-road-d.NY.gr 0
trace_save_dir=../OursTracesCollection_A100/pannotia/sssp
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

LD_PRELOAD=../../tracing-tool/tracer.so ./sssp/sssp_ell data/sssp_ell/data/USA-road-d.NY.gr 0
trace_save_dir=../OursTracesCollection_A100/pannotia/sssp_ell
mkdir $trace_save_dir
mv -f memory_traces/ sass_traces/ configs/ $trace_save_dir

exit_script
