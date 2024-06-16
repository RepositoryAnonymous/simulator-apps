#!/bin/bash

function exit_script() {                                                      
	exit 1                                                                    
}

apps_root=$(cd $(dirname $0); pwd)

cd $apps_root

rm -rf ./NsightCollection_A100
mkdir ./NsightCollection_A100

##################################################################################
###                                                                            ###
###                            cublas_GemmEx_HF_CC                             ###
###                                                                            ###
##################################################################################

cd $apps_root/cublas_GemmEx_HF_CC
make clean && make

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cublas_GemmEx_HF_CC_example_128x128x128 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cublas_GemmEx_HF_CC_example -m 128 -n 128 -k 128"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir


# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cublas_GemmEx_HF_CC_example_256x256x256 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cublas_GemmEx_HF_CC_example -m 256 -n 256 -k 256"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir


# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cublas_GemmEx_HF_CC_example_512x512x512 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cublas_GemmEx_HF_CC_example -m 512 -n 512 -k 512"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir


# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cublas_GemmEx_HF_CC_example_1024x1024x1024 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cublas_GemmEx_HF_CC_example -m 1024 -n 1024 -k 1024"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

ncu_cmd="/usr/local/cuda/bin/ncu --export ./cublas_GemmEx_HF_CC_example_2048x2048x2048 \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./cublas_GemmEx_HF_CC_example -m 2048 -n 2048 -k 2048"
expect ../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../expect.expect "$chmod_cmd"
nsight_save_dir=../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cublas_GemmEx_HF_CC_example_4096x4096x4096 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cublas_GemmEx_HF_CC_example -m 4096 -n 4096 -k 4096"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cublas_GemmEx_HF_CC_example_8192x8192x8192 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cublas_GemmEx_HF_CC_example -m 8192 -n 8192 -k 8192"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

##################################################################################
###                                                                            ###
###                            cublas_GemmEx_HF_TC                             ###
###                                                                            ###
##################################################################################

# cd $apps_root/cublas_GemmEx_HF_TC
# make clean && make

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cublas_GemmEx_HF_TC_example_128x128x128 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cublas_GemmEx_HF_TC_example -m 128 -n 128 -k 128"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cublas_GemmEx_HF_TC_example_256x256x256 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cublas_GemmEx_HF_TC_example -m 256 -n 256 -k 256"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cublas_GemmEx_HF_TC_example_512x512x512 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cublas_GemmEx_HF_TC_example -m 512 -n 512 -k 512"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cublas_GemmEx_HF_TC_example_1024x1024x1024 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cublas_GemmEx_HF_TC_example -m 1024 -n 1024 -k 1024"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cublas_GemmEx_HF_TC_example_2048x2048x2048 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cublas_GemmEx_HF_TC_example -m 2048 -n 2048 -k 2048"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cublas_GemmEx_HF_TC_example_4096x4096x4096 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cublas_GemmEx_HF_TC_example -m 4096 -n 4096 -k 4096"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cublas_GemmEx_HF_TC_example_8192x8192x8192 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cublas_GemmEx_HF_TC_example -m 8192 -n 8192 -k 8192"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

##################################################################################
###                                                                            ###
###                          cusparse_spmm_csr_HF_CC                           ###
###                                                                            ###
##################################################################################

# cd $apps_root/cusparse_spmm_csr_HF_CC
# make clean && make

# # 95% zero
# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cusparse_spmm_csr_HF_CC_example_512x512x13107x512 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cusparse_spmm_csr_HF_CC_example --A_num_rows 512 --A_num_cols 512 --A_nnz 13107 --B_num_cols 512"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir


# # 95% zero
# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cusparse_spmm_csr_HF_CC_example_1024x1024x52428x1024 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cusparse_spmm_csr_HF_CC_example --A_num_rows 1024 --A_num_cols 1024 --A_nnz 52428 --B_num_cols 1024"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir


# # 95% zero
# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cusparse_spmm_csr_HF_CC_example_2048x2048x209716x2048 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cusparse_spmm_csr_HF_CC_example --A_num_rows 2048 --A_num_cols 2048 --A_nnz 209716 --B_num_cols 2048"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir


# # 95% zero
# ncu_cmd="/usr/local/cuda/bin/ncu --export ./cusparse_spmm_csr_HF_CC_example_4096x4096x838860x4096 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./cusparse_spmm_csr_HF_CC_example --A_num_rows 4096 --A_num_cols 4096 --A_nnz 838860 --B_num_cols 4096"
# expect ../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir


##################################################################################
###                                                                            ###
###                                 PolyBench                                  ###
###                                                                            ###
##################################################################################

cd $apps_root/PolyBench/
make clean && make

cd $apps_root/PolyBench/CUDA/2DCONV
ncu_cmd="/usr/local/cuda/bin/ncu --export ./2DConvolution \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./2DConvolution.exe"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/PolyBench/CUDA/3DCONV
ncu_cmd="/usr/local/cuda/bin/ncu --export ./3DConvolution \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./3DConvolution.exe"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/PolyBench/CUDA/3MM
ncu_cmd="/usr/local/cuda/bin/ncu --export ./3mm \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./3mm.exe"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/PolyBench/CUDA/ATAX
ncu_cmd="/usr/local/cuda/bin/ncu --export ./atax \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./atax.exe"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/PolyBench/CUDA/BICG
ncu_cmd="/usr/local/cuda/bin/ncu --export ./bicg \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./bicg.exe"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/PolyBench/CUDA/GEMM
ncu_cmd="/usr/local/cuda/bin/ncu --export ./gemm \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./gemm.exe"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/PolyBench/CUDA/GESUMMV
ncu_cmd="/usr/local/cuda/bin/ncu --export ./gesummv \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./gesummv.exe"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/PolyBench/CUDA/GRAMSCHM
ncu_cmd="/usr/local/cuda/bin/ncu --export ./gramschmidt \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./gramschmidt.exe"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/PolyBench/CUDA/MVT
ncu_cmd="/usr/local/cuda/bin/ncu --export ./mvt \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./mvt.exe"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir



##################################################################################
###                                                                            ###
###                                  Rodinia                                   ###
###                                                                            ###
##################################################################################

cd $apps_root/Rodinia/
make clean && make

cd $apps_root/Rodinia/src/b+tree
ncu_cmd="/usr/local/cuda/bin/ncu --export ./b+tree \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./b+tree file ../../data/b+tree/mil.txt command ../../data/b+tree/command.txt"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/Rodinia/src/backprop
ncu_cmd="/usr/local/cuda/bin/ncu --export ./backprop \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./backprop 65536"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/Rodinia/src/bfs
ncu_cmd="/usr/local/cuda/bin/ncu --export ./bfs \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./bfs ../../data/bfs/graph1MW_6.txt"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/Rodinia/src/cfd
ncu_cmd="/usr/local/cuda/bin/ncu --export ./cfd \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./euler3d ../../data/cfd/fvcorr.domn.097K"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/Rodinia/src/dwt2d
ncu_cmd="/usr/local/cuda/bin/ncu --export ./dwt2d \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./dwt2d rgb.bmp -d 1024x1024 -f -5 -l 3"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/Rodinia/src/gaussian
ncu_cmd="/usr/local/cuda/bin/ncu --export ./gaussian \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./gaussian -f ../../data/gaussian/matrix1024.txt"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/Rodinia/src/hotspot
ncu_cmd="/usr/local/cuda/bin/ncu --export ./hotspot \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./hotspot 512 2 2 ../../data/hotspot/temp_1024 ../../data/hotspot/power_1024 output.out"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/Rodinia/src/hotspot3D
ncu_cmd="/usr/local/cuda/bin/ncu --export ./hotspot3D \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./3D 512 8 100 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.out"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/Rodinia/src/huffman
ncu_cmd="/usr/local/cuda/bin/ncu --export ./huffman \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./pavle ../../data/huffman/test1024_H2.206587175259.in"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/Rodinia/src/lavaMD
ncu_cmd="/usr/local/cuda/bin/ncu --export ./lavaMD \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./lavaMD -boxes1d 10"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/Rodinia/src/lud
ncu_cmd="/usr/local/cuda/bin/ncu --export ./lud \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./lud_cuda -i ../../data/lud/2048.dat"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/Rodinia/src/nn
ncu_cmd="/usr/local/cuda/bin/ncu --export ./nn \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./nn filelist -r 5 -lat 30 -lng 90"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/Rodinia/src/nw
ncu_cmd="/usr/local/cuda/bin/ncu --export ./nw \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./needle 2048 10"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


cd $apps_root/Rodinia/src/pathfinder
ncu_cmd="/usr/local/cuda/bin/ncu --export ./pathfinder \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./pathfinder 100000 100 20"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir


##################################################################################
###                                                                            ###
###                                  sputnik                                   ###
###                                                                            ###
##################################################################################

# cd $apps_root/vectorSparse
# env_cmd="export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yangjianchao/Github/sputnik/build/sputnik"

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./sputnik_spmm_csr_HF_CC_512x512x13107x512 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./spmm_benchmark --A_num_rows 512 --A_num_cols 512 --A_nnz 13107 --B_num_cols 512 --vec_length=1 --kernel=2 --sorted=1 --func=0 --mixed=1"
# expect ../expect.expect "$env_cmd && $ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./sputnik_spmm_csr_HF_CC_1024x1024x52428x1024 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./spmm_benchmark --A_num_rows 1024 --A_num_cols 1024 --A_nnz 52428 --B_num_cols 1024 --vec_length=1 --kernel=2 --sorted=1 --func=0 --mixed=1"
# expect ../expect.expect "$env_cmd && $ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./sputnik_spmm_csr_HF_CC_2048x2048x209716x2048 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./spmm_benchmark --A_num_rows 2048 --A_num_cols 2048 --A_nnz 209716 --B_num_cols 2048 --vec_length=1 --kernel=2 --sorted=1 --func=0 --mixed=1"
# expect ../expect.expect "$env_cmd && $ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./sputnik_spmm_csr_HF_CC_4096x4096x838860x4096 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./spmm_benchmark --A_num_rows 4096 --A_num_cols 4096 --A_nnz 838860 --B_num_cols 4096 --vec_length=1 --kernel=2 --sorted=1 --func=0 --mixed=1"
# expect ../expect.expect "$env_cmd && $ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir


##################################################################################
###                                                                            ###
###                                vectorSparse                                ###
###                                                                            ###
##################################################################################

# cd $apps_root/vectorSparse
# env_cmd="export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yangjianchao/Github/sputnik/build/sputnik"

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./vectorSparse_spmm_csr_HF_TC_512x512x13107x512 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./spmm_benchmark --A_num_rows 512 --A_num_cols 512 --A_nnz 13107 --B_num_cols 512 --vec_length=8 --kernel=0 --sorted=1 --func=0 --mixed=1"
# expect ../expect.expect "$env_cmd && $ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./vectorSparse_spmm_csr_HF_TC_1024x1024x52428x1024 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./spmm_benchmark --A_num_rows 1024 --A_num_cols 1024 --A_nnz 52428 --B_num_cols 1024 --vec_length=8 --kernel=0 --sorted=1 --func=0 --mixed=1"
# expect ../expect.expect "$env_cmd && $ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./vectorSparse_spmm_csr_HF_TC_2048x2048x209716x2048 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./spmm_benchmark --A_num_rows 2048 --A_num_cols 2048 --A_nnz 209716 --B_num_cols 2048 --vec_length=8 --kernel=0 --sorted=1 --func=0 --mixed=1"
# expect ../expect.expect "$env_cmd && $ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./vectorSparse_spmm_csr_HF_TC_4096x4096x838860x4096 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./spmm_benchmark --A_num_rows 4096 --A_num_cols 4096 --A_nnz 838860 --B_num_cols 4096 --vec_length=8 --kernel=0 --sorted=1 --func=0 --mixed=1"
# expect ../expect.expect "$env_cmd && $ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../expect.expect "$chmod_cmd"
# nsight_save_dir=../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

##################################################################################
###                                                                            ###
###                                 DeepBench                                  ###
###                                                                            ###
##################################################################################

cd $apps_root/DeepBench/code/nvidia
make clean && make
cd $apps_root/DeepBench/code/nvidia/bin

ncu_cmd="/usr/local/cuda/bin/ncu --export ./conv_bench_inference_halfx700x161x1x1x32x20x5x0x0x2x2 \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./conv_bench inference half 700 161 1 1 32 20 5 0 0 2 2"
expect ../../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

# ncu_cmd="/usr/local/cuda/bin/ncu --export ./conv_bench_train_halfx700x161x1x1x32x20x5x0x0x2x2 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./conv_bench train half 700 161 1 1 32 20 5 0 0 2 2"
# expect ../../../../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../../../../expect.expect "$chmod_cmd"
# nsight_save_dir=../../../../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

ncu_cmd="/usr/local/cuda/bin/ncu --export ./gemm_bench_inference_halfx1760x7000x1760x0x0 \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./gemm_bench inference half 1760 7000 1760 0 0"
expect ../../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

ncu_cmd="/usr/local/cuda/bin/ncu --export ./gemm_bench_train_halfx1760x7000x1760x0x0 \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./gemm_bench train half 1760 7000 1760 0 0"
expect ../../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

ncu_cmd="/usr/local/cuda/bin/ncu --export ./rnn_bench_inference_halfx1024x1x25xlstm \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./rnn_bench inference half 1024 1 25 lstm"
expect ../../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

ncu_cmd="/usr/local/cuda/bin/ncu --export ./rnn_bench_train_halfx1024x1x25xlstm \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./rnn_bench train half 1024 1 25 lstm"
expect ../../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

##################################################################################
###                                                                            ###
###                                   Tango                                    ###
###                                                                            ###
##################################################################################

cd $apps_root/Tango
make clean && make

cd $apps_root/Tango/GPU/AlexNet
ncu_cmd="/usr/local/cuda/bin/ncu --export ./AN_32 \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./AN 32"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

# cd $apps_root/Tango/GPU/CifarNet
# ncu_cmd="/usr/local/cuda/bin/ncu --export ./CN_32 \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./CN 32"
# expect ../../../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../../../expect.expect "$chmod_cmd"
# nsight_save_dir=../../../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

cd $apps_root/Tango/GPU/GRU
ncu_cmd="/usr/local/cuda/bin/ncu --export ./GRU \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./GRU"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

cd $apps_root/Tango/GPU/LSTM
ncu_cmd="/usr/local/cuda/bin/ncu --export ./LSTM_32 \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./LSTM 32"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

# cd $apps_root/Tango/GPU/ResNet
# ncu_cmd="/usr/local/cuda/bin/ncu --export ./RN \
#          --force-overwrite --target-processes application-only --devices 0 --set full \
# 	     ./RN"
# expect ../../../expect.expect "$ncu_cmd"
# chmod_cmd="chmod u+rw *.ncu-rep"
# expect ../../../expect.expect "$chmod_cmd"
# nsight_save_dir=../../../NsightCollection_A100
# mv -f *.ncu-rep $nsight_save_dir

cd $apps_root/Tango/GPU/SqueezeNet
ncu_cmd="/usr/local/cuda/bin/ncu --export ./SN_32 \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./SN 32"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

##################################################################################
###                                                                            ###
###                                   Lulesh                                   ###
###                                                                            ###
##################################################################################

cd $apps_root/LULESH/cuda/src
make clean && make

ncu_cmd="/usr/local/cuda/bin/ncu --export ./lulesh \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./lulesh -s 45"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

##################################################################################
###                                                                            ###
###                                   Pennant                                  ###
###                                                                            ###
##################################################################################

cd $apps_root/PENNANT
make clean && make

ncu_cmd="/usr/local/cuda/bin/ncu --export ./pennant \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./build/pennant test/sedovbig/sedovbig.pnt"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

#################################################################################
###                                                                            ###
###                                  pannotia                                  ###
###                                                                            ###
##################################################################################

cd $apps_root/pannotia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yangjianchao/Github/accel-sim-framework-dev-bak/gpu-app-collection/4.2/CUDALibraries/common/lib/
bash cleanall.sh && bash buildall.sh

ncu_cmd="/usr/local/cuda/bin/ncu --export ./bc \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./bc/bc data/bc/data/2k_1M.gr"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

ncu_cmd="/usr/local/cuda/bin/ncu --export ./color_max \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./color/color_max data/color_max/data/G3_circuit.graph 1"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

ncu_cmd="/usr/local/cuda/bin/ncu --export ./color_maxmin \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./color/color_maxmin data/color_maxmin/data/G3_circuit.graph 1"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

ncu_cmd="/usr/local/cuda/bin/ncu --export ./fw \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./fw/fw data/fw/data/256_16384.gr"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

ncu_cmd="/usr/local/cuda/bin/ncu --export ./mis \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./mis/mis data/mis/data/G3_circuit.graph 1"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

ncu_cmd="/usr/local/cuda/bin/ncu --export ./pagerank \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./pagerank/pagerank_spmv data/pagerank_spmv/data/coAuthorsDBLP.graph 1"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

ncu_cmd="/usr/local/cuda/bin/ncu --export ./pagerank_spmv \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./pagerank/pagerank_spmv data/pagerank_spmv/data/coAuthorsDBLP.graph 1"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

ncu_cmd="/usr/local/cuda/bin/ncu --export ./sssp \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./sssp/sssp data/sssp/data/USA-road-d.NY.gr 0"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

ncu_cmd="/usr/local/cuda/bin/ncu --export ./sssp_ell \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./sssp/sssp_ell data/sssp_ell/data/USA-road-d.NY.gr 0"
expect ../../../expect.expect "$ncu_cmd"
chmod_cmd="chmod u+rw *.ncu-rep"
expect ../../../expect.expect "$chmod_cmd"
nsight_save_dir=../../../NsightCollection_A100
mv -f *.ncu-rep $nsight_save_dir

exit_script
