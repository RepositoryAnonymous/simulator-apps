#!/bin/bash

function exit_script() {                                                      
	exit 1                                                                    
}

apps_root=$(cd $(dirname $0); pwd)

cd $apps_root

TrPATH="./OursTracesCollection"
CONFIG1="../gpu-simulator.x --configs"
CONFIG2="--config_file ../DEV-Def/QV100.config"


##################################################################################
###                                                                            ###
###                            cublas_GemmEx_HF_CC                             ###
###                                                                            ###
##################################################################################

# mpirun -np 10 $CONFIG1 $TrPATH/cublas_GemmEx_HF_CC/128x128x128/configs --kernel_id 0 $CONFIG2
# python3 ../merge_report.py --dir $TrPATH/cublas_GemmEx_HF_CC/128x128x128/outputs --kernel_id 0 --np 10
# mpirun -np 10 $CONFIG1 $TrPATH/cublas_GemmEx_HF_CC/256x256x256/configs --kernel_id 0 $CONFIG2
# python3 ../merge_report.py --dir $TrPATH/cublas_GemmEx_HF_CC/256x256x256/outputs --kernel_id 0 --np 10
# mpirun -np 10 $CONFIG1 $TrPATH/cublas_GemmEx_HF_CC/512x512x512/configs --kernel_id 0 $CONFIG2
# python3 ../merge_report.py --dir $TrPATH/cublas_GemmEx_HF_CC/512x512x512/outputs --kernel_id 0 --np 10
# mpirun -np 10 $CONFIG1 $TrPATH/cublas_GemmEx_HF_CC/1024x1024x1024/configs --kernel_id 0 $CONFIG2
# python3 ../merge_report.py --dir $TrPATH/cublas_GemmEx_HF_CC/1024x1024x1024/outputs --kernel_id 0 --np 10
mpirun -np 20 $CONFIG1 $TrPATH/cublas_GemmEx_HF_CC/2048x2048x2048/configs --kernel_id 0 $CONFIG2
python3 ../merge_report.py --dir $TrPATH/cublas_GemmEx_HF_CC/2048x2048x2048/outputs --kernel_id 0 --np 20
# mpirun -np 20 $CONFIG1 $TrPATH/cublas_GemmEx_HF_CC/4096x4096x4096/configs --kernel_id 0 $CONFIG2
# mpirun -np 20 $CONFIG1 $TrPATH/cublas_GemmEx_HF_CC/8192x8192x8192/configs --kernel_id 0 $CONFIG2

##################################################################################
###                                                                            ###
###                            cublas_GemmEx_HF_TC                             ###
###                                                                            ###
##################################################################################

# mpirun -np 10 $CONFIG1 $TrPATH/cublas_GemmEx_HF_TC/128x128x128/configs --kernel_id 0 $CONFIG2
# python3 ../merge_report.py --dir $TrPATH/cublas_GemmEx_HF_TC/128x128x128/outputs --kernel_id 0 --np 10
# mpirun -np 10 $CONFIG1 $TrPATH/cublas_GemmEx_HF_TC/256x256x256/configs --kernel_id 0 $CONFIG2
# python3 ../merge_report.py --dir $TrPATH/cublas_GemmEx_HF_TC/256x256x256/outputs --kernel_id 0 --np 10
# mpirun -np 10 $CONFIG1 $TrPATH/cublas_GemmEx_HF_TC/512x512x512/configs --kernel_id 0 $CONFIG2
# python3 ../merge_report.py --dir $TrPATH/cublas_GemmEx_HF_TC/512x512x512/outputs --kernel_id 0 --np 10
# mpirun -np 10 $CONFIG1 $TrPATH/cublas_GemmEx_HF_TC/1024x1024x1024/configs --kernel_id 0 $CONFIG2
# mpirun -np 20 $CONFIG1 $TrPATH/cublas_GemmEx_HF_TC/2048x2048x2048/configs --kernel_id 0 $CONFIG2
# mpirun -np 20 $CONFIG1 $TrPATH/cublas_GemmEx_HF_TC/4096x4096x4096/configs --kernel_id 0 $CONFIG2
# mpirun -np 20 $CONFIG1 $TrPATH/cublas_GemmEx_HF_TC/8192x8192x8192/configs --kernel_id 0 $CONFIG2

##################################################################################
###                                                                            ###
###                          cusparse_spmm_csr_HF_CC                           ###
###                                                                            ###
##################################################################################

# mpirun -np 10 $CONFIG1 $TrPATH/cusparse_spmm_csr_HF_CC/512x512x13107x512/configs --kernel_id 0 $CONFIG2  # Trace has bugs
# mpirun -np 20 $CONFIG1 $TrPATH/cusparse_spmm_csr_HF_CC/1024x1024x52428x1024/configs --kernel_id 0 $CONFIG2
# mpirun -np 20 $CONFIG1 $TrPATH/cusparse_spmm_csr_HF_CC/2048x2048x209716x2048/configs --kernel_id 0 $CONFIG2
# mpirun -np 40 $CONFIG1 $TrPATH/cusparse_spmm_csr_HF_CC/4096x4096x838860x4096/configs --kernel_id 0 $CONFIG2

##################################################################################
###                                                                            ###
###                                 DeepBench                                  ###
###                                                                            ###
##################################################################################
for kernel_id in {0..10}
do
	mpirun -np 20 $CONFIG1 $TrPATH/DeepBench/conv_bench_inference_halfx700x161x1x1x32x20x5x0x0x2x2/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/DeepBench/conv_bench_inference_halfx700x161x1x1x32x20x5x0x0x2x2/outputs --kernel_id $kernel_id --np 20
done
# for kernel_id in {0..27}
# do
# 	mpirun -np 20 $CONFIG1 $TrPATH/DeepBench/conv_bench_train_halfx700x161x1x1x32x20x5x0x0x2x2/configs --kernel_id $kernel_id $CONFIG2
# done
for kernel_id in {0..7}
do
	mpirun -np 20 $CONFIG1 $TrPATH/DeepBench/gemm_bench_inference_halfx1760x7000x1760x0x0/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/DeepBench/gemm_bench_inference_halfx1760x7000x1760x0x0/outputs --kernel_id $kernel_id --np 20
done
for kernel_id in {0..7}
do
	mpirun -np 20 $CONFIG1 $TrPATH/DeepBench/gemm_bench_train_halfx1760x7000x1760x0x0/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/DeepBench/gemm_bench_train_halfx1760x7000x1760x0x0/outputs --kernel_id $kernel_id --np 20
done
for kernel_id in {0..99}
do
	mpirun -np 10 $CONFIG1 $TrPATH/DeepBench/rnn_bench_inference_halfx1024x1x25xlstm/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/DeepBench/rnn_bench_inference_halfx1024x1x25xlstm/outputs --kernel_id $kernel_id --np 10
done
for kernel_id in {0..99}
do
	mpirun -np 10 $CONFIG1 $TrPATH/DeepBench/rnn_bench_train_halfx1024x1x25xlstm/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/DeepBench/rnn_bench_train_halfx1024x1x25xlstm/outputs --kernel_id $kernel_id --np 10
done

##################################################################################
###                                                                            ###
###                                   Lulesh                                   ###
###                                                                            ###
##################################################################################

for kernel_id in {0..80}
do
	mpirun -np 10 $CONFIG1 $TrPATH/LULESH/cuda/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/LULESH/cuda/outputs --kernel_id $kernel_id --np 10
done

##################################################################################
###                                                                            ###
###                                   Pennant                                  ###
###                                                                            ###
##################################################################################

for kernel_id in {0..13}
do
	mpirun -np 10 $CONFIG1 $TrPATH/PENNANT/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/PENNANT/outputs --kernel_id $kernel_id --np 10
done

##################################################################################
###                                                                            ###
###                                 PolyBench                                  ###
###                                                                            ###
##################################################################################

mpirun -np 20 $CONFIG1 $TrPATH/PolyBench/2DCONV/configs --kernel_id 0 $CONFIG2
python3 ../merge_report.py --dir $TrPATH/PolyBench/2DCONV/outputs --kernel_id 0 --np 20
for kernel_id in {0..99}
do
	mpirun -np 10 $CONFIG1 $TrPATH/PolyBench/3DCONV/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/PolyBench/3DCONV/outputs --kernel_id $kernel_id --np 10
done
for kernel_id in {0..2}
do
	mpirun -np 20 $CONFIG1 $TrPATH/PolyBench/3MM/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/PolyBench/3MM/outputs --kernel_id $kernel_id --np 20
done
for kernel_id in {0..1}
do
	mpirun -np 20 $CONFIG1 $TrPATH/PolyBench/ATAX/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/PolyBench/ATAX/outputs --kernel_id $kernel_id --np 20
done
for kernel_id in {0..1}
do
	mpirun -np 20 $CONFIG1 $TrPATH/PolyBench/BICG/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/PolyBench/BICG/outputs --kernel_id $kernel_id --np 20
done
mpirun -np 20 $CONFIG1 $TrPATH/PolyBench/GEMM/configs --kernel_id 0 $CONFIG2
python3 ../merge_report.py --dir $TrPATH/PolyBench/GEMM/outputs --kernel_id 0 --np 20
mpirun -np 20 $CONFIG1 $TrPATH/PolyBench/GESUMMV/configs --kernel_id 0 $CONFIG2
python3 ../merge_report.py --dir $TrPATH/PolyBench/GESUMMV/outputs --kernel_id 0 --np 20
for kernel_id in {0..2}
do
	mpirun -np 20 $CONFIG1 $TrPATH/PolyBench/GRAMSCHM/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/PolyBench/GRAMSCHM/outputs --kernel_id $kernel_id --np 20
done
for kernel_id in {0..1}
do
	mpirun -np 20 $CONFIG1 $TrPATH/PolyBench/MVT/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/PolyBench/MVT/outputs --kernel_id $kernel_id --np 20
done

##################################################################################
###                                                                            ###
###                                  Rodinia                                   ###
###                                                                            ###
##################################################################################
for kernel_id in {0..1}
do
	mpirun -np 20 $CONFIG1 $TrPATH/Rodinia/b+tree/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Rodinia/b+tree/outputs --kernel_id $kernel_id --np 20
done
for kernel_id in {0..1}
do
	mpirun -np 20 $CONFIG1 $TrPATH/Rodinia/backprop/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Rodinia/backprop/outputs --kernel_id $kernel_id --np 20
done
for kernel_id in {0..23}
do
	mpirun -np 10 $CONFIG1 $TrPATH/Rodinia/bfs/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Rodinia/bfs/outputs --kernel_id $kernel_id --np 10
done
for kernel_id in {0..9}
do
	mpirun -np 20 $CONFIG1 $TrPATH/Rodinia/cfd/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Rodinia/cfd/outputs --kernel_id $kernel_id --np 20
done
for kernel_id in {0..9}
do
	mpirun -np 20 $CONFIG1 $TrPATH/Rodinia/dwt2d/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Rodinia/dwt2d/outputs --kernel_id $kernel_id --np 20
done
for kernel_id in {0..99}
do
	mpirun -np 10 $CONFIG1 $TrPATH/Rodinia/gaussian/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Rodinia/gaussian/outputs --kernel_id $kernel_id --np 10
done
mpirun -np 20 $CONFIG1 $TrPATH/Rodinia/hotspot/configs --kernel_id 0 $CONFIG2
python3 ../merge_report.py --dir $TrPATH/Rodinia/hotspot/outputs --kernel_id 0 --np 20
for kernel_id in {0..99}
do
	mpirun -np 10 $CONFIG1 $TrPATH/Rodinia/hotspot3D/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Rodinia/hotspot3D/outputs --kernel_id $kernel_id --np 10
done
for kernel_id in {0..45}
do
	mpirun -np 10 $CONFIG1 $TrPATH/Rodinia/huffman/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Rodinia/huffman/outputs --kernel_id $kernel_id --np 10
done
mpirun -np 20 $CONFIG1 $TrPATH/Rodinia/lavaMD/configs --kernel_id 0 $CONFIG2
python3 ../merge_report.py --dir $TrPATH/Rodinia/lavaMD/outputs --kernel_id 0 --np 20
for kernel_id in {0..99}
do
	mpirun -np 10 $CONFIG1 $TrPATH/Rodinia/lud/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Rodinia/lud/outputs --kernel_id $kernel_id --np 10
done
mpirun -np 20 $CONFIG1 $TrPATH/Rodinia/nn/configs --kernel_id 0 $CONFIG2
python3 ../merge_report.py --dir $TrPATH/Rodinia/nn/outputs --kernel_id 0 --np 20
for kernel_id in {0..99}
do
	mpirun -np 10 $CONFIG1 $TrPATH/Rodinia/nw/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Rodinia/nw/outputs --kernel_id $kernel_id --np 10
done
for kernel_id in {0..4}
do
	mpirun -np 20 $CONFIG1 $TrPATH/Rodinia/pathfinder/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Rodinia/pathfinder/outputs --kernel_id $kernel_id --np 20
done

##################################################################################
###                                                                            ###
###                                  sputnik                                   ###
###                                                                            ###
##################################################################################

# mpirun -np 20 $CONFIG1 $TrPATH/sputnik_spmm_csr_HF_CC/512x512x13107x512/configs --kernel_id 0 $CONFIG2
# mpirun -np 20 $CONFIG1 $TrPATH/sputnik_spmm_csr_HF_CC/1024x1024x52428x1024/configs --kernel_id 0 $CONFIG2
# mpirun -np 20 $CONFIG1 $TrPATH/sputnik_spmm_csr_HF_CC/2048x2048x209716x2048/configs --kernel_id 0 $CONFIG2
# mpirun -np 20 $CONFIG1 $TrPATH/sputnik_spmm_csr_HF_CC/4096x4096x838860x4096/configs --kernel_id 0 $CONFIG2

##################################################################################
###                                                                            ###
###                                vectorSparse                                ###
###                                                                            ###
##################################################################################

# mpirun -np 20 $CONFIG1 $TrPATH/vectorSparse_spmm_csr_HF_TC/512x512x13107x512/configs --kernel_id 0 $CONFIG2
# mpirun -np 20 $CONFIG1 $TrPATH/vectorSparse_spmm_csr_HF_TC/1024x1024x52428x1024/configs --kernel_id 0 $CONFIG2
# mpirun -np 20 $CONFIG1 $TrPATH/vectorSparse_spmm_csr_HF_TC/2048x2048x209716x2048/configs --kernel_id 0 $CONFIG2
# mpirun -np 20 $CONFIG1 $TrPATH/vectorSparse_spmm_csr_HF_TC/4096x4096x838860x4096/configs --kernel_id 0 $CONFIG2

##################################################################################
###                                                                            ###
###                                   Tango                                    ###
###                                                                            ###
##################################################################################
for kernel_id in {0..21}
do
	mpirun -np 10 $CONFIG1 $TrPATH/Tango/AlexNet/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Tango/AlexNet/outputs --kernel_id $kernel_id --np 10
done
# mpirun -np 10 $CONFIG1 $TrPATH/Tango/CifarNet/configs --kernel_id 0 $CONFIG2
for kernel_id in {0..1}
do
	mpirun -np 10 $CONFIG1 $TrPATH/Tango/GRU/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Tango/GRU/outputs --kernel_id $kernel_id --np 10
done
mpirun -np 10 $CONFIG1 $TrPATH/Tango/LSTM/configs --kernel_id 0 $CONFIG2
python3 ../merge_report.py --dir $TrPATH/Tango/LSTM/outputs --kernel_id 0 --np 10
for kernel_id in {0..99}
do
	mpirun -np 10 $CONFIG1 $TrPATH/Tango/ResNet/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Tango/ResNet/outputs --kernel_id $kernel_id --np 10
done
for kernel_id in {0..29}
do
	mpirun -np 10 $CONFIG1 $TrPATH/Tango/SqueezeNet/configs --kernel_id $kernel_id $CONFIG2
	python3 ../merge_report.py --dir $TrPATH/Tango/SqueezeNet/outputs --kernel_id $kernel_id --np 10
done
