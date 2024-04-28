
#########################################################################################################################
###                                                                                                                   ###
###                                                Workload Analysis                                                  ###
###                                                                                                                   ###
#########################################################################################################################
verbose = 0

ncu_report_file_paths = [ \
        ["./NsightCollection/b+tree.ncu-rep", \
            "./PPT-GPU-Results/Rodinia/b+tree", \
                "./Accel-Sim-Results/Rodinia/b+tree", \
                    "./OursTracesCollection/Rodinia/b+tree/outputs"], \
        ["./NsightCollection/backprop.ncu-rep", \
            "./PPT-GPU-Results/Rodinia/backprop", \
                "./Accel-Sim-Results/Rodinia/backprop", \
                    "./OursTracesCollection/Rodinia/backprop/outputs"], \
        ["./NsightCollection/bfs.ncu-rep", \
            "./PPT-GPU-Results/Rodinia/bfs", \
                "./Accel-Sim-Results/Rodinia/bfs", \
                    "./OursTracesCollection/Rodinia/bfs/outputs"], \
        ["./NsightCollection/cfd.ncu-rep", \
            "./PPT-GPU-Results/Rodinia/cfd", \
                "./Accel-Sim-Results/Rodinia/cfd", \
                    "./OursTracesCollection/Rodinia/cfd/outputs"], \
        ["./NsightCollection/dwt2d.ncu-rep", \
            "./PPT-GPU-Results/Rodinia/dwt2d", \
                "./Accel-Sim-Results/Rodinia/dwt2d", \
                    "./OursTracesCollection/Rodinia/dwt2d/outputs"], \
        ["./NsightCollection/gaussian.ncu-rep", \
            "./PPT-GPU-Results/Rodinia/gaussian", \
                "./Accel-Sim-Results/Rodinia/gaussian", \
                    "./OursTracesCollection/Rodinia/gaussian/outputs"], \
        ["./NsightCollection/hotspot.ncu-rep", \
            "./PPT-GPU-Results/Rodinia/hotspot", \
                "./Accel-Sim-Results/Rodinia/hotspot", \
                    "./OursTracesCollection/Rodinia/hotspot/outputs"], \
        ["./NsightCollection/hotspot3D.ncu-rep", \
            "./PPT-GPU-Results/Rodinia/hotspot3D", \
                "./Accel-Sim-Results/Rodinia/hotspot3D", \
                    "./OursTracesCollection/Rodinia/hotspot3D/outputs"], \
        ["./NsightCollection/huffman.ncu-rep", \
            "./PPT-GPU-Results/Rodinia/huffman", \
                "./Accel-Sim-Results/Rodinia/huffman", \
                    "./OursTracesCollection/Rodinia/huffman/outputs"], \
        ["./NsightCollection/lavaMD.ncu-rep", \
            "./PPT-GPU-Results/Rodinia/lavaMD", \
                "./Accel-Sim-Results/Rodinia/lavaMD", \
                    "./OursTracesCollection/Rodinia/lavaMD/outputs"], \
        ["./NsightCollection/lud.ncu-rep", \
            "./PPT-GPU-Results/Rodinia/lud", \
                "./Accel-Sim-Results/Rodinia/lud", \
                    "./OursTracesCollection/Rodinia/lud/outputs"], \
        ["./NsightCollection/nn.ncu-rep", \
            "./PPT-GPU-Results/Rodinia/nn", \
                "./Accel-Sim-Results/Rodinia/nn", \
                    "./OursTracesCollection/Rodinia/nn/outputs"], \
        ["./NsightCollection/nw.ncu-rep", \
            "./PPT-GPU-Results/Rodinia/nw", \
                "./Accel-Sim-Results/Rodinia/nw", \
                    "./OursTracesCollection/Rodinia/nw/outputs"], \
        ["./NsightCollection/pathfinder.ncu-rep", \
            "./PPT-GPU-Results/Rodinia/pathfinder", \
                "./Accel-Sim-Results/Rodinia/pathfinder", \
                    "./OursTracesCollection/Rodinia/pathfinder/outputs"], \
        ["./NsightCollection/2DConvolution.ncu-rep", \
            "./PPT-GPU-Results/PolyBench/2DCONV", \
                "./Accel-Sim-Results/PolyBench/2DCONV", \
                    "./OursTracesCollection/PolyBench/2DCONV/outputs"], \
        ["./NsightCollection/3DConvolution.ncu-rep", \
            "./PPT-GPU-Results/PolyBench/3DCONV", \
                "./Accel-Sim-Results/PolyBench/3DCONV", \
                    "./OursTracesCollection/PolyBench/3DCONV/outputs"], \
        ["./NsightCollection/3mm.ncu-rep", \
            "./PPT-GPU-Results/PolyBench/3MM", \
                "./Accel-Sim-Results/PolyBench/3MM", \
                    "./OursTracesCollection/PolyBench/3MM/outputs"], \
        ["./NsightCollection/atax.ncu-rep", \
            "./PPT-GPU-Results/PolyBench/ATAX", \
                "./Accel-Sim-Results/PolyBench/ATAX", \
                    "./OursTracesCollection/PolyBench/ATAX/outputs"], \
        ["./NsightCollection/bicg.ncu-rep", \
            "./PPT-GPU-Results/PolyBench/BICG", \
                "./Accel-Sim-Results/PolyBench/BICG", \
                    "./OursTracesCollection/PolyBench/BICG/outputs"], \
        ["./NsightCollection/gemm.ncu-rep", \
            "./PPT-GPU-Results/PolyBench/GEMM", \
                "./Accel-Sim-Results/PolyBench/GEMM", \
                    "./OursTracesCollection/PolyBench/GEMM/outputs"], \
        ["./NsightCollection/gesummv.ncu-rep", \
            "./PPT-GPU-Results/PolyBench/GESUMMV", \
                "./Accel-Sim-Results/PolyBench/GESUMMV", \
                    "./OursTracesCollection/PolyBench/GESUMMV/outputs"], \
        ["./NsightCollection/gramschmidt.ncu-rep", \
            "./PPT-GPU-Results/PolyBench/GRAMSCHM", \
                "./Accel-Sim-Results/PolyBench/GRAMSCHM", \
                    "./OursTracesCollection/PolyBench/GRAMSCHM/outputs"], \
        ["./NsightCollection/mvt.ncu-rep", \
            "./PPT-GPU-Results/PolyBench/MVT", \
                "./Accel-Sim-Results/PolyBench/MVT", \
                    "./OursTracesCollection/PolyBench/MVT/outputs"], \
        # ["./NsightCollection/cublas_GemmEx_HF_CC_example_128x128x128.ncu-rep", \
        #     "./PPT-GPU-Results/cublas_GemmEx_HF_CC/128x128x128", \
        #         "./Accel-Sim-Results/cublas_GemmEx_HF_CC/128x128x128", \
        #             "./OursTracesCollection/cublas_GemmEx_HF_CC/128x128x128/outputs"], \
        # ["./NsightCollection/cublas_GemmEx_HF_CC_example_256x256x256.ncu-rep", \
        #     "./PPT-GPU-Results/cublas_GemmEx_HF_CC/256x256x256", \
        #         "./Accel-Sim-Results/cublas_GemmEx_HF_CC/256x256x256", \
        #             "./OursTracesCollection/cublas_GemmEx_HF_CC/256x256x256/outputs"], \
        # ["./NsightCollection/cublas_GemmEx_HF_CC_example_512x512x512.ncu-rep", \
        #     "./PPT-GPU-Results/cublas_GemmEx_HF_CC/512x512x512", \
        #         "./Accel-Sim-Results/cublas_GemmEx_HF_CC/512x512x512", \
        #             "./OursTracesCollection/cublas_GemmEx_HF_CC/512x512x512/outputs"], \
        # ["./NsightCollection/cublas_GemmEx_HF_CC_example_1024x1024x1024.ncu-rep", \
        #     "./PPT-GPU-Results/cublas_GemmEx_HF_CC/1024x1024x1024", \
        #         "./Accel-Sim-Results/cublas_GemmEx_HF_CC/1024x1024x1024", \
        #             "./OursTracesCollection/cublas_GemmEx_HF_CC/1024x1024x1024/outputs"], \
        ["./NsightCollection/cublas_GemmEx_HF_CC_example_2048x2048x2048.ncu-rep", \
            "./PPT-GPU-Results/cublas_GemmEx_HF_CC/2048x2048x2048", \
                "./Accel-Sim-Results/cublas_GemmEx_HF_CC/2048x2048x2048", \
                    "./OursTracesCollection/cublas_GemmEx_HF_CC/2048x2048x2048/outputs"], \
        # ["./NsightCollection/cublas_GemmEx_HF_CC_example_4096x4096x4096.ncu-rep", \
        #     "./PPT-GPU-Results/cublas_GemmEx_HF_CC/4096x4096x4096", \
        #         "./Accel-Sim-Results/cublas_GemmEx_HF_CC/4096x4096x4096", \
        #             "./OursTracesCollection/cublas_GemmEx_HF_CC/4096x4096x4096/outputs"], \
        # ["./NsightCollection/cublas_GemmEx_HF_CC_example_8192x8192x8192.ncu-rep", \
        #     "./PPT-GPU-Results/cublas_GemmEx_HF_CC/8192x8192x8192", \
        #         "./Accel-Sim-Results/cublas_GemmEx_HF_CC/8192x8192x8192", \
        #             "./OursTracesCollection/cublas_GemmEx_HF_CC/8192x8192x8192/outputs"], \
        # ["./NsightCollection/cublas_GemmEx_HF_TC_example_128x128x128.ncu-rep", \
        #     "./PPT-GPU-Results/cublas_GemmEx_HF_TC/128x128x128", \
        #         "./Accel-Sim-Results/cublas_GemmEx_HF_TC/128x128x128", \
        #             "./OursTracesCollection/cublas_GemmEx_HF_TC/128x128x128/outputs"], \
        # ["./NsightCollection/cublas_GemmEx_HF_TC_example_256x256x256.ncu-rep", \
        #     "./PPT-GPU-Results/cublas_GemmEx_HF_TC/256x256x256", \
        #         "./Accel-Sim-Results/cublas_GemmEx_HF_TC/256x256x256", \
        #             "./OursTracesCollection/cublas_GemmEx_HF_TC/256x256x256/outputs"], \
        # ["./NsightCollection/cublas_GemmEx_HF_TC_example_512x512x512.ncu-rep", \
        #     "./PPT-GPU-Results/cublas_GemmEx_HF_TC/512x512x512", \
        #         "./Accel-Sim-Results/cublas_GemmEx_HF_TC/512x512x512", \
        #             "./OursTracesCollection/cublas_GemmEx_HF_TC/512x512x512/outputs"], \
        # ["./NsightCollection/cublas_GemmEx_HF_TC_example_1024x1024x1024.ncu-rep", \
        #     "./PPT-GPU-Results/cublas_GemmEx_HF_TC/1024x1024x1024", \
        #         "./Accel-Sim-Results/cublas_GemmEx_HF_TC/1024x1024x1024", \
        #             "./OursTracesCollection/cublas_GemmEx_HF_TC/1024x1024x1024/outputs"], \
        # ["./NsightCollection/cublas_GemmEx_HF_TC_example_2048x2048x2048.ncu-rep", \
        #     "./PPT-GPU-Results/cublas_GemmEx_HF_TC/2048x2048x2048", \
        #         "./Accel-Sim-Results/cublas_GemmEx_HF_TC/2048x2048x2048", \
        #             "./OursTracesCollection/cublas_GemmEx_HF_TC/2048x2048x2048/outputs"], \
        # ["./NsightCollection/cublas_GemmEx_HF_TC_example_4096x4096x4096.ncu-rep", \
        #     "./PPT-GPU-Results/cublas_GemmEx_HF_TC/4096x4096x4096", \
        #         "./Accel-Sim-Results/cublas_GemmEx_HF_TC/4096x4096x4096", \
        #             "./OursTracesCollection/cublas_GemmEx_HF_TC/4096x4096x4096/outputs"], \
        # ["./NsightCollection/cublas_GemmEx_HF_TC_example_8192x8192x8192.ncu-rep", \
        #     "./PPT-GPU-Results/cublas_GemmEx_HF_TC/8192x8192x8192", \
        #         "./Accel-Sim-Results/cublas_GemmEx_HF_TC/8192x8192x8192", \
        #             "./OursTracesCollection/cublas_GemmEx_HF_TC/8192x8192x8192/outputs"], \
        # ["./NsightCollection/cusparse_spmm_csr_HF_CC_example_512x512x13107x512.ncu-rep", \
        #     "./PPT-GPU-Results/cusparse_spmm_csr_HF_CC/512x512x13107x512", \
        #         "./Accel-Sim-Results/cusparse_spmm_csr_HF_CC/512x512x13107x512", \
        #             "./OursTracesCollection/cusparse_spmm_csr_HF_CC/512x512x13107x512/outputs"], \
        # ["./NsightCollection/cusparse_spmm_csr_HF_CC_example_1024x1024x52428x1024.ncu-rep", \
        #     "./PPT-GPU-Results/cusparse_spmm_csr_HF_CC/1024x1024x52428x1024", \
        #         "./Accel-Sim-Results/cusparse_spmm_csr_HF_CC/1024x1024x52428x1024", \
        #             "./OursTracesCollection/cusparse_spmm_csr_HF_CC/1024x1024x52428x1024/outputs"], \
        # ["./NsightCollection/cusparse_spmm_csr_HF_CC_example_2048x2048x209716x2048.ncu-rep", \
        #     "./PPT-GPU-Results/cusparse_spmm_csr_HF_CC/2048x2048x209716x2048", \
        #         "./Accel-Sim-Results/cusparse_spmm_csr_HF_CC/2048x2048x209716x2048", \
        #             "./OursTracesCollection/cusparse_spmm_csr_HF_CC/2048x2048x209716x2048/outputs"], \
        # ["./NsightCollection/cusparse_spmm_csr_HF_CC_example_4096x4096x838860x4096.ncu-rep", \
        #     "./PPT-GPU-Results/cusparse_spmm_csr_HF_CC/4096x4096x838860x4096", \
        #         "./Accel-Sim-Results/cusparse_spmm_csr_HF_CC/4096x4096x838860x4096", \
        #             "./OursTracesCollection/cusparse_spmm_csr_HF_CC/4096x4096x838860x4096/outputs"], \
        # ["./NsightCollection/sputnik_spmm_csr_HF_CC_512x512x13107x512.ncu-rep", \
        #     "./PPT-GPU-Results/sputnik_spmm_csr_HF_CC/512x512x13107x512", \
        #         "./Accel-Sim-Results/sputnik_spmm_csr_HF_CC/512x512x13107x512", \
        #             "./OursTracesCollection/sputnik_spmm_csr_HF_CC/512x512x13107x512/outputs"], \
        # ["./NsightCollection/sputnik_spmm_csr_HF_CC_1024x1024x52428x1024.ncu-rep", \
        #     "./PPT-GPU-Results/sputnik_spmm_csr_HF_CC/1024x1024x52428x1024", \
        #         "./Accel-Sim-Results/sputnik_spmm_csr_HF_CC/1024x1024x52428x1024", \
        #             "./OursTracesCollection/sputnik_spmm_csr_HF_CC/1024x1024x52428x1024/outputs"], \
        # ["./NsightCollection/sputnik_spmm_csr_HF_CC_2048x2048x209716x2048.ncu-rep", \
        #     "./PPT-GPU-Results/sputnik_spmm_csr_HF_CC/2048x2048x209716x2048", \
        #         "./Accel-Sim-Results/sputnik_spmm_csr_HF_CC/2048x2048x209716x2048", \
        #             "./OursTracesCollection/sputnik_spmm_csr_HF_CC/2048x2048x209716x2048/outputs"], \
        # ["./NsightCollection/sputnik_spmm_csr_HF_CC_4096x4096x838860x4096.ncu-rep", \
        #     "./PPT-GPU-Results/sputnik_spmm_csr_HF_CC/4096x4096x838860x4096", \
        #         "./Accel-Sim-Results/sputnik_spmm_csr_HF_CC/4096x4096x838860x4096", \
        #             "./OursTracesCollection/sputnik_spmm_csr_HF_CC/4096x4096x838860x4096/outputs"], \
        ["./NsightCollection/AN_32.ncu-rep", \
            "./PPT-GPU-Results/Tango/AlexNet", \
                "./Accel-Sim-Results/Tango/AlexNet", \
                    "./OursTracesCollection/Tango/AlexNet/outputs"], \
        ["./NsightCollection/GRU.ncu-rep", \
            "./PPT-GPU-Results/Tango/GRU", \
                "./Accel-Sim-Results/Tango/GRU", \
                    "./OursTracesCollection/Tango/GRU/outputs"], \
        ["./NsightCollection/LSTM_32.ncu-rep", \
            "./PPT-GPU-Results/Tango/LSTM", \
                "./Accel-Sim-Results/Tango/LSTM", \
                    "./OursTracesCollection/Tango/LSTM/outputs"], \
        # ["./NsightCollection/RN.ncu-rep", \
        #     "./PPT-GPU-Results/Tango/ResNet", \
        #         "./Accel-Sim-Results/Tango/ResNet", \
        #             "./OursTracesCollection/Tango/ResNet/outputs"], \
        ["./NsightCollection/SN_32.ncu-rep", \
            "./PPT-GPU-Results/Tango/SqueezeNet", \
                "./Accel-Sim-Results/Tango/SqueezeNet", \
                    "./OursTracesCollection/Tango/SqueezeNet/outputs"], \
        ["./NsightCollection/conv_bench_inference_halfx700x161x1x1x32x20x5x0x0x2x2.ncu-rep", \
            "./PPT-GPU-Results/DeepBench/conv_bench_inference_halfx700x161x1x1x32x20x5x0x0x2x2", \
                "./Accel-Sim-Results/DeepBench/conv_bench_inference_halfx700x161x1x1x32x20x5x0x0x2x2", \
                    "./OursTracesCollection/DeepBench/conv_bench_inference_halfx700x161x1x1x32x20x5x0x0x2x2/outputs"], \
        # ["./NsightCollection/conv_bench_train_halfx700x161x1x1x32x20x5x0x0x2x2.ncu-rep", \
        #     "./PPT-GPU-Results/DeepBench/conv_bench_train_halfx700x161x1x1x32x20x5x0x0x2x2", \
        #         "./Accel-Sim-Results/DeepBench/conv_bench_train_halfx700x161x1x1x32x20x5x0x0x2x2", \
        #             "./OursTracesCollection/DeepBench/conv_bench_train_halfx700x161x1x1x32x20x5x0x0x2x2/outputs"], \
        ["./NsightCollection/gemm_bench_inference_halfx1760x7000x1760x0x0.ncu-rep", \
            "./PPT-GPU-Results/DeepBench/gemm_bench_inference_halfx1760x7000x1760x0x0", \
                "./Accel-Sim-Results/DeepBench/gemm_bench_inference_halfx1760x7000x1760x0x0", \
                    "./OursTracesCollection/DeepBench/gemm_bench_inference_halfx1760x7000x1760x0x0/outputs"], \
        ["./NsightCollection/gemm_bench_train_halfx1760x7000x1760x0x0.ncu-rep", \
            "./PPT-GPU-Results/DeepBench/gemm_bench_train_halfx1760x7000x1760x0x0", \
                "./Accel-Sim-Results/DeepBench/gemm_bench_train_halfx1760x7000x1760x0x0", \
                    "./OursTracesCollection/DeepBench/gemm_bench_train_halfx1760x7000x1760x0x0/outputs"], \
        ["./NsightCollection/rnn_bench_inference_halfx1024x1x25xlstm.ncu-rep", \
            "./PPT-GPU-Results/DeepBench/rnn_bench_inference_halfx1024x1x25xlstm", \
                "./Accel-Sim-Results/DeepBench/rnn_bench_inference_halfx1024x1x25xlstm", \
                    "./OursTracesCollection/DeepBench/rnn_bench_inference_halfx1024x1x25xlstm/outputs"], \
        ["./NsightCollection/rnn_bench_train_halfx1024x1x25xlstm.ncu-rep", \
            "./PPT-GPU-Results/DeepBench/rnn_bench_train_halfx1024x1x25xlstm", \
                "./Accel-Sim-Results/DeepBench/rnn_bench_train_halfx1024x1x25xlstm", \
                    "./OursTracesCollection/DeepBench/rnn_bench_train_halfx1024x1x25xlstm/outputs"], \
        ["./NsightCollection/lulesh.ncu-rep", \
            "./PPT-GPU-Results/LULESH/cuda", \
                "./Accel-Sim-Results/LULESH/cuda", \
                    "./OursTracesCollection/LULESH/cuda/outputs"], \
        ["./NsightCollection/pennant.ncu-rep", \
            "./PPT-GPU-Results/PENNANT", \
                "./Accel-Sim-Results/PENNANT", \
                    "./OursTracesCollection/PENNANT/outputs"], \
    ]

def get_dram__bytes_sum_per_second(kernel):
    """ Get Memory Throughput.
        # of bytes accessed in DRAM (This counter metric represents the sum of the number of operations per second across 
        all sub-unit instances)
        
        dram: Device (main) memory, where the GPUs global and local memory resides.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['dram__bytes.sum.per_second'].value()
    """
    global verbose
    if verbose:
        print("Memory Throughput [Gbyte/second]: ", kernel['dram__bytes.sum.per_second'].value())
    return kernel['dram__bytes.sum.per_second'].value()

def get_gpu__compute_memory_access_throughput_avg_pct_of_peak_sustained_elapsed(kernel):
    """ Get Mem Busy.
        Compute Memory Pipeline : throughput of internal activity within caches and DRAM (This throughput metric represe-
        nts the percent of the peak sustained rate achieved during elapsed cycles across all sub-unit instances)

        gpu: The entire Graphics Processing Unit.

        DRAM: Device (main) memory, where the GPUs global and local memory resides.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed'].value()
    """
    global verbose
    if verbose:
        print("Mem Busy [%]: ", 
              kernel['gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed'].value())
    return kernel['gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed'].value()

def get_l1tex__t_sector_hit_rate_pct(kernel):
    """ Get L1/TEX Hit Rate. 
        # of sector hits per sector (This ratio metric represents the value expressed as a percentage across all sub-unit 
        instances)
       
        l1tex: The Level 1 (L1)/Texture Cache is located within the GPC.
        It can be used as directed-mapped shared memory and/or store global, local and texture data in its cache portion.
        l1tex__t refers to its Tag stage. l1tex__m refers to its Miss stage. l1tex__d refers to its Data stage.

        sector: Aligned 32 byte-chunk of memory in a cache line or device memory.
        An L1 or L2 cache line is four sectors, i.e. 128 bytes.
        Sector accesses are classified as hits if the tag is present and the sector-data is present within the cache line.
        Tag-misses and tag-hit-data-misses are all classified as misses.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['l1tex__t_sector_hit_rate.pct'].value()
    """
    global verbose
    if verbose:
        print("L1/TEX Hit Rate [%]: ", kernel['l1tex__t_sector_hit_rate.pct'].value())
    return kernel['l1tex__t_sector_hit_rate.pct'].value()

def get_L1_Total_Requests(kernel):
    """ Get lts__t_requests_srcunit_tex.sum.
        l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum
        + l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
        + l1tex__t_requests_pipe_tex_mem_surface_op_ld.sum
        + l1tex__t_requests_pipe_tex_mem_texture.sum
        + l1tex__t_requests_pipe_lsu_mem_global_op_st.sum
        + l1tex__t_requests_pipe_lsu_mem_local_op_st.sum
        + l1tex__t_requests_pipe_tex_mem_surface_op_st.sum
        + l1tex__t_requests_pipe_lsu_mem_global_op_red.sum
        + l1tex__t_requests_pipe_tex_mem_surface_op_red.sum
        + l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum
        + l1tex__t_requests_pipe_tex_mem_surface_op_atom.sum
        
        l1tex: The Level 1 (L1)/Texture Cache is located within the GPC.
        It can be used as directed-mapped shared memory and/or store global, local and texture data in its cache portion.
        l1tex__t refers to its Tag stage. l1tex__m refers to its Miss stage. l1tex__d refers to its Data stage.

        requests: A command into a HW unit to perform some action, e.g. load data from some memory location.
        Each request accesses one or more sectors.

        lsu: Load Store Unit.
        The LSU pipeline issues load, store, atomic, and reduction instructions to the L1TEX unit for global, local, and shared memory.
        It also issues special register reads (S2R), shuffles, and CTA-level arrive/wait barrier instructions to the L1TEX unit.

        local: Local memory is private storage for an executing thread and is not visible outside of that thread.
        It is intended for thread-local data like thread stacks and register spills.
        Local memory has the same latency as global memory.

        global: Global memory is a 49-bit virtual address space that is mapped to physical memory on the device, pinned system memory, or peer memory.
        Global memory is visible to all threads in the GPU.
        Global memory is accessed through the SM L1 and GPU L2.

        tex: Texture Unit.
        The SM texture pipeline forwards texture and surface instructions to the L1TEX unit's TEXIN stage.
        On GPUs where FP64 or Tensor pipelines are decoupled, the texture pipeline forwards those types of instructions, too.

        surface: Surface memory

        texture: Texture memory

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['lts__t_requests_srcunit_tex.sum'].value()
    """
    global verbose
    
    value = \
        kernel['l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum'].value() + \
        kernel['l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum'].value() + \
        kernel['l1tex__t_requests_pipe_tex_mem_surface_op_ld.sum'].value() + \
        kernel['l1tex__t_requests_pipe_tex_mem_texture.sum'].value() + \
        kernel['l1tex__t_requests_pipe_lsu_mem_global_op_st.sum'].value() + \
        kernel['l1tex__t_requests_pipe_lsu_mem_local_op_st.sum'].value() + \
        kernel['l1tex__t_requests_pipe_tex_mem_surface_op_st.sum'].value() + \
        kernel['l1tex__t_requests_pipe_lsu_mem_global_op_red.sum'].value() + \
        kernel['l1tex__t_requests_pipe_tex_mem_surface_op_red.sum'].value() + \
        kernel['l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum'].value() + \
        kernel['l1tex__t_requests_pipe_tex_mem_surface_op_atom.sum'].value()
    
    if verbose:
        print("L1 Total Requests: ", value)
    
    return value

def get_gpu__compute_memory_request_throughput_avg_pct_of_peak_sustained_elapsed(kernel):
    """ Get Max Bandwidth.
        Compute Memory Pipeline : throughput of interconnects between SM<->Caches<->DRAM (This throughput metric represents 
        the percent of the peak sustained rate achieved during elapsed cycles across all sub-unit instances)
        
        gpu: The entire Graphics Processing Unit. 
        request: A command into a HW unit to perform some action, e.g. load data from some memory location. Each request acc-
        esses one or more sectors.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed'].value()
    """
    global verbose
    if verbose:
        print("Max Bandwidth [%]: ", 
              kernel['gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed'].value())
    return kernel['gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed'].value()

def get_lts__t_sector_hit_rate_pct(kernel):
    """ Get L2 Hit Rate. 
        proportion of L2 sector lookups that hit (This ratio metric represents the value expressed as a percentage across 
        all sub-unit instances)
       
        lts: A Level 2 (L2) Cache Slice is a sub-partition of the Level 2 cache.
        lts__t refers to its Tag stage. lts__m refers to its Miss stage. lts__d refers to its Data stage.

        sector: Aligned 32 byte-chunk of memory in a cache line or device memory.
        An L1 or L2 cache line is four sectors, i.e. 128 bytes.
        Sector accesses are classified as hits if the tag is present and the sector-data is present within the cache line.
        Tag-misses and tag-hit-data-misses are all classified as misses.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['lts__t_sector_hit_rate.pct'].value()
    """
    global verbose
    if verbose:
        print("L2 Hit Rate [%]: ", kernel['lts__t_sector_hit_rate.pct'].value())
    return kernel['lts__t_sector_hit_rate.pct'].value()

def get_lts__t_requests_srcunit_tex_sum(kernel):
    """ Get lts__t_requests_srcunit_tex.sum.
        # of LTS requests from unit TEX (This counter metric represents the sum across all sub-unit instances)
        
        lts: A Level 2 (L2) Cache Slice is a sub-partition of the Level 2 cache.
        lts__t refers to its Tag stage. lts__m refers to its Miss stage. lts__d refers to its Data stage.

        requests: A command into a HW unit to perform some action, e.g. load data from some memory location.
        Each request accesses one or more sectors.

        tex: Texture Unit.
        The SM texture pipeline forwards texture and surface instructions to the L1TEX unit's TEXIN stage.
        On GPUs where FP64 or Tensor pipelines are decoupled, the texture pipeline forwards those types of instructions, too.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['lts__t_requests_srcunit_tex.sum'].value()
    """
    global verbose
    if verbose:
        print("L2 Total Requests: ", kernel['lts__t_requests_srcunit_tex.sum'].value())
    return kernel['lts__t_requests_srcunit_tex.sum'].value()

def get_sm__memory_throughput_avg_pct_of_peak_sustained_elapsed(kernel):
    """ Get Mem Pipes Busy. 
        SM memory instruction throughput assuming ideal load balancing across SMSPs (This throughput metric represents the 
        percent of the peak sustained rate achieved during elapsed cycles across all sub-unit instances)
        
        sm: The Streaming Multiprocessor handles execution of a kernel as groups of 32 threads, called warps. Warps are fu-
        rther grouped into cooperative thread arrays (CTA), called blocks in CUDA. All warps of a CTA execute on the same 
        SM. CTAs share various resources across their threads, e.g. the shared memory.

        instruction: An assembly (SASS) instruction. Each executed instruction may generate zero or more requests.

        SMSPs: Each SM is partitioned into four processing blocks, called SM sub partitions. The SM sub partitions are the 
        primary processing elements on the SM. A sub partition manages a fixed size pool of warps.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['sm__memory_throughput.avg.pct_of_peak_sustained_elapsed'].value()
    """
    global verbose
    if verbose:
        print("Mem Pipes Busy [%]: ", kernel['sm__memory_throughput.avg.pct_of_peak_sustained_elapsed'].value())
    return kernel['sm__memory_throughput.avg.pct_of_peak_sustained_elapsed'].value()

def get_l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate_pct(kernel):
    """ Get l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct. 
        # of sector hits for global loads per sector for global loads (This ratio metric represents the value expressed as 
        a percentage across all sub-unit instances)
        
        l1tex: The Level 1 (L1)/Texture Cache is located within the GPC. It can be used as directed-mapped shared memory a-
        nd/or store global, local and texture data in its cache portion. l1tex__t refers to its Tag stage. l1tex__m refers 
        to its Miss stage. l1tex__d refers to its Data stage.

        sector: Aligned 32 byte-chunk of memory in a cache line or device memory. An L1 or L2 cache line is four sectors, 
        i.e. 128 bytes. Sector accesses are classified as hits if the tag is present and the sector-data is present within 
        the cache line. Tag-misses and tag-hit-data-misses are all classified as misses.

        lsu: Load Store Unit. The LSU pipeline issues load, store, atomic, and reduction instructions to the L1TEX unit for 
        global, local, and shared memory. It also issues special register reads (S2R), shuffles, and CTA-level arrive/wait 
        barrier instructions to the L1TEX unit.

        global: Global memory is a 49-bit virtual address space that is mapped to physical memory on the device, pinned sys-
        tem memory, or peer memory. Global memory is visible to all threads in the GPU. Global memory is accessed through 
        the SM L1 and GPU L2.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct'].value()
    """
    global verbose
    if verbose:
        print("l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct [%]: ", 
              kernel['l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct'].value())
    return kernel['l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct'].value()

def get_l1tex__t_requests_pipe_lsu_mem_global_op_ld_sum(kernel):
    """ Get l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum. 
        # of requests sent to T-Stage for global loads (This counter metric represents the sum across all sub-unit instan-
        ces)
        
        l1tex: The Level 1 (L1)/Texture Cache is located within the GPC. It can be used as directed-mapped shared memory a-
        nd/or store global, local and texture data in its cache portion. l1tex__t refers to its Tag stage. l1tex__m refers 
        to its Miss stage. l1tex__d refers to its Data stage.

        sector: Aligned 32 byte-chunk of memory in a cache line or device memory. An L1 or L2 cache line is four sectors, 
        i.e. 128 bytes. Sector accesses are classified as hits if the tag is present and the sector-data is present within 
        the cache line. Tag-misses and tag-hit-data-misses are all classified as misses.

        lsu: Load Store Unit. The LSU pipeline issues load, store, atomic, and reduction instructions to the L1TEX unit for 
        global, local, and shared memory. It also issues special register reads (S2R), shuffles, and CTA-level arrive/wait 
        barrier instructions to the L1TEX unit.

        global: Global memory is a 49-bit virtual address space that is mapped to physical memory on the device, pinned sys-
        tem memory, or peer memory. Global memory is visible to all threads in the GPU. Global memory is accessed through 
        the SM L1 and GPU L2.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum'].value()
    """
    global verbose
    if verbose:
        print("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum [request]: ", 
              kernel['l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum'].value())
    return kernel['l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum'].value()

def get_l1tex__t_requests_pipe_lsu_mem_global_op_st_sum(kernel):
    """ Get l1tex__t_requests_pipe_lsu_mem_global_op_st.sum. 
        # of requests sent to T-Stage for global stores (This counter metric represents the sum across all sub-unit instan-
        ces)
        
        l1tex: The Level 1 (L1)/Texture Cache is located within the GPC. It can be used as directed-mapped shared memory a-
        nd/or store global, local and texture data in its cache portion. l1tex__t refers to its Tag stage. l1tex__m refers 
        to its Miss stage. l1tex__d refers to its Data stage.

        sector: Aligned 32 byte-chunk of memory in a cache line or device memory. An L1 or L2 cache line is four sectors, 
        i.e. 128 bytes. Sector accesses are classified as hits if the tag is present and the sector-data is present within 
        the cache line. Tag-misses and tag-hit-data-misses are all classified as misses.

        lsu: Load Store Unit. The LSU pipeline issues load, store, atomic, and reduction instructions to the L1TEX unit for 
        global, local, and shared memory. It also issues special register reads (S2R), shuffles, and CTA-level arrive/wait 
        barrier instructions to the L1TEX unit.

        global: Global memory is a 49-bit virtual address space that is mapped to physical memory on the device, pinned sys-
        tem memory, or peer memory. Global memory is visible to all threads in the GPU. Global memory is accessed through 
        the SM L1 and GPU L2.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['l1tex__t_requests_pipe_lsu_mem_global_op_st.sum'].value()
    """
    global verbose
    if verbose:
        print("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum [request]: ", 
              kernel['l1tex__t_requests_pipe_lsu_mem_global_op_st.sum'].value())
    return kernel['l1tex__t_requests_pipe_lsu_mem_global_op_st.sum'].value()

def get_l1tex__t_sectors_pipe_lsu_mem_global_op_ld_sum(kernel):
    """ Get l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum. 
        # of sectors requested for global loads (This counter metric represents the sum across all sub-unit instances)
        
        l1tex: The Level 1 (L1)/Texture Cache is located within the GPC. It can be used as directed-mapped shared memory a-
        nd/or store global, local and texture data in its cache portion. l1tex__t refers to its Tag stage. l1tex__m refers 
        to its Miss stage. l1tex__d refers to its Data stage.

        sector: Aligned 32 byte-chunk of memory in a cache line or device memory. An L1 or L2 cache line is four sectors, 
        i.e. 128 bytes. Sector accesses are classified as hits if the tag is present and the sector-data is present within 
        the cache line. Tag-misses and tag-hit-data-misses are all classified as misses.

        lsu: Load Store Unit. The LSU pipeline issues load, store, atomic, and reduction instructions to the L1TEX unit for 
        global, local, and shared memory. It also issues special register reads (S2R), shuffles, and CTA-level arrive/wait 
        barrier instructions to the L1TEX unit.

        global: Global memory is a 49-bit virtual address space that is mapped to physical memory on the device, pinned sys-
        tem memory, or peer memory. Global memory is visible to all threads in the GPU. Global memory is accessed through 
        the SM L1 and GPU L2.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'].value()
    """
    global verbose
    if verbose:
        print("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum [sector]: ", 
              kernel['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'].value())
    return kernel['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'].value()

def get_l1tex__t_sectors_pipe_lsu_mem_global_op_st_sum(kernel):
    """ Get l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum. 
        # of sectors requested for global stores (This counter metric represents the sum across all sub-unit instances)
        
        l1tex: The Level 1 (L1)/Texture Cache is located within the GPC. It can be used as directed-mapped shared memory a-
        nd/or store global, local and texture data in its cache portion. l1tex__t refers to its Tag stage. l1tex__m refers 
        to its Miss stage. l1tex__d refers to its Data stage.

        sector: Aligned 32 byte-chunk of memory in a cache line or device memory. An L1 or L2 cache line is four sectors, 
        i.e. 128 bytes. Sector accesses are classified as hits if the tag is present and the sector-data is present within 
        the cache line. Tag-misses and tag-hit-data-misses are all classified as misses.

        lsu: Load Store Unit. The LSU pipeline issues load, store, atomic, and reduction instructions to the L1TEX unit for 
        global, local, and shared memory. It also issues special register reads (S2R), shuffles, and CTA-level arrive/wait 
        barrier instructions to the L1TEX unit.

        global: Global memory is a 49-bit virtual address space that is mapped to physical memory on the device, pinned sys-
        tem memory, or peer memory. Global memory is visible to all threads in the GPU. Global memory is accessed through 
        the SM L1 and GPU L2.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum'].value()
    """
    global verbose
    if verbose:
        print("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum [sector]: ", 
              kernel['l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum'].value())
    return kernel['l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum'].value()

def get_l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld_sum(kernel):
    """ Get l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum. 
        # of sectors read from L2 into L1TEX M-Stage for local/global loads (This counter metric represents the sum across 
        all sub-unit instances)
        
        l1tex: The Level 1 (L1)/Texture Cache is located within the GPC. It can be used as directed-mapped shared memory a-
        nd/or store global, local and texture data in its cache portion. l1tex__t refers to its Tag stage. l1tex__m refers 
        to its Miss stage. l1tex__d refers to its Data stage.

        sector: Aligned 32 byte-chunk of memory in a cache line or device memory. An L1 or L2 cache line is four sectors, 
        i.e. 128 bytes. Sector accesses are classified as hits if the tag is present and the sector-data is present within 
        the cache line. Tag-misses and tag-hit-data-misses are all classified as misses.

        lg: Local/Global memory.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum'].value()
    """
    global verbose
    if verbose:
        print("l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum [sector]: ", 
              kernel['l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum'].value())
    return kernel['l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum'].value()


def get_l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st_sum(kernel):
    """ Get l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum. 
        # of sectors written to L2 for local/global stores (This counter metric represents the sum across all sub-unit ins-
        tances)
        
        l1tex: The Level 1 (L1)/Texture Cache is located within the GPC. It can be used as directed-mapped shared memory a-
        nd/or store global, local and texture data in its cache portion. l1tex__t refers to its Tag stage. l1tex__m refers 
        to its Miss stage. l1tex__d refers to its Data stage.

        sector: Aligned 32 byte-chunk of memory in a cache line or device memory. An L1 or L2 cache line is four sectors, 
        i.e. 128 bytes. Sector accesses are classified as hits if the tag is present and the sector-data is present within 
        the cache line. Tag-misses and tag-hit-data-misses are all classified as misses.

        lg: Local/Global memory.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum'].value()
    """
    global verbose
    if verbose:
        print("l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum [sector]: ", 
              kernel['l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum'].value())
    return kernel['l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum'].value()

def get_dram__sectors_read_sum(kernel):
    """ Get dram__sectors_read.sum. 
        # of sectors read from DRAM (This counter metric represents the sum across all sub-unit instances)
        
        dram: Device (main) memory, where the GPUs global and local memory resides.

        sectors: Aligned 32 byte-chunk of memory in a cache line or device memory. An L1 or L2 cache line is four sectors, 
        i.e. 128 bytes. Sector accesses are classified as hits if the tag is present and the sector-data is present within 
        the cache line. Tag-misses and tag-hit-data-misses are all classified as misses.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['dram__sectors_read.sum'].value()
    """
    global verbose
    if verbose:
        print("dram__sectors_read.sum [sector]: ", kernel['dram__sectors_read.sum'].value())
    return kernel['dram__sectors_read.sum'].value()

def get_dram__sectors_write_sum(kernel):
    """ Get dram__sectors_write.sum. 
        # of sectors read from DRAM (This counter metric represents the sum across all sub-unit instances)
        
        dram: Device (main) memory, where the GPUs global and local memory resides.

        sectors: Aligned 32 byte-chunk of memory in a cache line or device memory. An L1 or L2 cache line is four sectors, 
        i.e. 128 bytes. Sector accesses are classified as hits if the tag is present and the sector-data is present within 
        the cache line. Tag-misses and tag-hit-data-misses are all classified as misses.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['dram__sectors_write.sum'].value()
    """
    global verbose
    if verbose:
        print("dram__sectors_write.sum [sector]: ", kernel['dram__sectors_write.sum'].value())
    return kernel['dram__sectors_write.sum'].value()

def get_l1tex__t_requests_pipe_lsu_mem_global_op_atom_sum(kernel):
    """ Get l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum. 
        # of requests sent to T-Stage for global atomics (This counter metric represents the sum across all sub-unit inst-
        ances)
        
        l1tex: The Level 1 (L1)/Texture Cache is located within the GPC. It can be used as directed-mapped shared memory 
        and/or store global, local and texture data in its cache portion. l1tex__t refers to its Tag stage. l1tex__m refe-
        rs to its Miss stage. l1tex__d refers to its Data stage.

        requests: A command into a HW unit to perform some action, e.g. load data from some memory location. Each request 
        accesses one or more sectors.

        lsu: Load Store Unit. The LSU pipeline issues load, store, atomic, and reduction instructions to the L1TEX unit for 
        global, local, and shared memory. It also issues special register reads (S2R), shuffles, and CTA-level arrive/wait 
        barrier instructions to the L1TEX unit.

        global: Global memory is a 49-bit virtual address space that is mapped to physical memory on the device, pinned sys-
        tem memory, or peer memory. Global memory is visible to all threads in the GPU. Global memory is accessed through the 
        SM L1 and GPU L2.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum'].value()
    """
    global verbose
    if verbose:
        print("l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum [request]: ", 
              kernel['l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum'].value())
    return kernel['l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum'].value()

def get_l1tex__t_requests_pipe_lsu_mem_global_op_red_sum(kernel):
    """ Get l1tex__t_requests_pipe_lsu_mem_global_op_red.sum. 
        # of requests sent to T-Stage for global reductions (This counter metric represents the sum across all sub-unit in-
        stances)
        
        l1tex: The Level 1 (L1)/Texture Cache is located within the GPC. It can be used as directed-mapped shared memory 
        and/or store global, local and texture data in its cache portion. l1tex__t refers to its Tag stage. l1tex__m refe-
        rs to its Miss stage. l1tex__d refers to its Data stage.

        requests: A command into a HW unit to perform some action, e.g. load data from some memory location. Each request 
        accesses one or more sectors.

        lsu: Load Store Unit. The LSU pipeline issues load, store, atomic, and reduction instructions to the L1TEX unit for 
        global, local, and shared memory. It also issues special register reads (S2R), shuffles, and CTA-level arrive/wait 
        barrier instructions to the L1TEX unit.

        global: Global memory is a 49-bit virtual address space that is mapped to physical memory on the device, pinned sys-
        tem memory, or peer memory. Global memory is visible to all threads in the GPU. Global memory is accessed through the 
        SM L1 and GPU L2.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['l1tex__t_requests_pipe_lsu_mem_global_op_red.sum'].value()
    """
    global verbose
    if verbose:
        print("l1tex__t_requests_pipe_lsu_mem_global_op_red.sum [request]: ", 
              kernel['l1tex__t_requests_pipe_lsu_mem_global_op_red.sum'].value())
    return kernel['l1tex__t_requests_pipe_lsu_mem_global_op_red.sum'].value()

#########################################################################################################################
###                                                                                                                   ###
###                                           Compute Workload Analysis                                               ###
###                                                                                                                   ###
#########################################################################################################################

def get_gpu__time_duration_sum(kernel):
    """ Get Duration.
        equals to gpu__time_duration_measured_user if collectable, otherwise equals to gpu__time_duration_measured_wallclo-
        ck (This counter metric represents the sum across all sub-unit instances)

        gpu: The entire Graphics Processing Unit.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['gpu__time_duration.sum'].value()
    """
    global verbose
    if verbose:
        print("Duration [nsecond]: ", kernel['gpu__time_duration.sum'].value())
    return kernel['gpu__time_duration.sum'].value()


def get_gpc__cycles_elapsed_max(kernel):
    """ Get Elapsed Cycles.
        # of cycles elapsed on GPC (This counter metric represents the max across all sub-unit instances)

        gpc: The General Processing Cluster contains SM, Texture and L1 in the form of TPC(s). It is replicated several 
        times across a chip.
        
    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['gpc__cycles_elapsed.max'].value()
    """
    global verbose
    if verbose:
        print("Elapsed Cycles [cycle]: ", kernel['gpc__cycles_elapsed.max'].value())
    return kernel['gpc__cycles_elapsed.max'].value()


def get_sm__cycles_active_avg(kernel):
    """ Get SM Active Cycles.
        # of cycles with at least one warp in flight (This counter metric represents the avg across all sub-unit instan-
        ces)
        
        sm: The Streaming Multiprocessor handles execution of a kernel as groups of 32 threads, called warps. Warps are 
        further grouped into cooperative thread arrays (CTA), called blocks in CUDA. All warps of a CTA execute on the 
        same SM. CTAs share various resources across their threads, e.g. the shared memory.

        warp: A a group of 32 threads within a CTA. A warp is allocated to a sub partition and resides on the sub parti-
        tion from launch to completion.
        
    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['sm__cycles_active.avg'].value()
    """
    global verbose
    if verbose:
        print("SM Active Cycles [cycle]: ", kernel['sm__cycles_active.avg'].value())
    return kernel['sm__cycles_active.avg'].value()

def get_gpc__cycles_elapsed_avg_per_second(kernel):
    """ Get SM Frequency.
        # of cycles elapsed on GPC (This counter metric represents the avg of the number of operations per second across 
        all sub-unit instances)
        
        gpc: The General Processing Cluster contains SM, Texture and L1 in the form of TPC(s). It is replicated several 
        times across a chip.
        
    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['gpc__cycles_elapsed.avg.per_second'].value()
    """
    global verbose
    if verbose:
        print("SM Frequency [cycle/second]: ", kernel['gpc__cycles_elapsed.avg.per_second'].value())
    return kernel['gpc__cycles_elapsed.avg.per_second'].value()


def get_dram__cycles_elapsed_avg_per_second(kernel):
    """ Get DRAM Frequency.
        # of elapsed DRAM memory clock cycles (This counter metric represents the avg of the number of operations per se-
        cond across all sub-unit instances)
        
        dram: Device (main) memory, where the GPUs global and local memory resides.
        
    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['dram__cycles_elapsed.avg.per_second'].value()
    """
    global verbose
    if verbose:
        print("DRAM Frequency [cycle/second]: ", kernel['dram__cycles_elapsed.avg.per_second'].value())
    return kernel['dram__cycles_elapsed.avg.per_second'].value()


def get_sm__inst_executed_avg_per_cycle_elapsed(kernel):
    """ Get Executed Ipc Elapsed.
        # of warp instructions executed (This counter metric represents the avg of the number of operations per elapsed cy-
        cle across all sub-unit instances)
        
        sm: The Streaming Multiprocessor handles execution of a kernel as groups of 32 threads, called warps. Warps are fur-
        ther grouped into cooperative thread arrays (CTA), called blocks in CUDA. All warps of a CTA execute on the same SM.
        CTAs share various resources across their threads, e.g. the shared memory.

        warp: A a group of 32 threads within a CTA. A warp is allocated to a sub partition and resides on the sub partition 
        from launch to completion.

        instructions: An assembly (SASS) instruction. Each executed instruction may generate zero or more requests.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['sm__inst_executed.avg.per_cycle_elapsed'].value()
    """
    global verbose
    if verbose:
        print("Executed Ipc Elapsed [inst/cycle]: ", kernel['sm__inst_executed.avg.per_cycle_elapsed'].value())
    return kernel['sm__inst_executed.avg.per_cycle_elapsed'].value()

#########################################################################################################################
###                                                                                                                   ###
###                                                    Occupancy                                                      ###
###                                                                                                                   ###
#########################################################################################################################


def get_launch__occupancy_limit_blocks(kernel):
    """ Get Block Limit SM.
        Occupancy limit due to maximum number of blocks managable per SM.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['launch__occupancy_limit_blocks'].value()
    """
    global verbose
    if verbose:
        print("Block Limit SM [block]: ", kernel['launch__occupancy_limit_blocks'].value())
    return kernel['launch__occupancy_limit_blocks'].value()

def get_launch__occupancy_limit_registers(kernel):
    """ Get Block Limit Registers.
        Occupancy limit due to register usage.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['launch__occupancy_limit_registers'].value()
    """
    global verbose
    if verbose:
        print("Block Limit Registers [block]: ", kernel['launch__occupancy_limit_registers'].value())
    return kernel['launch__occupancy_limit_registers'].value()

def get_launch__occupancy_limit_shared_mem(kernel):
    """ Get Block Limit Shared Mem.
        Occupancy limit due to shared memory usage.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['launch__occupancy_limit_shared_mem'].value()
    """
    global verbose
    if verbose:
        print("Block Limit Shared Mem [block]: ", kernel['launch__occupancy_limit_shared_mem'].value())
    return kernel['launch__occupancy_limit_shared_mem'].value()

def get_launch__occupancy_limit_warps(kernel):
    """ Get Block Limit Warps.
        Occupancy limit due to block size.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['launch__occupancy_limit_warps'].value()
    """
    global verbose
    if verbose:
        print("Block Limit Warps [block]: ", kernel['launch__occupancy_limit_warps'].value())
    return kernel['launch__occupancy_limit_warps'].value()

def get_sm__maximum_warps_avg_per_active_cycle(kernel):
    """ Get Theoretical Active Warps per SM.
        
    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['sm__maximum_warps_avg_per_active_cycle'].value()
    """
    global verbose
    if verbose:
        print("Theoretical Active Warps per SM [warp]: ", kernel['sm__maximum_warps_avg_per_active_cycle'].value())
    return kernel['sm__maximum_warps_avg_per_active_cycle'].value()

def get_sm__maximum_warps_per_active_cycle_pct(kernel):
    """ Get Theoretical Occupancy.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['sm__maximum_warps_per_active_cycle_pct'].value()
    """
    global verbose
    if verbose:
        print("Theoretical Occupancy [%]: ", kernel['sm__maximum_warps_per_active_cycle_pct'].value())
    return kernel['sm__maximum_warps_per_active_cycle_pct'].value()

def get_sm__warps_active_avg_per_cycle_active(kernel):
    """ Get Achieved Active Warps Per SM.
        cumulative # of warps in flight (This counter metric represents the avg of the number of operations per unit act-
        ive cycle across all sub-unit instances)
        
        sm: The Streaming Multiprocessor handles execution of a kernel as groups of 32 threads, called warps. Warps are 
        further grouped into cooperative thread arrays (CTA), called blocks in CUDA. All warps of a CTA execute on the sa-
        me SM. CTAs share various resources across their threads, e.g. the shared memory.

        warps: A a group of 32 threads within a CTA. A warp is allocated to a sub partition and resides on the sub parti-
        tion from launch to completion.
    
    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['sm__warps_active.avg.per_cycle_active'].value()
    """
    global verbose
    if verbose:
        print("Achieved Active Warps Per SM [warp]: ", kernel['sm__warps_active.avg.per_cycle_active'].value())
    return kernel['sm__warps_active.avg.per_cycle_active'].value()

def get_sm__warps_active_avg_pct_of_peak_sustained_active(kernel):
    """ Get Achieved Occupancy.
        cumulative # of warps in flight (This counter metric represents the avg percent of the peak sustained rate achie-
        ved during unit active cycles across all sub-unit instances)
        
        sm: The Streaming Multiprocessor handles execution of a kernel as groups of 32 threads, called warps. Warps are 
        further grouped into cooperative thread arrays (CTA), called blocks in CUDA. All warps of a CTA execute on the sa-
        me SM. CTAs share various resources across their threads, e.g. the shared memory.

        warps: A a group of 32 threads within a CTA. A warp is allocated to a sub partition and resides on the sub parti-
        tion from launch to completion.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['sm__warps_active.avg.pct_of_peak_sustained_active'].value()
    """
    global verbose
    if verbose:
        print("Achieved Occupancy [%]: ", kernel['sm__warps_active.avg.pct_of_peak_sustained_active'].value())
    return kernel['sm__warps_active.avg.pct_of_peak_sustained_active'].value()


#########################################################################################################################
###                                                                                                                   ###
###                                             Instruction Statistics                                                ###
###                                                                                                                   ###
#########################################################################################################################

def get_smsp__inst_executed_sum(kernel):
    """ Get Executed Instructions.
        # of warp instructions executed (This counter metric represents the sum across all sub-unit instances)
        
        smsp: Each SM is partitioned into four processing blocks, called SM sub partitions. The SM sub partitions are the 
        primary processing elements on the SM. A sub partition manages a fixed size pool of warps.

        warp: A a group of 32 threads within a CTA. A warp is allocated to a sub partition and resides on the sub partition 
        from launch to completion.

        instructions: An assembly (SASS) instruction. Each executed instruction may generate zero or more requests.

    Args:
        kernel (_type_): _description_

    Returns:
        float: kernel['smsp__inst_executed.sum'].value()
    """
    global verbose
    if verbose:
        print("Executed Instructions [inst]: ", kernel['smsp__inst_executed.sum'].value())
    return kernel['smsp__inst_executed.sum'].value()

