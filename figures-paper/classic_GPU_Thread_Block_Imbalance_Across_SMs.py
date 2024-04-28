import os
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import matplotlib.ticker as ticker

import pandas as pd

np.random.seed(19680801)

SM_NUM = 80

root_dir = "../"

apps_dir = [\
    "apps/OursTracesCollection/Rodinia/b+tree/", \
    "apps/OursTracesCollection/Rodinia/backprop/", \
    "apps/OursTracesCollection/Rodinia/bfs/", \
    "apps/OursTracesCollection/Rodinia/dwt2d/", \
    "apps/OursTracesCollection/Rodinia/gaussian/", \
    "apps/OursTracesCollection/Rodinia/hotspot3D/", \
    "apps/OursTracesCollection/Rodinia/lud/", \
    "apps/OursTracesCollection/PolyBench/TwoDCONV/", \
    "apps/OursTracesCollection/PolyBench/ThreeMM/", \
]

def get_instns_length(file_path):
    try:
        with open(file_path, 'r') as file:
            return len(file.readlines())
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return 0
    except IOError:
        print(f"Error reading file: {file_path}")
        return 0

def read_cfg_file(app_dir):
    num_blocks_per_sm = [0 for _ in range(SM_NUM)]
    num_warp_instns_per_sm = [0 for _ in range(SM_NUM)]
    
    kernel_block_ids = [[] for _ in range(SM_NUM)]
    
    issue_config_path = os.path.join(root_dir, app_dir, "configs/issue.config")
    f_issue_config = open(issue_config_path, "r")
    issue_config = f_issue_config.readlines()
    f_issue_config.close()
    
    for line in issue_config:
        if "-trace_issued_sm_id_" in line:
            sm_id = int(line.split(" ")[1].split(",")[1])
            num_blocks_per_sm[sm_id] = int(line.split(" ")[1].split(",")[0])
            kernel_block_str = line.split(",(")[1:]
            
            for pair in kernel_block_str:
                kid = int(pair.split(",")[0])
                blkid = int(pair.split(",")[1])
                kernel_block_ids[sm_id].append((kid, blkid))
    
    
    kernel_warps_per_block = {}
    
    app_config_path = os.path.join(root_dir, app_dir, "configs/app.config")
    f_app_config = open(app_config_path, "r")
    app_config = f_app_config.readlines()
    f_app_config.close()
    
    for line in app_config:
        if "tb_dim_x" in line:
            tb_dim_x = int(line.split(" ")[1])
        elif "tb_dim_y" in line:
            tb_dim_y = int(line.split(" ")[1])
        elif "tb_dim_z" in line:
            kernel_id = int(line.split("_")[1])
            tb_dim_z = int(line.split(" ")[1])
            threads_num = tb_dim_x * tb_dim_y * tb_dim_z
            kernel_warps_per_block[kernel_id] = int(threads_num / 32)
    
    
    for sm_id in range(len(kernel_block_ids)):
        for pair in kernel_block_ids[sm_id]:
            kid, blkid = pair[0], pair[1]
            gwarp_id_start = blkid * kernel_warps_per_block[kid]
            gwarp_id_end = blkid * kernel_warps_per_block[kid] + kernel_warps_per_block[kid] - 1
            for gwarp_id in range(gwarp_id_start, gwarp_id_end + 1):
                sass_file_name = "kernel_" + str(kid) + "_gwarp_id_" + str(gwarp_id) + ".split.sass"
                sass_file_path = os.path.join(root_dir, app_dir, "sass_traces", sass_file_name)
                length_instns = get_instns_length(sass_file_path)
                num_warp_instns_per_sm[sm_id] += length_instns
    
    sorted_pairs = sorted(zip(num_blocks_per_sm, num_warp_instns_per_sm))

    num_blocks_per_sm_sorted, num_warp_instns_per_sm_sorted = zip(*sorted_pairs)

    num_blocks_per_sm_sorted = list(num_blocks_per_sm_sorted)
    num_warp_instns_per_sm_sorted = list(num_warp_instns_per_sm_sorted)

    print("num_blocks_per_sm:     ", num_blocks_per_sm_sorted)
    print("num_warp_instns_per_sm:", num_warp_instns_per_sm_sorted)
    
    return num_blocks_per_sm, num_warp_instns_per_sm

data = []

    
    

def Normlize(x):
    max_value = max(x)
    normalized_data = [value / max_value for value in x]
    return normalized_data

btree_num_blocks_per_sm = [168, 168, 168, 168, 168, 168, 168, 168, 169, 170, 170, 170, 170, 170, 171, 171, 171, 171, 171, 171, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 174, 174, 174, 174, 174, 174, 174, 175, 175, 175, 175, 176, 176, 261, 261, 262, 263, 263, 263, 263, 263, 264, 264, 264, 264, 266, 266, 267, 267, 267, 268, 268, 268, 270, 270, 270, 272]
btree_num_warp_instns_per_sm = [161153, 161168, 161170, 161295, 161323, 161324, 161375, 161385, 162129, 162759, 162820, 162833, 162855, 162944, 163382, 163607, 163624, 163625, 163771, 163868, 164314, 164392, 164393, 164396, 164448, 164500, 164539, 164541, 164544, 164607, 164614, 164679, 165034, 165092, 165186, 165249, 165304, 165324, 165386, 165410, 165489, 165526, 165595, 165863, 165886, 166074, 166119, 166182, 166213, 166937, 166824, 166867, 166971, 166974, 167601, 167830, 249091, 249673, 250102, 251563, 251837, 251938, 252289, 252962, 253483, 253740, 253865, 253890, 254882, 255168, 255814, 256385, 256857, 256763, 257273, 257322, 258786, 258963, 259216, 260786]
btree_num_blocks_per_sm = Normlize(btree_num_blocks_per_sm)
btree_num_warp_instns_per_sm = Normlize(btree_num_warp_instns_per_sm)

backprop_num_blocks_per_sm = [80, 81, 81, 81, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 83, 83, 83, 83, 83, 83, 84, 84, 84, 84, 84, 84, 84, 84, 84, 85, 85, 85, 85, 85, 85, 85, 86, 86, 86, 86, 86, 87, 87, 87, 87, 88, 88, 88, 88, 88, 88, 89, 89, 91, 93, 136, 139, 139, 140, 140, 140, 142, 142, 142, 142, 143, 143, 144, 144, 144, 144, 146, 147, 147, 149, 149, 149, 149, 150]
backprop_num_warp_instns_per_sm = [56320, 56688, 57360, 57360, 58400, 58400, 58400, 58400, 58400, 58400, 58400, 58400, 58400, 58400, 58400, 58768, 58768, 59440, 59440, 59440, 59440, 59808, 59808, 60480, 60480, 60480, 60480, 60480, 60480, 60480, 59504, 60848, 60848, 61520, 61520, 61520, 61520, 61888, 62560, 62560, 62560, 62560, 59568, 61584, 62256, 62256, 61975, 62624, 63296, 63968, 63968, 64640, 61648, 65008, 64400, 64464, 93056, 96176, 96176, 97216, 97216, 97216, 97952, 99296, 99296, 99296, 97648, 100336, 98016, 98016, 99360, 101376, 100096, 99120, 100464, 101200, 101200, 101872, 103216, 102912]
backprop_num_blocks_per_sm = Normlize(backprop_num_blocks_per_sm)
backprop_num_warp_instns_per_sm = Normlize(backprop_num_warp_instns_per_sm)

bfs_num_blocks_per_sm = [490, 491, 492, 493, 493, 495, 496, 496, 496, 496, 496, 496, 496, 497, 497, 497, 497, 498, 498, 498, 498, 499, 499, 499, 499, 500, 500, 500, 501, 501, 501, 501, 501, 501, 501, 501, 502, 502, 502, 502, 502, 502, 504, 504, 504, 505, 505, 505, 506, 507, 507, 508, 509, 511, 513, 515, 773, 774, 778, 779, 780, 782, 782, 782, 783, 783, 783, 783, 784, 789, 790, 791, 791, 791, 792, 792, 794, 794, 800, 801]
bfs_num_warp_instns_per_sm = [397939, 397838, 402194, 406169, 413680, 407386, 403067, 405083, 408540, 410134, 410390, 410456, 411912, 407371, 407894, 408540, 413222, 403553, 403985, 408835, 415561, 402485, 407088, 409999, 416006, 405674, 409740, 414097, 403014, 407839, 408968, 409167, 409487, 410177, 412227, 413200, 405540, 406193, 408422, 409315, 415385, 416378, 400414, 406228, 416797, 409496, 409835, 411973, 406620, 401332, 418129, 407335, 413296, 401338, 411234, 409747, 673544, 666349, 671120, 666361, 673512, 669250, 672527, 673313, 670302, 674397, 676019, 678653, 661608, 688330, 672430, 670790, 676413, 687233, 666098, 679847, 688892, 692121, 681373, 677321]
bfs_num_blocks_per_sm = Normlize(bfs_num_blocks_per_sm)
bfs_num_warp_instns_per_sm = Normlize(bfs_num_warp_instns_per_sm)

dwt2d_num_blocks_per_sm = [44, 45, 45, 45, 46, 46, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 53, 53, 53, 54, 54, 54, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 56, 56, 56, 56, 57, 57, 58, 58, 69, 70, 71, 71, 71, 71, 71, 72, 72, 72, 73, 73, 73, 73, 74, 75, 76, 76, 76, 76, 77, 77, 78, 82]
dwt2d_num_warp_instns_per_sm = [46389, 46671, 46671, 46671, 46953, 46953, 47235, 47235, 50751, 47517, 47517, 47517, 47517, 47517, 51033, 51033, 51033, 47799, 65997, 65997, 48081, 48081, 65205, 65205, 67089, 63036, 65487, 65487, 61878, 65769, 65769, 65769, 67632, 67653, 66051, 66051, 67935, 67935, 69567, 69567, 71430, 71451, 100176, 66333, 66333, 68196, 73152, 102342, 66615, 66615, 100740, 100740, 100926, 109842, 101160, 101160, 53439, 57237, 54003, 54003, 54003, 66954, 72201, 54285, 54285, 54285, 67518, 71409, 73293, 76281, 71691, 71973, 72255, 74139, 77655, 109800, 72537, 72537, 72819, 107976]
dwt2d_num_blocks_per_sm = Normlize(dwt2d_num_blocks_per_sm)
dwt2d_num_warp_instns_per_sm = Normlize(dwt2d_num_warp_instns_per_sm)

gaussian_num_blocks_per_sm = [6244, 6265, 6267, 6270, 6278, 6293, 6298, 6299, 6303, 6316, 6317, 6319, 6321, 6326, 6327, 6330, 6334, 6336, 6336, 6342, 6344, 6346, 6347, 6348, 6350, 6351, 6352, 6355, 6364, 6365, 6371, 6381, 6395, 6402, 6404, 6411, 6428, 6429, 6434, 6436, 6441, 6443, 6445, 6453, 6457, 6466, 6476, 6489, 6494, 6504, 6519, 6521, 6535, 6544, 6551, 6560, 10595, 10642, 10644, 10663, 10674, 10675, 10677, 10692, 10723, 10736, 10739, 10745, 10750, 10752, 10768, 10778, 10782, 10788, 10809, 10822, 10858, 10872, 10876, 10908]
gaussian_num_warp_instns_per_sm = [1541612, 1544012, 1547936, 1542670, 1539276, 1539489, 1554836, 1538278, 1543566, 1550400, 1540862, 1552650, 1540412, 1562177, 1545850, 1552197, 1555186, 1540766, 1540853, 1557545, 1556055, 1561716, 1559936, 1547131, 1545461, 1543878, 1561511, 1547551, 1565254, 1544031, 1559478, 1542239, 1567316, 1548112, 1548414, 1549977, 1552104, 1553694, 1575203, 1545476, 1544388, 1570356, 1549831, 1577011, 1572177, 1548355, 1578841, 1552639, 1551517, 1589545, 1554070, 1551603, 1560263, 1558096, 1588748, 1553980, 2578582, 2582420, 2589048, 2587455, 2594215, 2580041, 2587359, 2589136, 2582661, 2580387, 2593561, 2594735, 2582712, 2607576, 2580946, 2580691, 2582353, 2582956, 2617975, 2589953, 2586309, 2590027, 2619503, 2591770]
gaussian_num_blocks_per_sm = Normlize(gaussian_num_blocks_per_sm)
gaussian_num_warp_instns_per_sm = Normlize(gaussian_num_warp_instns_per_sm)

hotspot3D_num_blocks_per_sm = [1101, 1107, 1110, 1117, 1120, 1121, 1124, 1125, 1125, 1128, 1129, 1130, 1130, 1132, 1133, 1133, 1136, 1137, 1138, 1141, 1141, 1142, 1142, 1142, 1142, 1143, 1143, 1143, 1144, 1145, 1146, 1146, 1147, 1148, 1148, 1149, 1149, 1149, 1149, 1149, 1152, 1152, 1153, 1153, 1157, 1158, 1159, 1160, 1161, 1163, 1163, 1164, 1164, 1168, 1174, 1175, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600]
hotspot3D_num_warp_instns_per_sm = [2166768, 2178576, 2184480, 2198256, 2204160, 2206128, 2212032, 2214000, 2214000, 2219904, 2221872, 2223840, 2223840, 2227776, 2229744, 2229744, 2235648, 2237616, 2239584, 2245488, 2245488, 2247456, 2247456, 2247456, 2247456, 2249424, 2249424, 2249424, 2251392, 2253360, 2255328, 2255328, 2257296, 2259264, 2259264, 2261232, 2261232, 2261232, 2261232, 2261232, 2267136, 2267136, 2269104, 2269104, 2276976, 2278944, 2280912, 2282880, 2284848, 2288784, 2288784, 2290752, 2290752, 2298624, 2310432, 2312400, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800, 3148800]
hotspot3D_num_blocks_per_sm = Normlize(hotspot3D_num_blocks_per_sm)
hotspot3D_num_warp_instns_per_sm = Normlize(hotspot3D_num_warp_instns_per_sm)

lud_num_blocks_per_sm = [7304, 7330, 7340, 7348, 7348, 7348, 7349, 7350, 7350, 7351, 7351, 7352, 7360, 7364, 7365, 7366, 7369, 7372, 7376, 7376, 7378, 7379, 7382, 7383, 7383, 7385, 7389, 7390, 7391, 7394, 7395, 7397, 7398, 7401, 7408, 7413, 7413, 7414, 7420, 7426, 7427, 7431, 7431, 7433, 7436, 7439, 7440, 7440, 7445, 7450, 7453, 7458, 7458, 7460, 7462, 7539, 11505, 11512, 11523, 11525, 11527, 11529, 11539, 11541, 11549, 11552, 11555, 11557, 11568, 11585, 11588, 11590, 11591, 11596, 11602, 11602, 11605, 11612, 11614, 11623]
lud_num_warp_instns_per_sm = [4147368, 4166000, 4173660, 4163096, 4184496, 4216596, 4170068, 4162060, 4189880, 4175452, 4205412, 4188844, 4229640, 4233988, 4200300, 4209412, 4243168, 4223424, 4174272, 4227772, 4231016, 4261528, 4318824, 4272296, 4276576, 4222040, 4249928, 4220520, 4351612, 4291208, 4313160, 4367764, 4276296, 4299352, 4348156, 4327376, 4331656, 4297968, 4322680, 4355952, 4386464, 4371552, 4375832, 4366236, 4404272, 4388808, 4367960, 4382940, 4407100, 4401300, 4422216, 4427116, 4429256, 4419660, 4414344, 4420908, 6487720, 6480884, 6499796, 6473080, 6472044, 6488128, 6521468, 6520432, 6537688, 6556464, 6543140, 6644824, 6569576, 6662420, 6619136, 6675880, 6646472, 6692032, 6622584, 6697484, 6686300, 6662344, 6714808, 6721916]
lud_num_blocks_per_sm = Normlize(lud_num_blocks_per_sm)
lud_num_warp_instns_per_sm = Normlize(lud_num_warp_instns_per_sm)

TwoDCONV_num_blocks_per_sm = [686, 687, 687, 687, 687, 687, 688, 688, 689, 689, 689, 689, 689, 689, 689, 689, 690, 690, 690, 690, 690, 690, 691, 691, 691, 691, 692, 692, 692, 692, 692, 692, 692, 692, 692, 692, 692, 693, 693, 693, 693, 693, 693, 693, 694, 694, 694, 695, 695, 695, 695, 695, 696, 696, 697, 697, 698, 1108, 1112, 1113, 1113, 1114, 1115, 1115, 1115, 1116, 1116, 1116, 1117, 1117, 1117, 1118, 1118, 1119, 1119, 1120, 1120, 1121, 1122, 1123, 1126]
TwoDCONV_num_warp_instns_per_sm = [213876, 214188, 214266, 214266, 214266, 214292, 214578, 214578, 214864, 214864, 214890, 214890, 214916, 214916, 214916, 214942, 215176, 215176, 215176, 215202, 215202, 215202, 215488, 215540, 215566, 215800, 215800, 215800, 215826, 215826, 215826, 215826, 215826, 215826, 215852, 215852, 216112, 216112, 216112, 216112, 216138, 216138, 216164, 216398, 216424, 216450, 216736, 216762, 216762, 216788, 216788, 217074, 217126, 217360, 217412, 217672, 345592, 346892, 347152, 347204, 347464, 347776, 347802, 347828, 348114, 348114, 348140, 348400, 348452, 348478, 348738, 348738, 348972, 349024, 349284, 349362, 349700, 350012, 350246, 351156]
TwoDCONV_num_blocks_per_sm = Normlize(TwoDCONV_num_blocks_per_sm)
TwoDCONV_num_warp_instns_per_sm = Normlize(TwoDCONV_num_warp_instns_per_sm)

ThreeMM_num_blocks_per_sm = [26, 26, 27, 29, 29, 29, 29, 30, 30, 30, 30, 31, 32, 32, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 40, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48]
ThreeMM_num_warp_instns_per_sm = [471120, 471120, 489240, 525480, 525480, 525480, 525480, 543600, 543600, 543600, 543600, 561720, 579840, 579840, 597960, 597960, 597960, 597960, 597960, 616080, 616080, 616080, 616080, 616080, 634200, 634200, 634200, 634200, 634200, 634200, 634200, 634200, 652320, 652320, 652320, 652320, 652320, 652320, 652320, 652320, 652320, 652320, 670440, 670440, 670440, 688560, 688560, 688560, 688560, 688560, 688560, 706680, 706680, 706680, 706680, 724800, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760, 869760]
ThreeMM_num_blocks_per_sm = Normlize(ThreeMM_num_blocks_per_sm)
ThreeMM_num_warp_instns_per_sm = Normlize(ThreeMM_num_warp_instns_per_sm)

data.append(btree_num_blocks_per_sm)
data.append(btree_num_warp_instns_per_sm)
data.append(backprop_num_blocks_per_sm)
data.append(backprop_num_warp_instns_per_sm)
data.append(bfs_num_blocks_per_sm)
data.append(bfs_num_warp_instns_per_sm)
data.append(dwt2d_num_blocks_per_sm)
data.append(dwt2d_num_warp_instns_per_sm)
data.append(gaussian_num_blocks_per_sm)
data.append(gaussian_num_warp_instns_per_sm)
data.append(hotspot3D_num_blocks_per_sm)
data.append(hotspot3D_num_warp_instns_per_sm)
data.append(lud_num_blocks_per_sm)
data.append(lud_num_warp_instns_per_sm)
data.append(TwoDCONV_num_blocks_per_sm)
data.append(TwoDCONV_num_warp_instns_per_sm)
data.append(ThreeMM_num_blocks_per_sm)
data.append(ThreeMM_num_warp_instns_per_sm)

subfig_names = ["CTAs", "Instructions", "CTAs", "Instructions", "CTAs", "Instructions", \
                "CTAs", "Instructions", "CTAs", "Instructions", "CTAs", "Instructions", \
                "CTAs", "Instructions", "CTAs", "Instructions", "CTAs", "Instructions", ]

app_names = ["b+tree", "b+tree", \
             "backprop", "backprop", \
             "bfs", "bfs", \
             "dwt2d", "dwt2d", \
             "gaussian", "gaussian", \
             "hotspot3D", "hotspot3D", \
             "lud", "lud", \
             "2DCONV", "2DCONV", \
             "3MM", "3MM", \
            ]

style_list = ['classic']

from matplotlib.lines import Line2D

for style_label in style_list:
    with plt.rc_context({"figure.max_open_warning": len(style_list)}):
        with plt.style.context(style_label):
            fig, axes = plt.subplots(3, 6, figsize=(20, 5.2), sharex=True, sharey=True, num=style_label)

            for i, ax in enumerate(axes.flat):
                ax.plot(data[i], ls='-', linewidth=2.0, color="#2d74b5")
                ax.set_ylim(0, 1.0)
                ax.set_yticks(np.arange(0, 1.1, 0.5))
                ax.set_xticks(range(0, 81, 20))
                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                
                ax.set_title(app_names[i], fontsize=20, pad=3, color="#2d74b5")
                ax.grid(True, which='major', axis='both', linestyle='--', color='gray', linewidth=2)
                if subfig_names[i] == "CTAs":
                    ax.text(30.3, 0.2, subfig_names[i], va='center', fontsize=20)
                else: 
                    ax.text(16.3, 0.2, subfig_names[i], va='center', fontsize=20)

            
            plt.subplots_adjust(bottom=0.15, hspace=0.4, left=0.06, right=0.98)
            
            gap_positions = [0.362, 0.677]

            for pos in gap_positions:
                line = Line2D([pos, pos], [0.148, 0.912], transform=fig.transFigure, linestyle='--', color='gray', linewidth=3)
                fig.add_artist(line)
            
            fig.suptitle('Sorted SM IDs', fontsize=25, y=0.07)
            fig.text(0.01, 0.5, 'Normalized Distribution', va='center', rotation='vertical', fontsize=25)
            plt.savefig('figs/'+style_label+'_blocks_balance.pdf', format='pdf')
