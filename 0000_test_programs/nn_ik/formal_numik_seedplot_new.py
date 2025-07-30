from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.rs007l.rs007l as rs007l
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.manipulators.ur3e.ur3e as ur3e
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi

import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
from scipy.signal import savgol_filter
from collections import defaultdict

with open("seed_range_reatt_new_new.jsonl", "r", encoding="utf-8") as f:
    seed_reatt = [json.loads(line) for line in f]
with open("seed_range_kdt_new_new.jsonl", "r", encoding="utf-8") as f:
    seed_kdt = [json.loads(line) for line in f]

def load_jsonl_as_numpy_by_robot(file_path):
    robot_data = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                robot = entry["robot"]

                # 提取你想转成 array 的字段
                row = [
                    float(entry["success_rate"].strip('%')),
                    float(entry["t_mean"].strip(' ms')),
                    float(entry["t_std"].strip(' ms'))
                ]
                robot_data[robot].append(row)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Skipped line due to error: {e}")

    # 转为 numpy.ndarray
    for robot in robot_data:
        robot_data[robot] = np.array(robot_data[robot])

    return robot_data

reatt = load_jsonl_as_numpy_by_robot("seed_range_reatt_new_new.jsonl")
kdt = load_jsonl_as_numpy_by_robot("seed_range_kdt_new_new.jsonl")
best_sol_num_list = range(1,21)


colors = ["#FF9F57", "#6BAFAD", "#6498B7", "#E75A4E"]
linewidth = 5
markers = ['o', 's', '^', 'd']  # 图标样式依次对应 C0, C1, UR3, IRB
labels = ['C0', 'UR3', 'C1', 'IRB']
markersize = 12
keys = ['Cobotta', 'CobottaPro1300WithRobotiq140', 'UR3', 'YumiSglArm']

'''success rate'''
linestyles = ['-', '--']
for color_id, color in enumerate(colors):
    plt.figure(figsize=(10, 8))
    sr_list = [reatt[keys[color_id]][:,0], kdt[keys[color_id]][:,0]]
    marker = markers[color_id]
    label = labels[color_id]
    for id, sr in enumerate(sr_list):
        if id == 0:
            plt.plot(best_sol_num_list, sr, marker=marker, label=label, color=colors[color_id], linewidth=4, markersize=markersize)
        if id == 1:
            plt.plot(best_sol_num_list, sr, marker=marker, label=label,
                 color=colors[color_id], linewidth=4, linestyle='--',
                 markersize=markersize, markerfacecolor='white')
        # plt.grid(True)
        plt.xlim(0.5, 20.5)
        plt.xticks([1, 5, 10, 15, 20])
        plt.savefig(f'0000_test_programs/nn_ik/res_figs/0730_save/{label}_sr.png', dpi = 600, bbox_inches='tight')
    # plt.show()

'''mean time'''
# linestyles = ['-', '--', ':']
# for color_id, color in enumerate(colors):
#     plt.figure(figsize=(10, 8))
#     if color_id == 0:
#         sr_list = [cbt_mean_time_reatt, cbt_mean_time_adjust, cbt_mean_time_kdt]
#         name = 'cbt'
#         ylim = [1.5,6.5]
#     elif color_id == 1:
#         sr_list = [ur3_mean_time_reatt, ur3_mean_time_adjust, ur3_mean_time_kdt]
#         name = 'ur3'
#         ylim = [2,9]
#     elif color_id == 2:
#         sr_list = [cbtpro1300_mean_time_reatt, cbtpro1300_mean_time_adjust, cbtpro1300_mean_time_kdt]
#         name = 'cbtpro1300'
#         ylim = [2,3.5]
#     elif color_id == 3:
#         sr_list = [yumi_mean_time_reatt, yumi_mean_time_adjust, yumi_mean_time_kdt]
#         name = 'yumi'
#         ylim = [1.5,3]
    
#     for id, sr in enumerate(sr_list):
#         plt.plot(sr, color = color, linewidth=5, linestyle=linestyles[id])
#         # plt.grid(True)
#         plt.xlim(0.8, 20.2)
#         plt.ylim(ylim) 
#         plt.xticks([1, 5, 10, 15, 20])
#         # plt.savefig(f'0000_test_programs/nn_ik/res_figs/0318_save/{name}_meant.png', dpi = 600, bbox_inches='tight')
#     plt.show()

'''mean and std time'''
# for fig_id in range(3):
#     if fig_id == 0:
#         mean_list = [cbt_mean_time_reatt, ur3_mean_time_reatt, cbtpro1300_mean_time_reatt, yumi_mean_time_reatt]
#         std_list = [cbt_std_time_reatt, ur3_std_time_reatt, cbtpro1300_std_time_reatt, yumi_std_time_reatt]
#         name = 'reatt_std'
#     elif fig_id == 1:
#         mean_list = [cbt_mean_time_adjust, ur3_mean_time_adjust, cbtpro1300_mean_time_adjust, yumi_mean_time_adjust]
#         std_list = [cbt_std_time_adjust, ur3_std_time_adjust, cbtpro1300_std_time_adjust, yumi_std_time_adjust]
#         name = 'woreatt_std'
#     elif fig_id == 2:
#         mean_list = [cbt_mean_time_kdt, ur3_mean_time_kdt, cbtpro1300_mean_time_kdt, yumi_mean_time_kdt]
#         std_list = [cbt_std_time_kdt, ur3_std_time_kdt, cbtpro1300_std_time_kdt, yumi_std_time_kdt]
#         name = 'kdt_std'
    
#     plt.figure(figsize=(14, 8))
#     for id, color in enumerate(colors):
#         plt.plot(best_sol_num_list, mean_list[id], color = color, linewidth=linewidth)
#         upper = np.array(mean_list[id]) + np.array(std_list[id])
#         lower = np.array(mean_list[id]) - np.array(std_list[id])
#         plt.fill_between(best_sol_num_list, lower, upper, color = color, alpha=0.15)
#         plt.axhline(y=0, color='gray', linewidth=3, linestyle=':')
#         plt.ylim([-4, 20]) 
#         plt.xlim(0.8, 20.2)
#         plt.xticks([1, 5, 10, 15, 20])
#         plt.yticks([-4,0,4,8,12,16,20])
#     # plt.savefig(f'0000_test_programs/nn_ik/res_figs/0318_save/{name}_std_t.png', dpi = 600, bbox_inches='tight')
#     plt.show()


'''
0621 revision
- 重新绘制了成功率和平均时间的图表
回应审稿人的不清晰的图
'''
# for fig_id in range(2):
#     if fig_id == 0:
#         mean_list = [cbt_mean_time_reatt, ur3_mean_time_reatt, cbtpro1300_mean_time_reatt, yumi_mean_time_reatt]
#         std_list = [cbt_std_time_reatt, ur3_std_time_reatt, cbtpro1300_std_time_reatt, yumi_std_time_reatt]
#         name = 'reatt_std'
#     elif fig_id == 1:
#         mean_list = [cbt_mean_time_kdt, ur3_mean_time_kdt, cbtpro1300_mean_time_kdt, yumi_mean_time_kdt]
#         std_list = [cbt_std_time_kdt, ur3_std_time_kdt, cbtpro1300_std_time_kdt, yumi_std_time_kdt]
#         name = 'kdt_std'
    
#     plt.figure(figsize=(14, 8))
#     for id, color in enumerate(colors):
#         plt.plot(best_sol_num_list, mean_list[id], color = color, linewidth=5)
#         plt.plot(best_sol_num_list, std_list[id], color = color, linewidth=3, linestyle='--',alpha=0.6)
#         # plt.axhline(y=0, color='gray', linewidth=3, linestyle=':')
#         plt.ylim([0, 12]) 
#         plt.xlim(0.8, 20.2)
#         plt.xticks([1, 5, 10, 15, 20])
#         plt.yticks([0,3,6,9,12])
#         plt.grid(axis='y', linestyle='--', alpha=0.6)
#     # plt.savefig(f'0000_test_programs/nn_ik/res_figs/0621_save/{name}_std_t.png', dpi = 600, bbox_inches='tight')
#     plt.show()

cbt_mean_time_reatt = reatt['Cobotta'][:,1]
ur3_mean_time_reatt = reatt['UR3'][:,1]
cbtpro1300_mean_time_reatt = reatt['CobottaPro1300WithRobotiq140'][:,1]
yumi_mean_time_reatt = reatt['YumiSglArm'][:,1]
cbt_std_time_reatt = reatt['Cobotta'][:,2]
ur3_std_time_reatt = reatt['UR3'][:,2]
cbtpro1300_std_time_reatt = reatt['CobottaPro1300WithRobotiq140'][:,2]
yumi_std_time_reatt = reatt['YumiSglArm'][:,2]
cbt_mean_time_kdt = kdt['Cobotta'][:,1]
ur3_mean_time_kdt = kdt['UR3'][:,1]
cbtpro1300_mean_time_kdt = kdt['CobottaPro1300WithRobotiq140'][:,1]
yumi_mean_time_kdt = kdt['YumiSglArm'][:,1]
cbt_std_time_kdt = kdt['Cobotta'][:,2]
ur3_std_time_kdt = kdt['UR3'][:,2]
cbtpro1300_std_time_kdt = kdt['CobottaPro1300WithRobotiq140'][:,2]
yumi_std_time_kdt = kdt['YumiSglArm'][:,2]  
for fig_id in range(2):
    if fig_id == 0:
        mean_list = [cbt_mean_time_reatt, ur3_mean_time_reatt, cbtpro1300_mean_time_reatt, yumi_mean_time_reatt]
        std_list = [cbt_std_time_reatt, ur3_std_time_reatt, cbtpro1300_std_time_reatt, yumi_std_time_reatt]
        name = 'reatt_std'
    elif fig_id == 1:
        mean_list = [cbt_mean_time_kdt, ur3_mean_time_kdt, cbtpro1300_mean_time_kdt, yumi_mean_time_kdt]
        std_list = [cbt_std_time_kdt, ur3_std_time_kdt, cbtpro1300_std_time_kdt, yumi_std_time_kdt]
        name = 'kdt_std'
    
    plt.figure(figsize=(10, 8))
    for id, color in enumerate(colors):
        plt.plot(best_sol_num_list, mean_list[id], color = color, linewidth=5)
        plt.plot(best_sol_num_list, std_list[id], color = color, linewidth=3, linestyle='--',alpha=0.6)
        # plt.axhline(y=0, color='gray', linewidth=3, linestyle=':')
        plt.xlim(0.8, 20.2)
        plt.xticks([1, 5, 10, 15, 20])
        if fig_id == 0:
            plt.ylim([0, 3]) 
            plt.yticks([0,1,2,3])
        elif fig_id == 1:
            # plt.axhline(y=3, color='gray', linewidth=3, linestyle=':')
            plt.ylim([0, 7]) 
            plt.yticks([0,2,4,6])            
        plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(f'0000_test_programs/nn_ik/res_figs/0730_save/{name}_std_t.png', dpi = 600, bbox_inches='tight')
    # plt.show()
