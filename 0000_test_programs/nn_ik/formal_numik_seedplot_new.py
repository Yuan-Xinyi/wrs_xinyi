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

with open("seed_range_reatt_new.jsonl", "r", encoding="utf-8") as f:
    seed_reatt = [json.loads(line) for line in f]
with open("seed_range_woreatt_new.jsonl", "r", encoding="utf-8") as f:
    seed_adjust = [json.loads(line) for line in f]
with open("seed_range_kdt_new.jsonl", "r", encoding="utf-8") as f:
    seed_kdt = [json.loads(line) for line in f]

'''reatt'''
sr_reatt = [float(item["success_rate"].replace("%", "")) for item in seed_reatt]
yumi_sr_reatt = sr_reatt[:20]
cbt_sr_reatt = sr_reatt[20:40]
ur3_sr_reatt = sr_reatt[40:60]
cbtpro1300_sr_reatt = sr_reatt[60:80]

mean_time_reatt = [float(item["time_statistics"]['mean'].replace(" ms", "")) for item in seed_reatt]
yumi_mean_time_reatt = mean_time_reatt[:20]
cbt_mean_time_reatt = mean_time_reatt[20:40]
ur3_mean_time_reatt = mean_time_reatt[40:60]
cbtpro1300_mean_time_reatt = mean_time_reatt[60:80]

std_time_reatt = [float(item["time_statistics"]['std'].replace(" ms", "")) for item in seed_reatt]
yumi_std_time_reatt = std_time_reatt[:20]
cbt_std_time_reatt = std_time_reatt[20:40]
ur3_std_time_reatt = std_time_reatt[40:60]
cbtpro1300_std_time_reatt = std_time_reatt[60:80]

'''woreatt'''
sr_adjust = [float(item["success_rate"].replace("%", "")) for item in seed_adjust]
yumi_sr_adjust = sr_adjust[:20]
cbt_sr_adjust = sr_adjust[20:40]
ur3_sr_adjust = sr_adjust[40:60]
cbtpro1300_sr_adjust = sr_adjust[60:80]

mean_time_adjust = [float(item["time_statistics"]['mean'].replace(" ms", "")) for item in seed_adjust]
yumi_mean_time_adjust = mean_time_adjust[:20]
cbt_mean_time_adjust = mean_time_adjust[20:40]
ur3_mean_time_adjust = mean_time_adjust[40:60]
cbtpro1300_mean_time_adjust = mean_time_adjust[60:80]

std_time_adjust = [float(item["time_statistics"]['std'].replace(" ms", "")) for item in seed_adjust]
yumi_std_time_adjust = std_time_adjust[:20]
cbt_std_time_adjust = std_time_adjust[20:40]
ur3_std_time_adjust = std_time_adjust[40:60]
cbtpro1300_std_time_adjust = std_time_adjust[60:80]

'''kdt'''
sr_kdt = [float(item["success_rate"].replace("%", "")) for item in seed_kdt]
yumi_sr_kdt = sr_kdt[:20]
cbt_sr_kdt = sr_kdt[20:40]
ur3_sr_kdt = sr_kdt[40:60]
cbtpro1300_sr_kdt = sr_kdt[60:80]

mean_time_kdt = [float(item["time_statistics"]['mean'].replace(" ms", "")) for item in seed_kdt]
yumi_mean_time_kdt = mean_time_kdt[:20]
cbt_mean_time_kdt = mean_time_kdt[20:40]
ur3_mean_time_kdt = mean_time_kdt[40:60]
cbtpro1300_mean_time_kdt = mean_time_kdt[60:80]

std_time_kdt = [float(item["time_statistics"]['std'].replace(" ms", "")) for item in seed_kdt]
yumi_std_time_kdt = std_time_kdt[:20]
cbt_std_time_kdt = std_time_kdt[20:40]
ur3_std_time_kdt = std_time_kdt[40:60]
cbtpro1300_std_time_kdt = std_time_kdt[60:80]


red = (250/255, 127/255, 111/255)
yellow = (255/255, 190/255, 122/255)
blue = (130/255, 176/255, 210/255)

nupdate = 10000
best_sol_num_list = range(1,21)


colors = ["#FF9F57", "#6BAFAD", "#6498B7", "#E75A4E"]
linewidth = 5

'''success rate'''
linestyles = ['-', '--', ':']
for color_id, color in enumerate(colors):
    plt.figure(figsize=(10, 8))
    if color_id == 0:
        sr_list = [cbt_sr_reatt, cbt_sr_adjust, cbt_sr_kdt]
        name = 'cbt'
        seed_100 = 97.26
        ylim = 60
    elif color_id == 1:
        sr_list = [ur3_sr_reatt, ur3_sr_adjust, ur3_sr_kdt]
        name = 'ur3'
        seed_100 = 97.69
        ylim = 50
    elif color_id == 2:
        sr_list = [cbtpro1300_sr_reatt, cbtpro1300_sr_adjust, cbtpro1300_sr_kdt]
        name = 'cbtpro1300'
        seed_100 = 99.58
        ylim = 75
    elif color_id == 3:
        sr_list = [yumi_sr_reatt, yumi_sr_adjust, yumi_sr_kdt]
        name = 'yumi'
        seed_100 = 99.89
        ylim = 80
    reatt_5 = 0
    plt.axhline(y=seed_100, color='grey', linewidth=3, linestyle=':')
    for id, sr in enumerate(sr_list):
        plt.plot(best_sol_num_list, sr, color = color, linewidth=5, linestyle=linestyles[id])
        if id == 0:
            plt.axhline(y=sr[4], color=color, linewidth=3, linestyle=':')
            reatt_5 = sr[4]

        if id in [1, 2]:
            y_diff = np.abs(np.array(sr) - reatt_5)
            closest_idx = np.argmin(y_diff) 
            closest_x = best_sol_num_list[closest_idx] 
            if np.min(y_diff) < 1:
                plt.plot([closest_x, closest_x], [ylim, reatt_5], color=color, linewidth=3, linestyle=':')
        # plt.grid(True)
        plt.xlim(0.8, 20.2)
        plt.ylim(ylim, 100) 
        plt.xticks([1, 5, 10, 15, 20])
        # plt.savefig(f'0000_test_programs/nn_ik/res_figs/0318_save/{name}_sr.png', dpi = 600, bbox_inches='tight')
    plt.show()

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
for fig_id in range(3):
    if fig_id == 0:
        mean_list = [cbt_mean_time_reatt, ur3_mean_time_reatt, cbtpro1300_mean_time_reatt, yumi_mean_time_reatt]
        std_list = [cbt_std_time_reatt, ur3_std_time_reatt, cbtpro1300_std_time_reatt, yumi_std_time_reatt]
        name = 'reatt_std'
    elif fig_id == 1:
        mean_list = [cbt_mean_time_adjust, ur3_mean_time_adjust, cbtpro1300_mean_time_adjust, yumi_mean_time_adjust]
        std_list = [cbt_std_time_adjust, ur3_std_time_adjust, cbtpro1300_std_time_adjust, yumi_std_time_adjust]
        name = 'woreatt_std'
    elif fig_id == 2:
        mean_list = [cbt_mean_time_kdt, ur3_mean_time_kdt, cbtpro1300_mean_time_kdt, yumi_mean_time_kdt]
        std_list = [cbt_std_time_kdt, ur3_std_time_kdt, cbtpro1300_std_time_kdt, yumi_std_time_kdt]
        name = 'kdt_std'
    
    plt.figure(figsize=(14, 8))
    for id, color in enumerate(colors):
        plt.plot(best_sol_num_list, mean_list[id], color = color, linewidth=5)
        plt.plot(best_sol_num_list, std_list[id], color = color, linewidth=3, linestyle='--',alpha=0.6)
        # plt.axhline(y=0, color='gray', linewidth=3, linestyle=':')
        plt.ylim([0, 12]) 
        plt.xlim(0.8, 20.2)
        plt.xticks([1, 5, 10, 15, 20])
        plt.yticks([0,3,6,9,12])
        plt.grid(axis='y', linestyle='--', alpha=0.6)
    # plt.savefig(f'0000_test_programs/nn_ik/res_figs/0621_save/{name}_std_t.png', dpi = 600, bbox_inches='tight')
    plt.show()


