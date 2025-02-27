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

red = (250/255, 127/255, 111/255)
yellow = (255/255, 190/255, 122/255)
blue = (130/255, 176/255, 210/255)

nupdate = 10000
best_sol_num_list = range(1,101)

robot_list = ['cobotta','ur3', 'cobotta_pro_1300','sglarm_yumi']
success_array = np.zeros((len(robot_list), len(best_sol_num_list)))
time_array = np.zeros((len(robot_list), len(best_sol_num_list), nupdate))

for id, robot in enumerate(robot_list):
    time_dir = f'0226_save/{robot}_time_ikseed.npy'
    success_dir = f'0226_save/{robot}_success_ikseed.npy'
    time_array[id] = np.load(time_dir).squeeze()
    success_array[id] = np.load(success_dir).squeeze()

# colors = ["#FFBE7A", "#8ECFC9", "#82B0D2", "#FA7F6F"]
colors = ["#FF9F57", "#6BAFAD", "#6498B7", "#E75A4E"]


# plt.figure(figsize=(12, 6))
# for i in range(success_array.shape[0]):
#     plt.plot(best_sol_num_list, success_array[i], label=robot_list[i],color = colors[i], linewidth=2.5)
#     # plt.legend()
#     plt.grid(True)
#     # plt.xlim(0, 100)
#     # plt.ylim(70, 100) 
#     plt.savefig('0226_save/success_seed0_100.png', dpi = 1200)

# plt.show()



plt.figure(figsize=(12, 6))
for i in range(time_array.shape[0]):
    time_array[i] = time_array[i] * 1000
    mean_time = np.mean(time_array[i], axis=1)
    std_time = np.std(time_array[i], axis=1)
    # print(std_time[99])
    plt.plot(best_sol_num_list, mean_time, label=robot_list[i],color = colors[i], linewidth=2.5)
    # plt.fill_between(best_sol_num_list, mean_time + std_time, mean_time - std_time, color = colors[i], alpha=0.2)
    plt.grid(True)
    plt.savefig('0226_save/mean_time_seed0_100.png', dpi = 1200)
    # plt.legend()
plt.show()
