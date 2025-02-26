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

# robot_list = ['sglarm_yumi', 'cbt','ur3', 'cbtpro1300']
robot_list = ['sglarm_yumi', 'cobotta','ur3']
success_array = np.zeros((len(robot_list), len(best_sol_num_list)))
time_array = np.zeros((len(robot_list), len(best_sol_num_list), nupdate))

for id, robot in enumerate(robot_list):
    time_dir = f'{robot}_time_ikseed.npy'
    success_dir = f'{robot}_success_ikseed.npy'
    time_array[id] = np.load(time_dir).squeeze()
    success_array[id] = np.load(success_dir).squeeze()



plt.figure(figsize=(12, 6))
for i in range(success_array.shape[0]):
    plt.plot(best_sol_num_list, success_array[i], label=robot_list[i], linewidth=2.5)
    plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
for i in range(time_array.shape[0]):
    mean_time = np.mean(time_array[i], axis=1)
    std_time = np.std(time_array[i], axis=1)
    plt.plot(best_sol_num_list, mean_time, label=robot_list[i], linewidth=2.5)
    plt.fill_between(best_sol_num_list, mean_time + std_time, alpha=0.1)
    plt.legend()
plt.show()
