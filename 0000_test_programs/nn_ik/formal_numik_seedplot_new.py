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

with open("seed_range_reatt.jsonl", "r", encoding="utf-8") as f:
    seed_reatt = [json.loads(line) for line in f]
with open("seed_range_woreatt.jsonl", "r", encoding="utf-8") as f:
    seed_adjust = [json.loads(line) for line in f]
with open("seed_range_kdt.jsonl", "r", encoding="utf-8") as f:
    seed_kdt = [json.loads(line) for line in f]

'''reatt'''
sr_reatt = [float(item["success_rate"].replace("%", "")) for item in seed_reatt]
yumi_sr_reatt = sr_reatt[:100]
cbt_sr_reatt = sr_reatt[100:200]
ur3_sr_reatt = sr_reatt[200:300]
cbtpro1300_sr_reatt = sr_reatt[300:400]

mean_time_reatt = [float(item["time_statistics"]['mean'].replace(" ms", "")) for item in seed_reatt]
yumi_mean_time_reatt = mean_time_reatt[:100]
cbt_mean_time_reatt = mean_time_reatt[100:200]
ur3_mean_time_reatt = mean_time_reatt[200:300]
cbtpro1300_mean_time_reatt = mean_time_reatt[300:400]

std_time_reatt = [float(item["time_statistics"]['std'].replace(" ms", "")) for item in seed_reatt]
yumi_std_time_reatt = std_time_reatt[:100]
cbt_std_time_reatt = std_time_reatt[100:200]
ur3_std_time_reatt = std_time_reatt[200:300]
cbtpro1300_std_time_reatt = std_time_reatt[300:400]

'''woreatt'''
sr_adjust = [float(item["success_rate"].replace("%", "")) for item in seed_adjust]
yumi_sr_adjust = sr_adjust[:100]
cbt_sr_adjust = sr_adjust[100:200]
ur3_sr_adjust = sr_adjust[200:300]
cbtpro1300_sr_adjust = sr_adjust[300:400]

mean_time_adjust = [float(item["time_statistics"]['mean'].replace(" ms", "")) for item in seed_adjust]
yumi_mean_time_adjust = mean_time_adjust[:100]
cbt_mean_time_adjust = mean_time_adjust[100:200]
ur3_mean_time_adjust = mean_time_adjust[200:300]
cbtpro1300_mean_time_adjust = mean_time_adjust[300:400]

std_time_adjust = [float(item["time_statistics"]['std'].replace(" ms", "")) for item in seed_adjust]
yumi_std_time_adjust = std_time_adjust[:100]
cbt_std_time_adjust = std_time_adjust[100:200]
ur3_std_time_adjust = std_time_adjust[200:300]
cbtpro1300_std_time_adjust = std_time_adjust[300:400]

'''kdt'''
sr_kdt = [float(item["success_rate"].replace("%", "")) for item in seed_kdt]
yumi_sr_kdt = sr_kdt[:100]
cbt_sr_kdt = sr_kdt[100:200]
ur3_sr_kdt = sr_kdt[200:300]
cbtpro1300_sr_kdt = sr_kdt[300:400]

mean_time_kdt = [float(item["time_statistics"]['mean'].replace(" ms", "")) for item in seed_kdt]
yumi_mean_time_kdt = mean_time_kdt[:100]
cbt_mean_time_kdt = mean_time_kdt[100:200]
ur3_mean_time_kdt = mean_time_kdt[200:300]
cbtpro1300_mean_time_kdt = mean_time_kdt[300:400]

std_time_kdt = [float(item["time_statistics"]['std'].replace(" ms", "")) for item in seed_kdt]
yumi_std_time_kdt = std_time_kdt[:100]
cbt_std_time_kdt = std_time_kdt[100:200]
ur3_std_time_kdt = std_time_kdt[200:300]
cbtpro1300_std_time_kdt = std_time_kdt[300:400]


red = (250/255, 127/255, 111/255)
yellow = (255/255, 190/255, 122/255)
blue = (130/255, 176/255, 210/255)

nupdate = 10000
best_sol_num_list = range(1,101)


colors = ["#FF9F57", "#6BAFAD", "#6498B7", "#E75A4E"]
linewidth = 3

'''success rate'''
plt.figure(figsize=(10, 8))
plt.plot(yumi_sr_reatt, color = colors[0], linewidth=linewidth, linestyle='-')
plt.plot(yumi_sr_adjust, color = colors[0], linewidth=linewidth, linestyle='--')
plt.plot(yumi_sr_kdt, color = colors[0], linewidth=linewidth, linestyle=':')
# plt.ylim(70, 100.5)
plt.xlim(-1, 100)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(cbt_sr_reatt, color = colors[1], linewidth=linewidth, linestyle='-')
plt.plot(cbt_sr_adjust, color = colors[1], linewidth=linewidth, linestyle='--')
plt.plot(cbt_sr_kdt, color = colors[1], linewidth=linewidth, linestyle=':')
# plt.ylim(70, 100.5)
plt.xlim(-1, 100)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(ur3_sr_reatt, color = colors[2], linewidth=linewidth, linestyle='-')
plt.plot(ur3_sr_adjust, color = colors[2], linewidth=linewidth, linestyle='--')
plt.plot(ur3_sr_kdt, color = colors[2], linewidth=linewidth, linestyle=':')
# plt.ylim(70, 100.5)
plt.xlim(-1, 100)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(cbtpro1300_sr_reatt, color = colors[3], linewidth=linewidth, linestyle='-')
plt.plot(cbtpro1300_sr_adjust, color = colors[3], linewidth=linewidth, linestyle='--')
plt.plot(cbtpro1300_sr_kdt, color = colors[3], linewidth=linewidth, linestyle=':')
# plt.ylim(70, 100.5)
plt.xlim(-1, 100)
plt.grid(True)
plt.show()
