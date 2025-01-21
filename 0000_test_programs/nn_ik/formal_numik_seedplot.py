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

def append_to_logfile(filename, data):
    with open(filename, "a") as f: 
        json.dump(data, f) 
        f.write("\n")

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

'''define robot'''
# robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
# robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), ik_solver='d' ,enable_cc=True)
# robot = rs007l.RS007L(pos=rm.vec(0.1, .3, .5), enable_cc=True)

nupdate = 1000

'''best sol num iteration'''
success_rates = []
best_sol_range = range(1, 51)

# filename = f"{robot.name}_ik_seed.json"

# for best_sol_num in tqdm(best_sol_range):
#     success_rate = 0
#     for i in range(nupdate):
#         jnt_values = robot.rand_conf()
#         tgt_pos, tgt_rotmat = robot.fk(jnt_values=jnt_values)
#         result = robot.ik(tgt_pos, tgt_rotmat, best_sol_num=best_sol_num)
#         if result is not None:
#             success_rate += 1
    
#     # Calculate success rate for this best_sol_num
#     success_rates.append(success_rate / nupdate)

# data = {"pinv_cw_sort": success_rates}
# append_to_logfile(filename, data)  


# # yumi
# pinv = [0.638, 0.811, 0.896, 0.933, 0.957, 0.95, 0.963, 0.974, 0.972, 0.988, 0.982, 0.991, 0.983, 0.987, 0.988, 0.984, 0.991, 0.991, 0.986, 0.994, 0.991, 0.989, 0.992, 0.992, 0.996, 0.991, 0.997, 0.996, 0.995, 0.998, 0.993, 0.99, 0.989, 0.989, 0.995, 0.995, 0.998, 0.997, 0.997, 0.998, 0.993, 0.998, 0.994, 0.993, 0.995, 0.996, 0.997, 0.989, 0.993, 0.996]
# pinv_cw = [0.704, 0.852, 0.915, 0.94, 0.957, 0.977, 0.981, 0.985, 0.99, 0.991, 0.983, 0.991, 0.993, 0.994, 0.993, 0.996, 0.993, 0.995, 0.993, 0.997, 0.991, 0.992, 0.995, 0.997, 0.996, 0.993, 0.993, 0.997, 0.993, 0.994, 0.997, 0.996, 0.997, 0.996, 0.997, 0.998, 0.992, 0.997, 0.994, 0.997, 0.998, 0.998, 0.994, 0.997, 0.994, 0.996, 0.997, 0.998, 0.997, 0.995]
# pinv_cw_sort = [0.885, 0.965, 0.979, 0.991, 0.989, 0.99, 0.993, 0.991, 0.996, 0.996, 0.987, 0.998, 0.993, 0.998, 0.998, 0.999, 0.99, 0.997, 0.994, 0.995, 0.996, 0.991, 0.997, 0.995, 0.996, 1.0, 0.999, 0.999, 0.998, 0.995, 0.996, 0.998, 0.998, 0.998, 0.997, 0.996, 0.998, 0.999, 1.0, 0.997, 0.999, 0.998, 0.995, 1.0, 0.998, 0.999, 0.999, 0.997, 0.998, 1.0]

# red = (250/255, 127/255, 111/255)
# yellow = (255/255, 190/255, 122/255)
# blue = (130/255, 176/255, 210/255)

# plt.figure(figsize=(16, 8))
# plt.plot(best_sol_range, pinv, label='pinv', color=yellow, linewidth=4.0)
# plt.plot(best_sol_range, pinv_cw, label='pinv_cw', color=blue, linewidth=4.0)
# plt.plot(best_sol_range, pinv_cw_sort, label='pinv_cw_sort', color=red, linewidth=4.0)

# plt.xlabel('Best Solution Number (best_sol_num)')
# plt.ylabel('Success Rate')
# plt.xlim(-0.2, 50.3) 
# plt.ylim(0.4, 1.01)  
# ax = plt.gca()
# ax.spines['top'].set_linewidth(2)   
# ax.spines['bottom'].set_linewidth(2) 
# ax.spines['left'].set_linewidth(2)  
# ax.spines['right'].set_linewidth(2)
# # plt.legend()
# plt.savefig("yumi_seed_plot", dpi=600, bbox_inches='tight')
# plt.show()

# cobotta
pinv = [0.414, 0.531, 0.614, 0.655, 0.72, 0.71, 0.76, 0.746, 0.764, 0.784, 0.809, 0.8, 0.803, 0.81, 0.826, 0.826, 0.816, 0.808, 0.836, 0.839, 0.843, 0.853, 0.819, 0.83, 0.832, 0.851, 0.861, 0.856, 0.867, 0.858, 0.845, 0.866, 0.841, 0.873, 0.867, 0.885, 0.874, 0.884, 0.863, 0.872, 0.876, 0.898, 0.874, 0.905, 0.878, 0.89, 0.891, 0.887, 0.88, 0.908]
pinv_cw_sort = [0.799, 0.856, 0.88, 0.885, 0.912, 0.922, 0.934, 0.937, 0.931, 0.937, 0.939, 0.939, 0.946, 0.945, 0.942, 0.945, 0.943, 0.945, 0.946, 0.96, 0.954, 0.947, 0.943, 0.959, 0.951, 0.948, 0.957, 0.952, 0.962, 0.962, 0.96, 0.956, 0.96, 0.957, 0.963, 0.956, 0.951, 0.949, 0.96, 0.957, 0.96, 0.966, 0.964, 0.966, 0.965, 0.957, 0.968, 0.955, 0.969, 0.975]
pinv_cw = [0.518, 0.651, 0.709, 0.776, 0.801, 0.792, 0.839, 0.83, 0.852, 0.859, 0.865, 0.882, 0.879, 0.892, 0.896, 0.902, 0.906, 0.905, 0.919, 0.911, 0.916, 0.924, 0.909, 0.923, 0.929, 0.927, 0.925, 0.925, 0.939, 0.917, 0.947, 0.931, 0.938, 0.945, 0.925, 0.928, 0.933, 0.94, 0.938, 0.937, 0.943, 0.94, 0.951, 0.945, 0.959, 0.959, 0.953, 0.942, 0.939, 0.951]

pinv = savgol_filter(pinv, window_length=5, polyorder=2)
pinv_cw = savgol_filter(pinv_cw, window_length=5, polyorder=2)
pinv_cw_sort = savgol_filter(pinv_cw_sort, window_length=5, polyorder=2)

red = (250/255, 127/255, 111/255)
yellow = (255/255, 190/255, 122/255)
blue = (130/255, 176/255, 210/255)

plt.figure(figsize=(12, 6))
plt.plot(best_sol_range, pinv, label='cbt_pinv', color=yellow, linewidth=2.5, linestyle='--')
plt.plot(best_sol_range, pinv_cw, label='cbt_pinv_cw', color=blue, linewidth=2.5, linestyle='--')
plt.plot(best_sol_range, pinv_cw_sort, label='cbt_pinv_cw_sort', color=red, linewidth=2.5, linestyle='--')

pinv = [0.638, 0.811, 0.896, 0.933, 0.957, 0.95, 0.963, 0.974, 0.972, 0.988, 0.982, 0.991, 0.983, 0.987, 0.988, 0.984, 0.991, 0.991, 0.986, 0.994, 0.991, 0.989, 0.992, 0.992, 0.996, 0.991, 0.997, 0.996, 0.995, 0.998, 0.993, 0.99, 0.989, 0.989, 0.995, 0.995, 0.998, 0.997, 0.997, 0.998, 0.993, 0.998, 0.994, 0.993, 0.995, 0.996, 0.997, 0.989, 0.993, 0.996]
pinv_cw = [0.704, 0.852, 0.915, 0.94, 0.957, 0.977, 0.981, 0.985, 0.99, 0.991, 0.983, 0.991, 0.993, 0.994, 0.993, 0.996, 0.993, 0.995, 0.993, 0.997, 0.991, 0.992, 0.995, 0.997, 0.996, 0.993, 0.993, 0.997, 0.993, 0.994, 0.997, 0.996, 0.997, 0.996, 0.997, 0.998, 0.992, 0.997, 0.994, 0.997, 0.998, 0.998, 0.994, 0.997, 0.994, 0.996, 0.997, 0.998, 0.997, 0.995]
pinv_cw_sort = [0.885, 0.965, 0.979, 0.991, 0.989, 0.99, 0.993, 0.991, 0.996, 0.996, 0.987, 0.998, 0.993, 0.998, 0.998, 0.999, 0.99, 0.997, 0.994, 0.995, 0.996, 0.991, 0.997, 0.995, 0.996, 1.0, 0.999, 0.999, 0.998, 0.995, 0.996, 0.998, 0.998, 0.998, 0.997, 0.996, 0.998, 0.999, 1.0, 0.997, 0.999, 0.998, 0.995, 1.0, 0.998, 0.999, 0.999, 0.997, 0.998, 1.0]

pinv = savgol_filter(pinv, window_length=3, polyorder=2)
pinv_cw = savgol_filter(pinv_cw, window_length=3, polyorder=2)
pinv_cw_sort = savgol_filter(pinv_cw_sort, window_length=3, polyorder=2)

plt.plot(best_sol_range, pinv, label='yumi_pinv', color=yellow, linewidth=2.8)
plt.plot(best_sol_range, pinv_cw, label='yumi_pinv_cw', color=blue, linewidth=2.8)
plt.plot(best_sol_range, pinv_cw_sort, label='yumi_pinv_cw_sort', color=red, linewidth=2.8)

plt.xlabel('Best Solution Number (best_sol_num)')
plt.ylabel('Success Rate')
plt.xlim(-0.2, 50.3) 
plt.ylim(0.4, 1.01)  
ax = plt.gca()
ax.spines['top'].set_linewidth(2)   
ax.spines['bottom'].set_linewidth(2) 
ax.spines['left'].set_linewidth(2)  
ax.spines['right'].set_linewidth(2)
# plt.legend()
plt.savefig("yumi_cobotta_seed_plot", dpi=600, bbox_inches='tight')
plt.show()

