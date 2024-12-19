from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import numpy as np
import sys
import wrs.robot_sim.manipulators.manipulator_interface as mi

def append_to_logfile(filename, data):
    with open(filename, "a") as f: 
        json.dump(data, f) 
        f.write("\n")

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
nupdate = 100

data = {}

with open("ik_time_label.json", "w") as json_file:
    json.dump(data, json_file)

count_1 = 0
count_2 = 0
limit = 1000

for _ in range(1000):
# while True:
    success_rate = 0
    time_list = []
    for _ in range(nupdate):
        jnt_values = robot.rand_conf()
        tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
        tic = time.time()
        result = robot.ik(tgt_pos, tgt_rotmat)
        toc = time.time()
        t = toc - tic
        time_list.append(t)
        if result is not None:
            success_rate += 1
        
        if result is None:
            result = np.zeros(6)
        # save data
        if count_1 <= limit and 0.004 < t < 0.008:
            pos_rot = np.concatenate((tgt_pos, rm.rotmat_to_wvec(tgt_rotmat)))
            data = {"label": '1',
                    "res_bool": False if result is None or all(x == 0 for x in result) else True,
                    "target": pos_rot.tolist(), 
                    "jnt_result": result.tolist()}
            append_to_logfile("ik_time_label.json", data)  
            count_1 += 1
            print('label 1: ', count_1)
        elif count_2 <= limit and t > 0.008:
            pos_rot = np.concatenate((tgt_pos, rm.rotmat_to_wvec(tgt_rotmat)))
            data = {"label": '2',
                    "res_bool": False if result is None or all(x == 0 for x in result) else True,
                    "target": pos_rot.tolist(), 
                    "jnt_result": result.tolist()}
            append_to_logfile("ik_time_label.json", data)
            count_2 += 1
            print('label 2: ', count_2)
        elif count_1 > limit and count_2 > limit:
            sys.exit()
        else:
            continue

    # print(success_rate)
    # time.sleep(5)

    # plt.plot(range(nupdate), time_list)
    # plt.show()