from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.rs007l.rs007l as rs007l
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.manipulators.ur3e.ur3e as ur3e
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi
import wrs.robot_sim.robots.cobotta_pro1300.cobotta_pro1300 as cbtpro1300
import wrs.robot_sim.robots.cobotta_pro900.cobotta_pro900_spine as cbtpro900

import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
import random

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)
np.random.seed(42)
random_seed_num = 1


nupdate = 10000
best_sol_num_list = range(1,101)
robot_list = ['yumi', 'cbt','ur3', 'cbtpro1300']

if __name__ == '__main__':
# while True:

    for robot in robot_list:
        time_ikseed = np.zeros((len(best_sol_num_list), nupdate))
        success_ikseed = np.zeros((len(best_sol_num_list), 1))
        if robot == 'yumi':
            robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
        elif robot == 'cbt':
            robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
        elif robot == 'ur3':
            robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), enable_cc=True)
        elif robot == 'cbtpro1300':
            robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)
        else:
            print("Invalid robot name")
        
        for best_sol_num in best_sol_num_list:
            for _ in range(random_seed_num):
                '''set random seed'''
                np_seed = random.randint(0, 2**32 - 1)
                np.random.seed(np_seed)
                
                success_num = 0
                time_list = []
                pos_err_list = []
                rot_err_list = []

                for i in tqdm(range(nupdate)):
                    jnt_values = robot.rand_conf()
                    tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
                    tic = time.time()
                    result = robot.ik(tgt_pos, tgt_rotmat, best_sol_num = best_sol_num)
                    toc = time.time()
                    time_list.append(toc-tic)
                    time_ikseed[best_sol_num-1, i] = toc - tic
                    if result is not None:
                        success_num += 1
                        pred_pos, pred_rotmat = robot.fk(jnt_values=result)
                        pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
                        pos_err_list.append(pos_err)
                        rot_err_list.append(rot_err)
            success_ikseed[best_sol_num-1] = success_num / (nupdate * random_seed_num) * 100

        np.save(f"{robot.name}_time_ikseed.npy", time_ikseed)
        np.save(f"{robot.name}_success_ikseed.npy", success_ikseed)
