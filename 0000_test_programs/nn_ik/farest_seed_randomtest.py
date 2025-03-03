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

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

'''define robot'''
# robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
# robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), ik_solver='d' ,enable_cc=True)
# robot = rs007l.RS007L(pos=rm.vec(0.1, .3, .5), enable_cc=True)
# robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)
# robot = cbtpro900.CobottaPro900Spine(pos=rm.vec(0.1, .3, .5), enable_cc=True)




nupdate = 1

if __name__ == '__main__':
# while True:
    success_num = 0
    time_list = []
    pos_err_list = []
    rot_err_list = []
    best_sol_num = 3

    for i in tqdm(range(nupdate)):
        # jnt_values = robot.rand_conf()
        jnt_values = [-0.89298267, -0.99745798,  2.20644947, -1.90284232, -0.62446831,
       -0.94061137]
        print("*" * 150 + "\n")
        print('jnt', repr(jnt_values))
        tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
        # print('tgt_pos', tgt_pos)
        tic = time.time()
        result = robot.ik(tgt_pos, tgt_rotmat, best_sol_num = best_sol_num)
        toc = time.time()
        time_list.append(toc-tic)
        
        if result is not None:
            success_num += 1

            # calculate the pos error and rot error
            pred_pos, pred_rotmat = robot.fk(jnt_values=result)
            pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
            pos_err_list.append(pos_err)
            rot_err_list.append(rot_err)
        print("*" * 150 + "\n")

