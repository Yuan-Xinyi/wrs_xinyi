
from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.rs007l.rs007l as rs007l
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.manipulators.ur3e.ur3e as ur3e
import wrs.robot_sim.robots.cobotta_pro1300.cobotta_pro1300 as cbtpro1300
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

'''define robot'''
# robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
# robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
# robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), ik_solver='d' ,enable_cc=True)
# robot = rs007l.RS007L(pos=rm.vec(0.1, .3, .5), enable_cc=True)
robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)

file_name = f'0000_test_programs/nn_ik/datasets/formal/{robot.name}_ik_dataset_rotquat.npz'
nupdate = 1000000

pos_rotv_list = []
jnt_list = [] 
print('Generating dataset...')
print('current robot: ', robot.name)
for _ in tqdm(range(nupdate)):
    jnt_values = robot.rand_conf()
    tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
    result = robot.ik(tgt_pos, tgt_rotmat, best_sol_num = 1)
    if result is not None:
        # tgt_rotv = rm.rotmat_to_wvec(tgt_rotmat)
        tgt_rotv = rm.rotmat_to_quaternion(tgt_rotmat)
        pos_rotv_list.append(np.concatenate((tgt_pos.flatten(), tgt_rotv.flatten())))
        jnt_list.append(result)

pos_rotv = np.array(pos_rotv_list)
jnt = np.array(jnt_list)

np.savez(file_name, pos_rotv=pos_rotv, jnt=jnt)

