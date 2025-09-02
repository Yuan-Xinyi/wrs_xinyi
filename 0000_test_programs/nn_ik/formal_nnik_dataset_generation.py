
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
robot_list = ['cbt','cbtpro1300', 'ur3', 'yumi']

'''define robot'''
for robot in robot_list:
    if robot == 'yumi':
        robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
        abv_name = 'yumi'
    elif robot == 'cbt':
        robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
        abv_name = 'cbt'
    elif robot == 'ur3':
        robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), enable_cc=True)
        abv_name = 'ur3'
    elif robot == 'cbtpro1300':
        robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)
        abv_name = 'cbtpro1300'
    else:
        print("Invalid robot name")

    file_name = f'0000_test_programs/nn_ik/datasets/ikdiff/{abv_name}_ik_dataset_rotquat.npz'
    nupdate = 1000000

    pos_rotq_list = []
    jnt_list = [] 
    print('Generating dataset...')
    print('current robot: ', robot.name)
    for _ in tqdm(range(nupdate)):
        jnt_values = robot.rand_conf()
        tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
        result = robot.ik(tgt_pos, tgt_rotmat)
        if result is not None:
            tgt_rotq = rm.rotmat_to_quaternion(tgt_rotmat)
            pos_rotq_list.append(np.concatenate((tgt_pos.flatten(), tgt_rotq.flatten())))
            jnt_list.append(result)

    pos_rotq = np.array(pos_rotq_list)
    jnt = np.array(jnt_list)

    np.savez(file_name, pos_rotq=pos_rotq, jnt=jnt)

