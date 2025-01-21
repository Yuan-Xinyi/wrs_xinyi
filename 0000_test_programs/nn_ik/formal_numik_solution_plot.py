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
import pickle

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

'''define robot'''
# robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
# robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), ik_solver='d' ,enable_cc=True)
# robot = rs007l.RS007L(pos=rm.vec(0.1, .3, .5), enable_cc=True)



nupdate = 1

if __name__ == '__main__':

    mode = 'plot' # ['search' or 'plot']
    if mode == 'search':
        for i in range(nupdate):
            # jnt_values = robot.rand_conf()
            jnt_values = [-1.72052066, -0.25810076,  2.21686054, -0.40935568, -0.89245543,  2.51885436]
            tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
            tic = time.time()
            result = robot.ik(tgt_pos, tgt_rotmat, best_sol_num = 1)
            toc = time.time()
            if result is not None:

                # calculate the pos error and rot error
                pred_pos, pred_rotmat = robot.fk(jnt_values=result)
                pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)

        print(f'jnt_values: {repr(jnt_values)}')
    
    elif mode == 'plot':
        with open('wrs/robot_sim/_data_files/cobotta_arm_jnt_data.pkl', 'rb') as f_jnt:
            kdt_jnt_data = pickle.load(f_jnt)

        jnt_values = [-1.72052066, -0.25810076,  2.21686054, -0.40935568, -0.89245543,  2.51885436]

        tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
        mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        '''success'''
        # jnt_list = [
        #             ([-1.45444111, -0.34906625,  1.22671717,  0.42386571,  0.3490655 ,
        # 1.780236  ]),
        #             ([-2.38708622e+00, -4.55161658e-01,  1.26536246e+00,  2.63233592e+00,
        # 6.20158249e-01, -2.38581972e-03]),
        #             ([-2.45475725, -0.63076311,  1.9041648 ,  2.45657771,  0.96693438,
        # 0.19329105]),
        #             ([-2.40630761, -0.34218633,  1.92287747,  2.23384308,  1.14038202,
        # 0.19958151]),
        #             ([-2.36435034, -0.29588893,  1.78238233,  2.05551734,  0.99864575,
        # 0.32113778]),
        #             ([-2.35329643, -0.29183192,  1.77999731,  2.03880547,  1.00466214,
        # 0.33869052])]
        # for jnt in jnt_list:
        #     robot.goto_given_conf(jnt_values=jnt)
        #     # robot.goto_given_conf(jnt_values=kdt_jnt_data[jnt])
        #     arm_mesh = robot.gen_meshmodel(alpha=.2, rgb=[1,0,0])
        #     arm_mesh.attach_to(base)
        # # robot.goto_given_conf(jnt_values=kdt_jnt_data[best_delta_q])
        # # arm_mesh = robot.gen_meshmodel(alpha=.3)
        # # arm_mesh.attach_to(base)
        # base.run()

        '''fail'''
    #     jnt_list = [
    #         ([-1.45444111, -0.69813213,  1.53090313,  0.42386571,  0.3490655 ,
    #             1.780236  ]),
    #         ([ 0.48123896, -0.94904702,  3.19472325, -3.02431877, -1.57175925,
    #     3.99599829]),
    #     ([ 0.62123236, -0.25373376,  3.02415906, -2.92238896, -1.29910815,
    #     3.86131957]),
    #     ([ 0.07847416, -1.1768159 ,  2.86961554, -3.47265712, -2.6518819 ,
    #     3.73117999]),
    #     ([ 0.11593302, -1.06753271,  2.7295879 , -3.35561678, -2.50236743,
    #     3.60542655]),
    #     ([ 0.17841884, -0.96607634,  2.60271272, -3.24252109, -2.36030214,
    #     3.48391143]),
    #     ([ 0.26420653, -0.46707115,  2.4877546 , -3.13323712, -2.22531489,
    #     3.36649178]),
    #     ([ 4.48085217e-01,  2.07045788e-03,  2.38359421e+00, -3.02763639e+00,
    #    -2.09705303e+00,  3.25302957e+00]),
    #    ([ 0.70091937, -0.04559144,  2.77232836, -2.92559476, -1.97518148,
    #     3.14339142]),
    #     ([ 1.07742916,  0.3912977 ,  2.64143867, -2.75474989, -1.85938187,
    #     3.03744846])
    #     ]
    #     for jnt in jnt_list:
    #         robot.goto_given_conf(jnt_values=jnt)
    #         # robot.goto_given_conf(jnt_values=kdt_jnt_data[jnt])
    #         arm_mesh = robot.gen_meshmodel(alpha=.2, rgb=[0,0,1])
    #         arm_mesh.attach_to(base)
    #     # robot.goto_given_conf(jnt_values=kdt_jnt_data[best_delta_q])
    #     # arm_mesh = robot.gen_meshmodel(alpha=.3)
    #     # arm_mesh.attach_to(base)
    #     base.run()

        '''plot all ten seeds'''
        # jnt_list = [ 1151,  6791, 12431,  2006,  7646,  2842, 13623, 18071,  1387, 7027]
        # for jnt in jnt_list:
        #     # robot.goto_given_conf(jnt_values=jnt)
        #     robot.goto_given_conf(jnt_values=kdt_jnt_data[jnt])
        #     arm_mesh = robot.gen_meshmodel(alpha=.5)
        #     arm_mesh.attach_to(base)
        # robot.goto_given_conf(jnt_values=kdt_jnt_data[best_delta_q])
        # arm_mesh = robot.gen_meshmodel(alpha=.3)
        # arm_mesh.attach_to(base)
        # base.run()
        
        '''plot 2 seeds'''
        # robot.goto_given_conf(jnt_values=kdt_jnt_data[1151])
        # arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[0,0,1])
        # arm_mesh.attach_to(base)
        # tgt_pos, tgt_rotmat = robot.fk(jnt_values = kdt_jnt_data[1151])
        # mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        # robot.goto_given_conf(jnt_values=kdt_jnt_data[6791])
        # arm_mesh = robot.gen_meshmodel(alpha=0.25, rgb=[1,0,0])
        # arm_mesh.attach_to(base)
        # tgt_pos, tgt_rotmat = robot.fk(jnt_values = kdt_jnt_data[6791])
        # mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        # base.run()


