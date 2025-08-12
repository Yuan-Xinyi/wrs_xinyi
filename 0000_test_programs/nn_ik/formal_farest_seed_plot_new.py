from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.rs007l.rs007l as rs007l
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.manipulators.ur3e.ur3e as ur3e
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi
import wrs.modeling.geometric_model as mgm  

import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

'''define robot'''
robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)

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
        gth_jnt_values = [ 2.46465061, -0.03160745,  0.54272041, -1.63604277,  1.34401953,
        1.51478841]

        tgt_pos, tgt_rotmat = robot.fk(jnt_values = gth_jnt_values)
        mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        
        '''success'''
#         jnt_list = [
#                     ([-2.18596054, -0.81857843,  1.73693075, -2.63695461,  1.8117939 ,
#        -2.49685258]),
# ([-2.55899905, -1.1104697 ,  1.86977377, -2.58178888,  1.71483111,
#        -2.48331381]),
# ([-2.03065462, -0.93039289,  1.73876373, -2.99449183,  1.95414809,
#        -2.78579135]),
# ([-1.64060564, -0.78567362,  1.55696414, -3.41606813,  1.8960046 ,
#        -3.15179414]),
# ([-1.81161747, -0.85253106,  1.62312858, -3.22614298,  1.86691306,
#        -3.00593267]),
# ([-1.7672869 , -0.83804827,  1.60479022, -3.27379818,  1.84788227,
#        -3.03815331]),
# ([-1.76798665, -0.83844364,  1.60521683, -3.27259873,  1.84716927,
#        -3.03720474])]

#         for i in range(len(jnt_list) - 1): 
#             jnt1 = jnt_list[i]
#             jnt2 = jnt_list[i + 1]

#             s_pos, _ = robot.fk(jnt_values=jnt1)
#             e_pos, _ = robot.fk(jnt_values=jnt2)
            
#             mgm.gen_arrow(spos=s_pos, epos=e_pos, stick_radius=.0025, rgb=[0,0,0]).attach_to(base)

#             robot.goto_given_conf(jnt_values=jnt1)
#             arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[1, 0, 0])
#             arm_mesh.attach_to(base)
        
#         robot.goto_given_conf(jnt_values=jnt_list[-1])
#         final_arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[1, 0, 0])
#         final_arm_mesh.attach_to(base)
#         base.run()

        '''fail'''
#         jnt_list = [([ 0.17613876, -0.42602273,  1.916269  , -0.86957682, -0.93069285,
#        -1.68385147]),
# ([-0.05728397, -0.53170945,  1.97600567, -1.20460233, -1.21435122,
#        -1.28800557]),
# ([-0.06841161, -0.51433495,  1.92900443, -1.10927678, -1.25765267,
#        -1.35121119])
#         ]

#         for i in range(len(jnt_list) - 1): 
#             jnt1 = jnt_list[i]
#             jnt2 = jnt_list[i + 1]

#             s_pos, _ = robot.fk(jnt_values=jnt1)
#             e_pos, _ = robot.fk(jnt_values=jnt2)
#             # mgm.gen_arrow(spos=s_pos, epos=e_pos, stick_radius=.0025, rgb=[0,0,1]).attach_to(base)

#             robot.goto_given_conf(jnt_values=jnt1)
#             arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[0, 0, 1])
#             arm_mesh.attach_to(base)
        
#         robot.goto_given_conf(jnt_values=jnt_list[-1])
#         final_arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[0, 0, 1])
#         final_arm_mesh.attach_to(base)

#         robot.goto_given_conf(jnt_values=gth_jnt_values)
#         arm_mesh = robot.gen_meshmodel(alpha=0.25, rgb=[0,1,0])
#         arm_mesh.attach_to(base)

#         base.run()

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
        success_seed = [ 2.46022508, -0.05216709,  0.58026736, -1.63710527,  1.34223996,
        1.5326172 ]
        fail_seed = [ 2.53921936,  0.06472877, -0.32109406,  1.62045224, -1.36265589,
       -2.4176524 ]
        
        robot.goto_given_conf(jnt_values=fail_seed)
        arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[1,0,0])
        arm_mesh.attach_to(base)
        tgt_pos, tgt_rotmat = robot.fk(jnt_values = fail_seed)
        mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        robot.goto_given_conf(jnt_values=success_seed)
        arm_mesh = robot.gen_meshmodel(alpha=0.25, rgb=[0,0,1])
        arm_mesh.attach_to(base)
        tgt_pos, tgt_rotmat = robot.fk(jnt_values = success_seed)
        mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

        robot.goto_given_conf(jnt_values=gth_jnt_values)
        arm_mesh = robot.gen_meshmodel(alpha=0.25, rgb=[0,1,0])
        arm_mesh.attach_to(base)
        base.run()

        '''calculate the pos and rot bewteen seeds'''
        # succeed_seed = [-1.45444111, -0.34906625,  1.22671717,  0.42386571,  0.3490655 , 1.780236  ]
        # fail_seed = [-1.45444111, -0.69813213,  1.53090313,  0.42386571,  0.3490655 , 1.780236  ]

        # s_pos, s_rotmat = robot.fk(jnt_values=succeed_seed)
        # f_pos, f_rotmat = robot.fk(jnt_values=fail_seed)
        
        # s_pos_err, s_rot_err, _ = rm.diff_between_poses(s_pos*1000, s_rotmat, tgt_pos*1000, tgt_rotmat)
        # f_pos_err, f_rot_err, _ = rm.diff_between_poses(f_pos*1000, f_rotmat, tgt_pos*1000, tgt_rotmat)
        # print(f'succeed seed pos error: {s_pos_err}, rot error: {s_rot_err}')
        # print(f'fail seed pos error: {f_pos_err}, rot error: {f_rot_err}')
