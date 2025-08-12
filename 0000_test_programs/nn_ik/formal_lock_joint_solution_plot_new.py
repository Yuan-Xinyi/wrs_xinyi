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
        kdt_jnt_data = kdt_jnt_data[0]
        jnt_values = [ 2.46465061, -0.03160745,  0.54272041, -1.63604277,  1.34401953,
        1.51478841]
        
        # robot.goto_given_conf(jnt_values=jnt_values)
        # arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[0, 1, 0])
        # arm_mesh.attach_to(base)

        # tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
        # mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        
        '''success'''
        jnt_list = [([ 1.8319041 , -0.12240591,  1.42100487, -1.77418962,  1.01852414,
        2.15477549]),
([ 1.96601352, -0.39683286,  1.34041698, -1.4550684 ,  0.94739692,
        1.91770006]),
([ 2.37143097, -0.2550836 ,  0.89672728, -1.48047744,  1.35139901,
        1.63405669]),
([ 2.46942197, -0.1091759 ,  0.6871292 , -1.65588044,  1.36204283,
        1.58879198]),
([ 2.46022508, -0.05216709,  0.58026736, -1.63710527,  1.34223996,
        1.5326172 ])
        ]
        for i in range(len(jnt_list) - 1): 
        # for i in range(3):
            jnt1 = jnt_list[i]
            jnt2 = jnt_list[i + 1]

            s_pos, _ = robot.fk(jnt_values=jnt1)
            e_pos, _ = robot.fk(jnt_values=jnt2)
            
            mgm.gen_arrow(spos=s_pos, epos=e_pos, stick_radius=.0015, rgb=[0,0,0]).attach_to(base)

        #     robot.goto_given_conf(jnt_values=jnt1)
        #     arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[0, 0, 1])
        #     arm_mesh.attach_to(base)
        
        # robot.goto_given_conf(jnt_values=jnt_list[-1])
        # final_arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[0, 0, 1])
        # final_arm_mesh.attach_to(base)
        # tgt_pos, tgt_rotmat = robot.fk(jnt_values=jnt_list[-1])
        # mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        base.run()

        '''fail'''
#         jnt_list = [
#                     ([ 1.47052074, -0.6708641 ,  0.41723254,  2.45137195, -1.42199221,
#        -2.67201078]),
# ([ 1.69769405, -0.35806462,  0.40171886,  2.3138503 , -1.27356563,
#        -2.26779433]),
# ([ 2.34865091e+00,  1.25287107e-03, -1.24958001e-01,  1.76539201e+00,
#        -1.44890420e+00, -2.23047901e+00]),
# ([ 2.62174387,  0.28938752, -0.73397048,  1.53617625, -1.39336807,
#        -2.59961558]),
# ([ 2.53921936,  0.06472877, -0.32109406,  1.62045224, -1.36265589,
#        -2.4176524 ])]

#         for i in range(len(jnt_list) - 1): 
#             jnt1 = jnt_list[i]
#             jnt2 = jnt_list[i + 1]

#             s_pos, _ = robot.fk(jnt_values=jnt1)
#             e_pos, _ = robot.fk(jnt_values=jnt2)
            
#             mgm.gen_arrow(spos=s_pos, epos=e_pos, stick_radius=.0025, rgb=[0,0,0]).attach_to(base)

#             robot.goto_given_conf(jnt_values=jnt1)
#             arm_mesh = robot.gen_meshmodel(alpha=0.15, rgb=[1, 0, 0])
#             arm_mesh.attach_to(base)

#             if i == 6:
#                 tmp_arm_stick = robot.gen_stickmodel(toggle_flange_frame=True)
#                 tmp_arm_stick.attach_to(base)
        
#         robot.goto_given_conf(jnt_values=jnt_list[-1])
#         final_arm_mesh = robot.gen_meshmodel(alpha=0.15, rgb=[1, 0, 0])
#         final_arm_mesh.attach_to(base)

#         tgt_pos, tgt_rotmat = robot.fk(jnt_values=jnt_list[-1])
#         mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

#         base.run()

        '''plot tcp nearest ten seeds'''
    #     # jnt_list = np.array([14502,  7100,  8862,  7704,  2064,  3222, 15013,  3342, 12740, 13344])
    #     jnt_list = np.array([11619,   456, 14897,  7825,  3852, 36524, 29957, 44895, 16774,
    #    31690, 49088,  8515, 11058, 28791,  1202, 47109,  6112, 26581,
    #      991, 58375,   691, 35186, 44620, 53486,  8198, 15269, 29849,
    #    26585, 17882,  7650, 37036, 27648,  9033, 40566, 44444, 27591,
    #    32594, 10652, 31555, 29393, 40610, 40164,  1602, 35799, 14634,
    #    20556, 50621, 36010, 20813, 46415, 24796, 13162,  2915, 16215,
    #    19305, 13908, 56962, 57705, 26923, 25187, 58243,  6025, 47522,
    #    19878, 43017, 53260,  3046, 12359, 14345, 44752, 34841,  7236,
    #    52267, 54013, 37934, 24766, 18147, 39420, 45697, 32604, 30267,
    #     2390, 34248, 40069,  7684, 50813, 15989, 48616, 11736, 39803,
    #    57484, 41242, 30377, 32465, 47444, 56812,  7834, 47373, 47150,
    #    46664, 48134, 31644, 44660,  5224, 26389, 12387,  7362, 39881,
    #    43714, 40023, 43659, 36367, 53318, 34556, 50509, 12843, 48812,
    #    26816, 55019, 32939, 50405, 26778, 42903, 13116, 10261, 31839,
    #     8242, 58395,  8417, 50754, 47958, 60336, 57959, 29663, 30289,
    #    50116, 49076, 60298, 24058, 43791, 33150, 14477, 12251, 20223,
    #    54956, 30983, 14867,  8301, 17060, 54361, 56911,   955, 15133,
    #    25881, 35671, 59890, 53465,  5764, 13215, 55001, 45268, 59530,
    #     2750, 45547, 28349, 48795, 35917, 18529, 26963, 12640, 56702,
    #    10431, 18473, 10940,  7587, 18313, 39912,  7110,  4571, 54765,
    #    10909, 16023, 43182, 59261, 39545, 52751, 40637, 46731,  3602,
    #    33848, 53713, 38906, 51705, 56125, 51005, 38974, 15732,  3320,
    #    33282, 31973])
    #     for jnt in jnt_list:
    #         # robot.goto_given_conf(jnt_values=jnt)
    #         robot.goto_given_conf(jnt_values=kdt_jnt_data[jnt])
    #         arm_mesh = robot.gen_meshmodel(alpha=.15)
    #         arm_mesh.attach_to(base)
        
    #     robot.goto_given_conf(jnt_values=jnt_values)
    #     arm_mesh = robot.gen_meshmodel(alpha=0.9, rgb=[0.133, 0.545, 0.133])
    #     arm_mesh.attach_to(base)
    #     base.run()

        '''plot the most linear 10 seeds'''
    #     # jnt_list = np.array([ 9923,  3512,  9032,  4232,  9152,  1295,  9207,  6935, 15563,
    #     # 9872,  1340])
    #     jnt_list = np.array([ 8417, 26923,  7236, 53486, 14477,   456, 31644, 10431, 52751,
    #    37036, 49088, 29849, 12387, 11619, 43182, 26581, 48134, 32604,
    #    39545, 52267])
    #     fail_idx = jnt_list[0]
    #     success_idx = jnt_list[1]

    #     # 先画普通种子
    #     for jnt in jnt_list:
    #         if jnt in [fail_idx, success_idx]:
    #             continue  # 跳过成功和失败种子
    #         robot.goto_given_conf(jnt_values=kdt_jnt_data[jnt])
    #         arm_mesh = robot.gen_meshmodel(alpha=.15)
    #         arm_mesh.attach_to(base)

    #     # 再画失败种子（红色）
    #     robot.goto_given_conf(jnt_values=kdt_jnt_data[fail_idx])
    #     arm_mesh = robot.gen_meshmodel(alpha=.4, rgb=[1,0,0])
    #     arm_mesh.attach_to(base)
    #     tgt_pos, tgt_rotmat = robot.fk(jnt_values=kdt_jnt_data[fail_idx])
    #     mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

    #     # 再画成功种子（蓝色）
    #     robot.goto_given_conf(jnt_values=kdt_jnt_data[success_idx])
    #     arm_mesh = robot.gen_meshmodel(alpha=.4, rgb=[0,0,1])
    #     arm_mesh.attach_to(base)
    #     tgt_pos, tgt_rotmat = robot.fk(jnt_values=kdt_jnt_data[success_idx])
    #     mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

    #     base.run()
    #     # fail_idx = jnt_list[0]
    #     # success_idx = jnt_list[1]
    #     # for jnt in jnt_list:
    #     #     if jnt == fail_idx:
    #     #         robot.goto_given_conf(jnt_values=kdt_jnt_data[jnt])
    #     #         arm_mesh = robot.gen_meshmodel(alpha=.35, rgb=[1,0,0])
    #     #         arm_mesh.attach_to(base)

    #     #         tgt_pos, tgt_rotmat = robot.fk(jnt_values = kdt_jnt_data[jnt])
    #     #         mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

    #     #     elif jnt == success_idx:
    #     #         robot.goto_given_conf(jnt_values=kdt_jnt_data[jnt])
    #     #         arm_mesh = robot.gen_meshmodel(alpha=.35, rgb=[0,0,1])
    #     #         arm_mesh.attach_to(base)

    #     #         tgt_pos, tgt_rotmat = robot.fk(jnt_values = kdt_jnt_data[jnt])
    #     #         mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    #     #     else:
    #     #         robot.goto_given_conf(jnt_values=kdt_jnt_data[jnt])
    #     #         arm_mesh = robot.gen_meshmodel(alpha=.15)
    #     #         arm_mesh.attach_to(base)

    #     # base.run()

        
        '''plot 2 seeds'''
    #     success_seed = [-1.45444111, -0.69813213,  2.13927504, -1.27159714, -0.98902017,
    #    -1.780236  ]
    #     fail_seed = [ 0.87266467, -0.34906625,  1.83508909,  1.27159714, -0.98902017,
    #     1.780236  ]
    #     robot.goto_given_conf(jnt_values=fail_seed)
    #     arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[1,0,0])
    #     arm_mesh.attach_to(base)
    #     tgt_pos, tgt_rotmat = robot.fk(jnt_values = fail_seed)
    #     mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        
        
    #     robot.goto_given_conf(jnt_values=success_seed)
    #     arm_mesh = robot.gen_meshmodel(alpha=0.25, rgb=[0,0,1])
    #     arm_mesh.attach_to(base)
    #     tgt_pos, tgt_rotmat = robot.fk(jnt_values = success_seed)
    #     mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

    #     base.run()