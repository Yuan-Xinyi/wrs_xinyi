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
        gth_jnt_values = [-2.51300455,  0.47110806,  1.85232117, -1.92527543, -0.72548964,
       -0.1249064 ]

        tgt_pos, tgt_rotmat = robot.fk(jnt_values = gth_jnt_values)
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

        # for i in range(len(jnt_list) - 1): 
        #     jnt1 = jnt_list[i]
        #     jnt2 = jnt_list[i + 1]

        #     s_pos, _ = robot.fk(jnt_values=jnt1)
        #     e_pos, _ = robot.fk(jnt_values=jnt2)
        #     if i < 3:
        #         mgm.gen_arrow(spos=s_pos, epos=e_pos, stick_radius=.0025, rgb=[1,0,0]).attach_to(base)

        #     robot.goto_given_conf(jnt_values=jnt1)
        #     arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[1, 0, 0])
        #     arm_mesh.attach_to(base)
        
        # robot.goto_given_conf(jnt_values=jnt_list[-1])
        # final_arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[1, 0, 0])
        # final_arm_mesh.attach_to(base)
        # base.run()

        '''fail'''
        # jnt_list = [[-2.0362175555555555, 0.6981313749999998, 1.8350890857142859, -1.2715971428571429, 0.3490654999999998, -1.780236], [-2.357027262645955, 0.9109718315023481, 2.013393539534542, -1.7590575858911615, -0.048002566707148975, -1.490029605230652], [-2.0936323356471145, 1.0091413675302257, 1.844885178030642, -0.7313874288673763, 0.12489139607684513, -2.329261523411607], [-1.4072324261069318, 1.3367667458247157, 0.446464366168678, 3.414435272195292, 0.9939154090999356, -5.863016479821201], [-1.548619156849455, 0.8316407333384748, 1.4863326403716044, 3.299357205954795, 1.1772579553235296, -5.6654129099634565], [-1.7116442621595436, 0.6349222302557589, 2.2297485630135094, 3.188157661424021, 1.3953929125235964, -5.474469251595798], [-1.8390028046222917, 1.035440962224811, 2.3343695543492617, 3.0807059192474546, 1.6058479680536948, -5.289961043079765], [-1.8670854176924452, 0.2328442954837957, 1.842074389766048, 2.9768756657558684, 0.042172407859287775, -5.111671387896527], [-1.677007093957025, 0.4447280540634502, 1.397842508356551, 2.8765448444797275, -0.4452704555315874, -4.939390699676655]]
        
        # for i in range(len(jnt_list) - 1): 
        #     jnt1 = jnt_list[i]
        #     jnt2 = jnt_list[i + 1]

        #     s_pos, _ = robot.fk(jnt_values=jnt1)
        #     e_pos, _ = robot.fk(jnt_values=jnt2)
        #     mgm.gen_arrow(spos=s_pos, epos=e_pos, stick_radius=.0025, rgb=[0,0,0]).attach_to(base)

        #     robot.goto_given_conf(jnt_values=jnt1)
        #     arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[0, 0, 1])
        #     arm_mesh.attach_to(base)
        
        # robot.goto_given_conf(jnt_values=jnt_list[-1])
        # final_arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[0, 0, 1])
        # final_arm_mesh.attach_to(base)

        # robot.goto_given_conf(jnt_values=gth_jnt_values)
        # arm_mesh = robot.gen_meshmodel(alpha=0.25, rgb=[0,1,0])
        # arm_mesh.attach_to(base)

        # base.run()

        '''plot all ten seeds'''
        center_seed = [-1.45444111,  0.3490655 ,  1.83508909, -0.42386571,  0.3490655 ,
       -0.593412  ]
        all = np.array([[-2.03621756e+00,  3.49065500e-01,  1.22671717e+00,
         4.23865714e-01,  1.01810833e+00, -1.78023600e+00],
       [-2.03621756e+00,  3.49065500e-01,  1.83508909e+00,
        -2.11932857e+00, -9.89020167e-01,  5.93412000e-01],
       [-2.03621756e+00,  6.98131375e-01,  9.22531214e-01,
         4.23865714e-01,  1.01810833e+00, -1.78023600e+00],
       [-2.03621756e+00, -3.75000000e-07,  1.53090313e+00,
         4.23865714e-01,  1.01810833e+00, -1.78023600e+00],
       [-2.03621756e+00, -3.75000000e-07,  2.13927504e+00,
        -2.11932857e+00, -9.89020167e-01,  5.93412000e-01],
       [-2.03621756e+00,  3.49065500e-01,  1.53090313e+00,
        -2.11932857e+00, -9.89020167e-01,  5.93412000e-01],
       [-1.45444111e+00,  6.98131375e-01,  1.53090313e+00,
         4.23865714e-01,  3.49065500e-01, -1.78023600e+00],
       [-1.45444111e+00,  3.49065500e-01,  1.83508909e+00,
        -4.23865714e-01,  3.49065500e-01, -5.93412000e-01],
       [-2.03621756e+00, -3.75000000e-07,  1.83508909e+00,
        -2.11932857e+00, -9.89020167e-01,  5.93412000e-01],
       [-1.45444111e+00,  6.98131375e-01,  2.13927504e+00,
        -4.23865714e-01, -3.19977333e-01, -5.93412000e-01],
       [-1.45444111e+00, -3.75000000e-07,  1.53090313e+00,
         4.23865714e-01,  1.01810833e+00, -1.78023600e+00],
       [-2.03621756e+00, -3.49066250e-01,  1.53090313e+00,
         4.23865714e-01,  1.68715117e+00, -1.78023600e+00],
       [-2.03621756e+00,  6.98131375e-01,  2.13927504e+00,
        -1.27159714e+00, -9.89020167e-01, -5.93412000e-01],
       [-2.03621756e+00,  1.04719725e+00,  1.83508909e+00,
        -1.27159714e+00, -9.89020167e-01, -5.93412000e-01],
       [-1.45444111e+00,  6.98131375e-01,  1.83508909e+00,
        -4.23865714e-01, -3.19977333e-01, -5.93412000e-01],
       [-2.03621756e+00,  3.49065500e-01,  9.22531214e-01,
         4.23865714e-01,  1.01810833e+00, -1.78023600e+00],
       [-1.45444111e+00,  3.49065500e-01,  1.83508909e+00,
         4.23865714e-01,  3.49065500e-01, -1.78023600e+00],
       [-2.03621756e+00,  3.49065500e-01,  1.83508909e+00,
         4.23865714e-01,  3.49065500e-01, -1.78023600e+00],
       [-2.03621756e+00,  6.98131375e-01,  1.22671717e+00,
        -2.11932857e+00, -9.89020167e-01,  5.93412000e-01],
       [-1.45444111e+00, -3.75000000e-07,  2.13927504e+00,
        -4.23865714e-01,  3.49065500e-01, -5.93412000e-01]])

        for i in range(all.shape[0]):
            # robot.goto_given_conf(jnt_values=jnt)
            robot.goto_given_conf(jnt_values=all[i])
            arm_mesh = robot.gen_meshmodel(alpha=.5)
            arm_mesh.attach_to(base)
        
        robot.goto_given_conf(jnt_values=center_seed)
        arm_mesh = robot.gen_meshmodel(alpha=.3, rgb=[0,0,1])
        arm_mesh.attach_to(base)

        robot.goto_given_conf(jnt_values=gth_jnt_values)
        arm_mesh = robot.gen_meshmodel(alpha=.3, rgb=[0,1,0])
        arm_mesh.attach_to(base)

        base.run()
        
        '''plot 2 seeds'''
    #     success_seed = [ 2.03621756,  1.39626312,  2.13927504,  2.11932857,  1.01810833,
    #    -1.780236  ]
    #     fail_seed = [-2.03621756,  1.39626312,  1.53090313,  0.42386571, -0.98902017,
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

    #     robot.goto_given_conf(jnt_values=gth_jnt_values)
    #     arm_mesh = robot.gen_meshmodel(alpha=0.25, rgb=[0,1,0])
    #     arm_mesh.attach_to(base)
    #     base.run()

        '''calculate the pos and rot bewteen seeds'''
        # succeed_seed = [-1.45444111, -0.34906625,  1.22671717,  0.42386571,  0.3490655 , 1.780236  ]
        # fail_seed = [-1.45444111, -0.69813213,  1.53090313,  0.42386571,  0.3490655 , 1.780236  ]

        # s_pos, s_rotmat = robot.fk(jnt_values=succeed_seed)
        # f_pos, f_rotmat = robot.fk(jnt_values=fail_seed)
        
        # s_pos_err, s_rot_err, _ = rm.diff_between_poses(s_pos*1000, s_rotmat, tgt_pos*1000, tgt_rotmat)
        # f_pos_err, f_rot_err, _ = rm.diff_between_poses(f_pos*1000, f_rotmat, tgt_pos*1000, tgt_rotmat)
        # print(f'succeed seed pos error: {s_pos_err}, rot error: {s_rot_err}')
        # print(f'fail seed pos error: {f_pos_err}, rot error: {f_rot_err}')
