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
        gth_jnt_values = [ 2.50413245,  1.73323309,  1.59021775,  2.2408374 ,  1.17974473,
       -1.82249448]

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
            
        #     mgm.gen_arrow(spos=s_pos, epos=e_pos, stick_radius=.0025, rgb=[0,0,0]).attach_to(base)

        #     robot.goto_given_conf(jnt_values=jnt1)
        #     arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[1, 0, 0])
        #     arm_mesh.attach_to(base)
        
        # robot.goto_given_conf(jnt_values=jnt_list[-1])
        # final_arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[1, 0, 0])
        # final_arm_mesh.attach_to(base)
        # base.run()

        '''fail'''
        # jnt_list = [[2.0362175555555555, 1.3962631249999997, 1.8350890857142859, -2.1193285714285715, 1.018108333333333, -1.780236], [1.9953442386687867, 1.5978257717622828, 2.0891666964283977, -2.568655956337472, 1.3546735264971765, -1.3249159564108548], [2.069828807892358, 1.7489069567002367, 1.9440666478235353, -2.3571179027488016, 1.4205525806788677, -1.4334428591509933], [2.134419070302458, 1.6486507044455414, 2.013444831949458, -2.306220072343389, 1.378377672874309, -1.4271696851720643], [2.0740369180342135, 1.7518284100453472, 1.931298795897575, -2.3574008003900064, 1.4159206391835715, -1.433420888535725], [2.1298245982242543, 1.6513629241219991, 2.0127035590110824, -2.309785715082258, 1.3818267894015746, -1.4278098205387049], [2.073674105264744, 1.7519258858700795, 1.930950188147889, -2.357579688910322, 1.4159426798815216, -1.433924900584717], [2.1297151374675236, 1.6514534187555212, 2.0126618109618173, -2.309871583429196, 1.3819114858327428, -1.4278178185141555], [2.0736655416394294, 1.751927850560981, 1.9309401485466235, -2.357583646473237, 1.4159417280442217, -1.4339383680691151], [2.1297120584193103, 1.6514552427358118, 2.0126613551462995, -2.309873967152302, 1.3819138886991285, -1.4278181178047373], [2.073665302419872, 1.7519279143858024, 1.930939919632688, -2.3575837657632692, 1.4159417424422056, -1.4339386982939002], [2.129711986598552, 1.6514553019895175, 2.012661327859258, -2.309874023467051, 1.3819139442934671, -1.4278181230350004], [2.073665296807979, 1.7519279156734489, 1.9309399130590856, -2.357583768359025, 1.4159417418211449, -1.4339387071141698], [2.129711984582468, 1.651455303184943, 2.0126613275596483, -2.309874025027894, 1.3819139458667884, -1.4278181232308051], [2.0736652966513485, 1.7519279157152232, 1.930939912909118, -2.3575837684371175, 1.415941741830503, -1.4339387073304626], [2.129711984535419, 1.6514553032237256, 2.012661327541806, -2.3098740250647842, 1.3819139459032084, -1.4278181232342353], [2.0736652966476723, 1.7519279157160677, 1.930939912904814, -2.3575837684388183, 1.4159417418300981, -1.4339387073362382], [2.1297119845340986, 1.6514553032245094, 2.012661327541609, -2.309874025065806, 1.3819139459042382, -1.4278181232343634], [2.0736652966475697, 1.7519279157160947, 1.9309399129047158, -2.3575837684388694, 1.415941741830104, -1.43393870733638], [2.1297119845340675, 1.6514553032245347, 2.012661327541598, -2.309874025065831, 1.3819139459042626, -1.4278181232343656], [2.0736652966475675, 1.7519279157160954, 1.930939912904713, -2.3575837684388703, 1.415941741830104, -1.4339387073363836], [2.1297119845340666, 1.6514553032245352, 2.0126613275415974, -2.3098740250658314, 1.381913945904263, -1.427818123234366]]
        
        # for i in range(len(jnt_list) - 1): 
        #     jnt1 = jnt_list[i]
        #     jnt2 = jnt_list[i + 1]

        #     s_pos, _ = robot.fk(jnt_values=jnt1)
        #     e_pos, _ = robot.fk(jnt_values=jnt2)
        #     # mgm.gen_arrow(spos=s_pos, epos=e_pos, stick_radius=.0025, rgb=[0,0,1]).attach_to(base)

        #     robot.goto_given_conf(jnt_values=jnt1)
        #     arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[0, 0, 1])
        #     arm_mesh.attach_to(base)
        
        # robot.goto_given_conf(jnt_values=jnt_list[-1])
        # final_arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[0, 0, 1])
        # final_arm_mesh.attach_to(base)

        # robot.goto_given_conf(jnt_values=jnt_values)
        # arm_mesh = robot.gen_meshmodel(alpha=0.25, rgb=[0,1,0])
        # arm_mesh.attach_to(base)

        # base.run()

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
        success_seed = [ 2.03621756,  1.39626312,  2.13927504,  2.11932857,  1.01810833,
       -1.780236  ]
        fail_seed = [-2.03621756,  1.39626312,  1.53090313,  0.42386571, -0.98902017,
        1.780236  ]
        
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
