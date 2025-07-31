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
            jnt_values = [ 1.98596409, -0.56365319,  0.44936407, -0.09795888, -0.21335116,
        0.20067067]
            tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
            tic = time.time()
            result = robot.ik(tgt_pos, tgt_rotmat, best_sol_num = 1)
            toc = time.time()
            if result is not None:
                print('#' * 50)
                print(f"Found solution in {toc - tic:.4f} seconds")
                print('#' * 50)
                # calculate the pos error and rot error
                pred_pos, pred_rotmat = robot.fk(jnt_values=result)
                pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
                
                robot.goto_given_conf(jnt_values=jnt_values)
                arm_mesh = robot.gen_meshmodel(alpha=.3, rgb=[0,1,0])
                arm_mesh.attach_to(base)
        print(f'jnt_values: {repr(jnt_values)}')
        # base.run()
    
    elif mode == 'plot':
        gth_jnt_values = [1.98596409, -0.56365319, 0.44936407, -0.09795888, -0.21335116, 0.20067067]

        tgt_pos, tgt_rotmat = robot.fk(jnt_values = gth_jnt_values)
        mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        
        '''success'''
        # jnt_list = [
        #             ([-1.11208318, -0.06787872,  0.45720109,  2.30594382,  0.03111745,
        # 1.20545655]),
        #             ([-1.73747477, -0.02906479,  0.54489941,  2.96447956,  0.26735655,
        # 0.82182843]),
        #             ([-1.43581982,  0.04822471,  0.48549347,  2.80459122,  0.24365499,
        # 0.69632033]),
        #             ([-1.46110301,  0.06247807,  0.45100052,  2.77127913,  0.20928767,
        # 0.75757947]),
        # ]

        # for i in range(len(jnt_list) - 1): 
        #     jnt1 = jnt_list[i]
        #     jnt2 = jnt_list[i + 1]

        #     s_pos, _ = robot.fk(jnt_values=jnt1)
        #     e_pos, _ = robot.fk(jnt_values=jnt2)
        #     if i < 3:
        #         mgm.gen_arrow(spos=s_pos, epos=e_pos, stick_radius=.0025, rgb=[1,0,0]).attach_to(base)

        #     robot.goto_given_conf(jnt_values=jnt1)
        #     arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[0, 0, 1])
        #     arm_mesh.attach_to(base)
        
        # robot.goto_given_conf(jnt_values=jnt_list[-1])
        # final_arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[0, 0, 1])
        # final_arm_mesh.attach_to(base)
        # base.run()

        '''fail'''
        # jnt_list = [
        #             np.array([-2.28174006, -0.81538117,  1.19386317,  2.05865428,  0.43705984,
        #                     2.28134067]),
        #             np.array([-5.50913844, -1.00762857,  0.76221663,  2.79356645,  1.11148844,
        #                     4.76655987]),
        #             np.array([-5.1391227 , -1.05338323,  1.2612284 ,  2.52801611,  1.26025271,
        #                     4.62956424]),
        #             np.array([-4.35498414, -0.7915569 ,  1.14181475,  1.95591725,  0.99839665,
        #                     4.31186558]),
        #             np.array([-3.28770454, -0.75719778,  0.86324016,  1.14968602,  0.34172116,
        #                     4.23069059]),
        #             np.array([-3.99867133, -0.79454621,  0.55356757,  1.86538279,  0.40894751,
        #                     4.25508885]),
        #             np.array([-3.87390018, -0.71774423,  0.2881113 ,  1.37448394,  0.10218449,
        #                     4.64320048]),
        #             np.array([-4.43378034, -0.59900405,  0.15023784,  0.23865901, -0.04810621,
        #                     6.26779534]),
        #             np.array([-4.34605509, -0.36856491, -0.00721923,  0.14998929,  0.05114879,
        #                     6.2883264 ]),
        #             np.array([-4.28457998, -0.29855966, -0.1728242 ,  0.20716901,  0.14665995,
        #                     6.16992025]),
        #             np.array([-4.3041423 , -0.18439027, -0.3369722 ,  0.05864697,  0.1940708 ,
        #                     6.33810247]),
        #             np.array([-4.29720816, -0.21711751, -0.29341776,  0.1103486 ,  0.18471341,
        #                     6.27970639]),
        #             np.array([-4.29630381, -0.22019268, -0.28930187,  0.11499665,  0.18405111,
        #                     6.27416506])
        # ]
        
        # for i in range(len(jnt_list) - 1): 
        # # for i in range(10,12):
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

        # robot.goto_given_conf(jnt_values=gth_jnt_values)
        # arm_mesh = robot.gen_meshmodel(alpha=0.25, rgb=[0,1,0])
        # arm_mesh.attach_to(base)

        # base.run()

        '''plot all ten seeds'''
      #   all = np.array([[-2.28174006, -0.81538117,  1.19386317,  2.05865428,  0.43705984,
      #    2.28134067],
      #  [-1.09182223, -0.37613506,  0.65194752,  2.03566165, -0.06560216,
      #    1.30068812],
      #  [ 2.03896468, -0.80453703,  0.4555546 , -2.50521392, -0.02695897,
      #    2.66708294],
      #  [-2.25123179,  0.26790624,  0.41754076,  2.56935772,  0.56570525,
      #    1.58339376],
      #  [ 2.39055764, -0.81823765,  1.40620545, -0.31424009, -0.83987396,
      #   -0.24316871],
      #  [-1.29629825, -0.6623173 ,  0.91126126, -2.49884589, -0.06363031,
      #   -0.51878046],
      #  [-2.07812926, -0.39027834,  0.94846866, -0.60043786, -0.31194216,
      #   -1.73719027],
      #  [-0.76055251, -0.29698602,  1.47302097,  0.02475748, -0.81266674,
      #    2.67726341],
      #  [ 1.1874773 , -0.43644674,  1.28478586,  0.3985929 , -0.99162754,
      #    0.63811196],
      #  [-1.33863606, -0.81385818,  1.55679558, -0.41555987, -0.47016753,
      #   -2.61743648],
      #  [-1.82639494, -0.51800126,  1.0228545 ,  2.63110363,  0.28908976,
      #    0.9617406 ],
      #  [ 0.38465326, -0.84226475,  0.83237044,  1.934335  , -0.24794586,
      #   -0.08512657],
      #  [ 2.19164566, -0.36379256,  0.46251145, -0.65166914, -0.48653241,
      #    0.65102106],
      #  [ 0.95486382,  0.22744828,  0.56073217,  0.37833981, -1.11150477,
      #    1.065389  ],
      #  [ 0.58635083, -0.59030197,  0.48422567,  1.85620294, -0.16009078,
      #   -0.51662785],
      #  [-2.04983673, -0.27846387,  0.69572134,  1.39371979,  0.15500409,
      #    2.72637595],
      #  [-1.19930389, -0.86256653,  1.5217576 ,  2.70120045,  0.2678634 ,
      #    0.79094309],
      #  [-1.11208318, -0.06787872,  0.45720109,  2.30594382,  0.03111745,
      #    1.20545655],
      #  [ 1.73455714, -0.85627217,  1.79232072,  0.19429733, -1.26100928,
      #    0.52676221],
      #  [-1.42190981, -0.8289869 ,  2.2201311 , -0.30906461, -1.04065872,
      #   -2.5286475 ]])

      #   for i in range(all.shape[0]):
      #       # robot.goto_given_conf(jnt_values=jnt)
      #       robot.goto_given_conf(jnt_values=all[i])
      #       arm_mesh = robot.gen_meshmodel(alpha=.5)
      #       arm_mesh.attach_to(base)

      #   robot.goto_given_conf(jnt_values=gth_jnt_values)
      #   arm_mesh = robot.gen_meshmodel(alpha=.3, rgb=[0,1,0])
      #   arm_mesh.attach_to(base)

      #   base.run()
        
        '''plot 2 seeds'''
        success_seed = [-1.11208318, -0.06787872,  0.45720109,  2.30594382,  0.03111745,
        1.20545655]
        fail_seed = [-2.28174006, -0.81538117,  1.19386317,  2.05865428,  0.43705984,
        2.28134067]
        
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
