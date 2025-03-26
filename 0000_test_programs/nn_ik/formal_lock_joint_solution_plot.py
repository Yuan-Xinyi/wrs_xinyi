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
        jnt_values = [-0.89298267, -0.99745798,  2.20644947, -1.90284232, -0.62446831,
       -0.94061137]
        
        # robot.goto_given_conf(jnt_values=jnt_values)
        # arm_mesh = robot.gen_meshmodel(alpha=0.2, rgb=[0, 1, 0])
        # arm_mesh.attach_to(base)

        tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
        mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        
        '''success'''
        # jnt_list = [[-1.454441111111111, -0.6981321250000001, 2.139275042857143, -1.2715971428571429, -0.9890201666666668, -1.780236], [-1.222086878338565, -1.1701142783611538, 2.3715757181934993, -1.6173777053544494, -0.7726805891835274, -1.2222532140856952], [-1.0953275735222732, -1.061311049002143, 2.226727322784939, -1.8333548544984961, -0.8008473701797493, -0.9492613272231316], [-0.7095031148464365, -0.960300276393524, 2.197945917736475, -1.9869817797883254, -0.45547655665452214, -0.9274960480892287], [-0.8667322945912056, -0.9972303099377424, 2.207611258977268, -1.909814869591795, -0.6004945817751378, -0.9421056601976645]]

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
        # tgt_pos, tgt_rotmat = robot.fk(jnt_values=jnt_list[-1])
        # mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        # base.run()

        '''fail'''
        # jnt_list = [[0.8726646666666666, -0.34906625000000013, 1.8350890857142859, 1.2715971428571429, -0.9890201666666668, 1.7802360000000004], [0.518242863568018, -0.922163909458882, 1.7326400995218987, 1.8967977572096923, -0.6228689512603456, 1.0036273723201714], [1.966289513324845, -0.47937499468709, 1.3328619191120696, 0.583911892734549, -1.8911306675590702, 1.4512574569182826], [1.7966230414040718, -0.8877408319841692, 1.6987837272703554, 0.8780489822393712, -1.779518672235449, 1.3103975682673572], [1.748323247999939, -0.8911059424270954, 1.5503179737644897, 1.0307246849224851, -1.6734674566687602, 0.9604947820764266], [1.4727137883218058, -0.8707656475924307, 1.4553041144341485, 1.3456235045959766, -1.572699969351991, 0.7624230551920557], [2.4036170093151554, -0.787239460308739, 1.5314819096891605, 0.5636798620990665, -2.1282702519971908, 0.7674476201817555], [2.237074172280133, -0.9376666549905723, 1.5988762486705492, 0.6715004086563674, -2.0048433885952117, 0.5593356486387248], [2.0597181925442207, -0.9253768576962774, 1.5296261151338426, 0.8305575166987407, -1.887565950274984, 0.6028152870948592]]

        # for i in range(len(jnt_list) - 1): 
        #     jnt1 = jnt_list[i]
        #     jnt2 = jnt_list[i + 1]

        #     s_pos, _ = robot.fk(jnt_values=jnt1)
        #     e_pos, _ = robot.fk(jnt_values=jnt2)
            
        #     mgm.gen_arrow(spos=s_pos, epos=e_pos, stick_radius=.0025, rgb=[0,0,0]).attach_to(base)

        #     robot.goto_given_conf(jnt_values=jnt1)
        #     arm_mesh = robot.gen_meshmodel(alpha=0.15, rgb=[1, 0, 0])
        #     arm_mesh.attach_to(base)

        #     if i == 6:
        #         tmp_arm_stick = robot.gen_stickmodel(toggle_flange_frame=True)
        #         tmp_arm_stick.attach_to(base)
        
        # robot.goto_given_conf(jnt_values=jnt_list[-1])
        # final_arm_mesh = robot.gen_meshmodel(alpha=0.15, rgb=[1, 0, 0])
        # final_arm_mesh.attach_to(base)

        # tgt_pos, tgt_rotmat = robot.fk(jnt_values=jnt_list[-1])
        # mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

        # base.run()

        '''plot tcp nearest ten seeds'''
        # jnt_list = np.array([14502,  7100,  8862,  7704,  2064,  3222, 15013,  3342, 12740, 13344])
        jnt_list = np.array([14502,  7100,  8862,  7704,  2064,  3222, 15013,  3342, 12740,
       13344, 20653, 18984,  8982,  9923, 26293,  2788, 15563,  8487,
       18380,  2847, 18260, 12620, 22237, 14127,  2131, 23900, 24624,
        1081, 39518,  8428,  6980, 21203,  7771, 19588, 29540, 13948,
       14622, 25228, 24478, 31933, 19767, 12241, 27877,  6721,  8308,
        1340,  6601, 24020, 13411, 14068, 30264, 12961,  7824, 26843,
        1925, 13291, 18931,  7565,  9323,   961,  2668,  3567, 24504,
       12361,  7651, 25407, 30118, 20483, 14843, 24571,  9207, 33517,
       13205, 37573, 20262, 26123, 23618, 18864,  9203, 14963,  7321,
       29258, 21497, 37453, 14847, 17978,  2011, 19051, 31813, 19708,
       34898,  1415, 18845, 32483,  4232, 24358, 26173, 25287, 29660,
       20773, 29998, 31763, 27757, 13464, 18001, 13224, 10043, 33397,
       27137, 18718,  9032,  3563, 22117, 19647, 14672, 31047,  7055,
        3392, 35638, 39037, 39157, 20487, 20603, 20533, 12954,  1681,
       20312,  8607, 16477, 18594, 10456, 12575, 16096, 14007,  6935,
        8529, 24485, 35257,  4816,  7584, 18215, 32777, 26007, 21736,
       20367, 24691,  9872, 12695,  1295, 24234, 25348, 25952, 18725,
       15683, 14893, 14727, 23855, 18714, 19689, 35300, 14169, 27376,
        8367,  2045, 25329, 14049, 26127, 26413,  4936, 19434,  2746,
        3512, 19104,  8386, 29874, 26243, 13085, 14247, 27997, 30969,
       12860, 38417,  1944,  8409, 14026, 18335, 19314,  8742, 24954,
         719, 13674, 15512,  6359, 10576,   699,  9253,  6339,  9152,
       19809,  2851])
        for jnt in jnt_list:
            # robot.goto_given_conf(jnt_values=jnt)
            robot.goto_given_conf(jnt_values=kdt_jnt_data[jnt])
            arm_mesh = robot.gen_meshmodel(alpha=.15)
            arm_mesh.attach_to(base)
        
        robot.goto_given_conf(jnt_values=jnt_values)
        arm_mesh = robot.gen_meshmodel(alpha=0.9, rgb=[0.133, 0.545, 0.133])
        arm_mesh.attach_to(base)
        base.run()

        '''plot the most linear 10 seeds'''
        # # jnt_list = np.array([ 9923,  3512,  9032,  4232,  9152,  1295,  9207,  6935, 15563,
        # # 9872,  1340])
        # jnt_list = np.array([ 9923,  3512,  9032,  4232,  9152,  1295,  9207,  6935, 15563,
        # 9872,  1340,  1081,  3392, 14847,  8308,  6980, 15683,  9253,
        # 6339,  7100])
        # fail_idx = 9923
        # success_idx = 1340
        # for jnt in jnt_list:
        #     if jnt == fail_idx:
        #         robot.goto_given_conf(jnt_values=kdt_jnt_data[jnt])
        #         arm_mesh = robot.gen_meshmodel(alpha=.25, rgb=[1,0,0])
        #         arm_mesh.attach_to(base)

        #         tgt_pos, tgt_rotmat = robot.fk(jnt_values = kdt_jnt_data[jnt])
        #         mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

        #     elif jnt == success_idx:
        #         robot.goto_given_conf(jnt_values=kdt_jnt_data[jnt])
        #         arm_mesh = robot.gen_meshmodel(alpha=.25, rgb=[0,0,1])
        #         arm_mesh.attach_to(base)

        #         tgt_pos, tgt_rotmat = robot.fk(jnt_values = kdt_jnt_data[jnt])
        #         mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        #     else:
        #         robot.goto_given_conf(jnt_values=kdt_jnt_data[jnt])
        #         arm_mesh = robot.gen_meshmodel(alpha=.15)
        #         arm_mesh.attach_to(base)

        # base.run()

        
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