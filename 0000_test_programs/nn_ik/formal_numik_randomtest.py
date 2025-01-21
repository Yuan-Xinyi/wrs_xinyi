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

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

'''define robot'''
robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
# robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
# robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), ik_solver='d' ,enable_cc=True)
# robot = rs007l.RS007L(pos=rm.vec(0.1, .3, .5), enable_cc=True)



nupdate = 10000

if __name__ == '__main__':
# while True:
    success_num = 0
    time_list = []
    pos_err_list = []
    rot_err_list = []

    for i in tqdm(range(nupdate)):
        jnt_values = robot.rand_conf()
        tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
        tic = time.time()
        result = robot.ik(tgt_pos, tgt_rotmat, best_sol_num = 1)
        toc = time.time()
        time_list.append(toc-tic)
        if result is not None:
            success_num += 1

            # calculate the pos error and rot error
            pred_pos, pred_rotmat = robot.fk(jnt_values=result)
            pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
            pos_err_list.append(pos_err)
            rot_err_list.append(rot_err)
             
    print('=============================')
    print(f'Average time: {np.mean(time_list) * 1000:.2f} ms')
    print(f'success rate: {success_num / nupdate * 100:.2f}%')
    print(f'Average position error: {np.mean(pos_err_list)}')
    print(f'Average rotation error: {np.mean(rot_err_list)*180/np.pi}')
    print('=============================')
    # plt.plot(range(nupdate), time_list)
    # plt.show()


    # arm_mesh = robot.gen_meshmodel(alpha=.3)
    # arm_mesh.attach_to(base)
    # tmp_arm_stick = robot.gen_stickmodel(toggle_flange_frame=True)
    # tmp_arm_stick.attach_to(base)
    # mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

    # if jnt_values is not None:
    #     print('jnt degree values: ',np.degrees(result))
    #     robot.goto_given_conf(jnt_values=result)
    # arm_mesh = robot.gen_meshmodel(alpha=.3)
    # arm_mesh.attach_to(base)
    # tmp_arm_stick = robot.gen_stickmodel(toggle_flange_frame=True)
    # tmp_arm_stick.attach_to(base)
    # base.run()

