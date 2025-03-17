from wrs import wd, rm, mcm, mgm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.rs007l.rs007l as rs007l
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.manipulators.ur3e.ur3e as ur3e
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi
import wrs.robot_sim.robots.cobotta_pro1300.cobotta_pro1300 as cbtpro1300
import wrs.robot_sim.robots.cobotta_pro900.cobotta_pro900_spine as cbtpro900


import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)


robot_list = ['yumi', 'cbt','ur3', 'cbtpro1300']
# robot_list = ['ur3']

if __name__ == '__main__':
# while True:

    for robot in robot_list:
        if robot == 'yumi':
            robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
            not_shift_list = [2,4]
            length = 0.05
            rot_axis_pos = 0.04
        elif robot == 'cbt':
            robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
            not_shift_list = [3]
            length = 0.08
            rot_axis_pos = 0.04
        elif robot == 'ur3':
            robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), enable_cc=True)
            not_shift_list = [3]
        elif robot == 'cbtpro1300':
            robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)
        else:
            print("Invalid robot name")

        color_list = {
            "cobotta": np.array([1.0, 0.0, 0.0]),  # 红色
            "cobotta_pro_1300": np.array([0.0, 1.0, 0.0]),  # 绿色
            "sglarm_yumi": np.array([0.0, 0.0, 1.0]),  # 蓝色
            "ur3": np.array([1.0, 1.0, 0.0])   # 黄色
        }
        jnt = np.zeros(robot.n_dof)
        print(repr(jnt))
        robot.goto_given_conf(jnt)
        arm_mesh = robot.gen_meshmodel(alpha=.2, rgb=color_list[robot.name])
        arm_mesh.attach_to(base)
        tmp_arm_stick = robot.gen_stickmodel(toggle_flange_frame=False, toggle_jnt_frames=False)
        tmp_arm_stick.attach_to(base)

        portion = 0.75
        length = 0.1
        rot_axis_pos = 0.05
        radius = 0.03

        for i in range(robot.n_dof):
            if robot.name in ['cobotta', 'cobotta_pro_1300', 'sglarm_yumi']:
                if i not in not_shift_list:
                    mgm.gen_stick(spos = robot.manipulator.jlc.jnts[i].gl_pos_0, 
                                        epos = robot.manipulator.jlc.jnts[i].gl_pos_0 + length*(robot.manipulator.jlc.jnts[i].gl_motion_ax),
                                        radius=.0025, rgb=color_list[robot.name]).attach_to(base)
                    mgm.gen_stick(spos = robot.manipulator.jlc.jnts[i].gl_pos_0, 
                                        epos = robot.manipulator.jlc.jnts[i].gl_pos_0 - length*(robot.manipulator.jlc.jnts[i].gl_motion_ax),
                                        radius=.0025, rgb=color_list[robot.name]).attach_to(base)
                    mgm.gen_circarrow(axis = robot.manipulator.jlc.jnts[i].gl_motion_ax, 
                                    center = robot.manipulator.jlc.jnts[i].gl_pos_0 + rot_axis_pos*(robot.manipulator.jlc.jnts[i].gl_motion_ax), 
                                    portion=portion, major_radius=radius, rgb=color_list[robot.name]).attach_to(base)
                else:
                    mgm.gen_stick(spos = robot.manipulator.jlc.jnts[i].gl_pos_0, 
                                        epos = robot.manipulator.jlc.jnts[i].gl_pos_0 + length*(robot.manipulator.jlc.jnts[i].gl_motion_ax),
                                        radius=.0025, rgb=color_list[robot.name]).attach_to(base)
                    mgm.gen_stick(spos = robot.manipulator.jlc.jnts[i].gl_pos_0, 
                                        epos = robot.manipulator.jlc.jnts[i].gl_pos_0 - length*(robot.manipulator.jlc.jnts[i].gl_motion_ax),
                                        radius=.0025, rgb=color_list[robot.name]).attach_to(base)
                    mgm.gen_circarrow(axis = robot.manipulator.jlc.jnts[i].gl_motion_ax, 
                                    center = robot.manipulator.jlc.jnts[i].gl_pos_0, 
                                    portion=portion, major_radius=radius, rgb=color_list[robot.name]).attach_to(base)
            else:
                mgm.gen_stick(spos = robot.jlc.jnts[i].gl_pos_0, 
                                    epos = robot.jlc.jnts[i].gl_pos_0 + length*(robot.jlc.jnts[i].gl_motion_ax),
                                    radius=.0025, rgb=color_list[robot.name]).attach_to(base)
                mgm.gen_stick(spos = robot.jlc.jnts[i].gl_pos_0, 
                                    epos = robot.jlc.jnts[i].gl_pos_0 - length*(robot.jlc.jnts[i].gl_motion_ax),
                                    radius=.0025, rgb=color_list[robot.name]).attach_to(base)
                mgm.gen_circarrow(axis = robot.jlc.jnts[i].gl_motion_ax, 
                                center = robot.jlc.jnts[i].gl_pos_0 + rot_axis_pos*(robot.jlc.jnts[i].gl_motion_ax), 
                                portion=portion, major_radius=radius, rgb=color_list[robot.name]).attach_to(base)

        # robot.show_cdprim()
    base.run()

