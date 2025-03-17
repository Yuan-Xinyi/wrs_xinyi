import wrs.robot_con.xarm_lite6.xarm_lite6_x as xarm_x
import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm_s
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
import wrs.motion.probabilistic.rrt as rrt

import cv2
import time
import yaml
import numpy as np
import zarr
import os
current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))

base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)
robot = xarm_s.XArmLite6(enable_cc=True)
rrtc = rrt.RRT(robot)

start_conf = robot.rand_conf()
goal_conf = robot.rand_conf()
print(repr(start_conf))
print(repr(goal_conf))

robot.goto_given_conf(jnt_values=start_conf)
robot.gen_meshmodel(rgb=rm.const.steel_blue, alpha=.3).attach_to(base)
robot.goto_given_conf(jnt_values=goal_conf)
robot.gen_meshmodel(rgb=[0,1,0], alpha=.3).attach_to(base)
# base.run()


path = rrtc.plan(start_conf=start_conf,
                 goal_conf=goal_conf,
                 ext_dist=.1,
                 max_time=300,
                 animation=True)
print(path)