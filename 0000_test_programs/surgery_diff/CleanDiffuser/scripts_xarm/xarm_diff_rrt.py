import wrs.robot_con.xarm_lite6.xarm_lite6_x as xarm_x
import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm_s
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
from wrs.motion.probabilistic.rrt import RRT

import cv2
import time
import yaml
import numpy as np
import zarr
import os

current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))

'''load the config file'''
config_file = os.path.join(current_file_dir, 'gendata_config.yaml')
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

'''init the actual robot'''
robot_x = xarm_x.XArmLite6X(ip = '192.168.1.190')

'''init the robot sim'''
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)
robot_s = xarm_s.XArmLite6(enable_cc=True)
# robot_x.move_j(config['start_jnt'])
# robot_x.move_j(config['end_jnt'])

rrtc = RRT(robot_s)
start_conf = config['start_jnt']
end_conf = config['end_jnt']

robot_s.goto_given_conf(jnt_values=start_conf)
robot_s.gen_meshmodel(rgb=rm.const.steel_blue, alpha=.3).attach_to(base)
robot_s.goto_given_conf(jnt_values=end_conf)
robot_s.gen_meshmodel(rgb=[0,1,0], alpha=.3).attach_to(base)

# base.run()

path = rrtc.plan(start_conf=start_conf,
                    goal_conf=end_conf,
                    ext_dist=.2,
                    max_time=300,
                    animation=True)

print('path:', path)