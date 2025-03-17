import wrs.robot_con.xarm_lite6.xarm_lite6_x as xarm_x
import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm_s
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm

import cv2
import time
import yaml
import numpy as np
import zarr
import os
current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))

'''init'''
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)
robot_s = xarm_s.XArmLite6(enable_cc=True)

'''load the config file'''
config_file = os.path.join(current_file_dir, 'config.yaml')
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dataset_dir = os.path.join(parent_dir, 'datasets')
dataset_name = os.path.join(dataset_dir, 'xarm_mp.zarr')
store = zarr.DirectoryStore(dataset_name)
root = zarr.group(store=store)