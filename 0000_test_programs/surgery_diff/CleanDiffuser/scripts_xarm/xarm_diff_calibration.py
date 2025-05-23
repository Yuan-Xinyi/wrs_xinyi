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

'''load the config file'''
config_file = os.path.join(current_file_dir, 'gendata_config.yaml')
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

'''init the robot'''
robot_x = xarm_x.XArmLite6X(ip = '192.168.1.190')

print('current joint:', repr(robot_x.get_jnt_values()))
print('current pose:', repr(robot_x.get_pose()))

'''init the camera'''
rgb_camera = []
cam_idx = config['camera_idx']
rgb_camera.append(cv2.VideoCapture(cam_idx[0]))
rgb_camera.append(cv2.VideoCapture(cam_idx[1]))

'''img capture test'''
for id, camera in enumerate(rgb_camera):
    ret, frame = camera.read()

    '''crop the image'''
    crop_size = config['crop_size']
    h, w, _ = frame.shape
    crop_center = config['crop_center']
    center_x, center_y = crop_center[id][0], crop_center[id][1]
    x1 = max(center_x - crop_size // 2, 0)
    x2 = min(center_x + crop_size // 2, w)
    y1 = max(center_y - crop_size // 2, 0)
    y2 = min(center_y + crop_size // 2, h)
    cropped = frame[y1:y2, x1:x2]

    if ret:
        cv2.imshow('captured frame', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('no image captured')

    time.sleep(1)


'''init the robot sim'''
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)
robot_s = xarm_s.XArmLite6(enable_cc=True)


'''print the pos and q'''
print('current joint:', repr(robot_x.get_jnt_values()))
print('current pose:', repr(robot_x.get_pose()))