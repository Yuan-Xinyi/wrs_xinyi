import wrs.robot_con.xarm_lite6.xarm_lite6_x as xarm_x
import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm_s
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm

import cv2
import time
import yaml
import numpy as np

'''load the config file'''
with open("0000_test_programs/surgery_diff/gendata_config.yaml", "r") as file:
    config = yaml.safe_load(file)

'''init the robot'''
robot_x = xarm_x.XArmLite6X(ip = '192.168.1.190')
end_j = robot_x.get_jnt_values()
end_p = robot_x.get_pose()
# print('current joint:', repr(robot_x.get_jnt_values()))
# print('current pose:', repr(robot_x.get_pose()))

'''init the camera'''
rgb_camera = []
cam_idx = config['camera_idx']
rgb_camera.append(cv2.VideoCapture(cam_idx[0]))
rgb_camera.append(cv2.VideoCapture(cam_idx[1]))

'''img capture test'''
# for camera in rgb_camera:
#     ret, frame = camera.read()
#     if ret:
#         cv2.imshow('captured frame', frame)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print('no image captured')

#     time.sleep(1)

'''init the robot sim'''
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)
robot_s = xarm_s.XArmLite6(enable_cc=True)
rrtc = RRTConnect(robot_s)

