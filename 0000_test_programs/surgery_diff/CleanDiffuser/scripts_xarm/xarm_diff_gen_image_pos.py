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
trails = 20


'''load the config file'''
config_file = os.path.join(current_file_dir, 'gendata_config.yaml')
with open(config_file, "r") as file:
    config = yaml.safe_load(file)


'''init'''
robot_x = xarm_x.XArmLite6X(ip = '192.168.1.190')
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)
robot_s = xarm_s.XArmLite6(enable_cc=True)
rgb_camera = []
cam_idx = config['camera_idx']
rgb_camera.append(cv2.VideoCapture(cam_idx[0]))
rgb_camera.append(cv2.VideoCapture(cam_idx[1]))


'''init the data storage'''
dataset_dir = os.path.join(parent_dir, 'datasets')
dataset_name = os.path.join(dataset_dir, 'surgery.zarr')
store = zarr.DirectoryStore(dataset_name)
root = zarr.group(store=store)
print('dataset created in:', dataset_name)

meta_group = root.create_group("meta")
data_group = root.create_group("data")
episode_ends = meta_group.create_dataset("episode_ends", data=0, dtype=np.int32)
state = data_group.create_dataset("state", shape=(0, 7), chunks=(1, 7), dtype=np.float32, append=True)
action = data_group.create_dataset("action", shape=(0, 7), chunks=(1, 7), dtype=np.float32, append=True)
img = data_group.create_dataset("img", shape=(0, 96, 192, 3), chunks=(1, 96, 192, 3), dtype=np.uint8, append=True)

'''start data generation'''
jnt_list = np.load(os.path.join(dataset_dir, 'jnt_array.npy'))

for _ in range(trails):
    for id, jnt in enumerate(jnt_list):
        robot_x.move_j(jnt)
        img = np.zeros((2, 96, 96, 3), dtype=np.uint8)

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

            if id == 0:
                img[id] = frame[y1:y2, x1:x2]
            else:
                cropped = frame[y1:y2, x1:x2]
                img[id] = np.rot90(cropped, 2)
                
        concatenated_image = np.hstack(img)
        pos, rot_mat = robot_s.fk(jnt)
        rot_quat = rm.quaternion_from_rotmat(rot_mat)
        agent_pos = np.concatenate((pos, rot_quat))

        agent_pos_ds.append(np.expand_dims(agent_pos, axis=0))  # (1, 7)
        image_ds.append(np.expand_dims(concatenated_image, axis=0))  # (1, 96, 192, 3)


