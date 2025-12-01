# import wrs.robot_sim.manipulators.franka_research_3_arm.franka_research_3_arm as franka
# import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from wrs.robot_sim.robots.xarmlite6_wg.x6wg2 import XArmLite6WG2
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
import matplotlib.pyplot as plt
import wrs.motion.trajectory.totg as toppra

import utils

import cv2
import time
import yaml
import numpy as np
import zarr
import os
from tqdm import tqdm
from copy import copy

'''configurations'''
MAX_WAYPOINT = 200
MAX_TRY_TIME = 5.0
MAX_TRAJ_NUM = 100000
waypoint_interval = 0.01
current_dir = os.path.dirname(__file__)

'''Initialize the world and robot'''
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)
robot = XArmLite6WG2(enable_cc=True)
robot.gen_meshmodel().attach_to(base)

table_size = np.array([1.5, 1.5, 0.005])
table_pos  = np.array([0.0, 0.0, table_size[2]/2])
table = mcm.gen_box(xyz_lengths=table_size, pos=table_pos, rgb=np.array([0.6, 0.4, 0.2]), alpha=1)
table.attach_to(base)

paper_size = np.array([1.3, 1.3, 0.002])
paper_pos = table_pos.copy()
paper_pos[2] = table_pos[2] + table_size[2]/2 + paper_size[2]/2
paper = mcm.gen_box(xyz_lengths=paper_size, pos=paper_pos, rgb=np.array([1, 1, 1]), alpha=1)
paper.attach_to(base)


if __name__ == '__main__':
    '''init the parameters'''

    jnt_samples = [robot.rand_conf()]
    print(f"Total {len(jnt_samples)} joint configurations sampled.")
    print('--' * 100)

    '''simple visualization of the straight line path (20 directions)'''
    directions = fibonacci_sphere(20)
    for _ in tqdm(range(1)):
        pos_init, rotmat_init = robot.fk(jnt_values=jnt_samples[_])

        jnt_tracks = {}
        for dir_id, direction in enumerate(directions):
            jnt_list, pos = [jnt_samples[_]], pos_init.copy()

            for waypts in range(MAX_WAYPOINT):
                pos_try = pos + direction * waypoint_interval
                jnt = robot.ik(tgt_pos=pos_try, tgt_rotmat=rotmat_init, seed_jnt_values=jnt_list[-1])

                if jnt is None:
                    print(f"IK failed for direction {dir_id} at pos {pos_try}.")
                    break
                pos = pos_try
                jnt_list.append(jnt)

            jnt_tracks[dir_id] = jnt_list
            print(f"Generated {len(jnt_list)} waypoints for direction {dir_id}.")

    # visualize the generated joint paths
    path = []
    for dir_id in range(20):  # 可视化前5个方向
        path.extend(jnt_tracks[dir_id])
    utils.visualize_anime_path(robot, path)
    base.run()

    '''dataset generation'''
    dataset_name = os.path.join('/home/lqin/zarr_datasets', f'fr3_straight_direct10.zarr')
    store = zarr.DirectoryStore(dataset_name)
    root = zarr.group(store=store)
    print('dataset created in:', dataset_name)
    meta_group = root.create_group("meta")
    data_group = root.create_group("data")
    
    dof = robot.n_dof
    episode_ends_ds = meta_group.create_dataset("episode_ends", shape=(0,), chunks=(1,), dtype=np.float32, append=True)
    jnt_p = data_group.create_dataset("jnt_pos", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
    workspc_pos = data_group.create_dataset("position", shape=(0, 3), chunks=(1, 3), dtype=np.float32, append=True)
    workspc_rotmat = data_group.create_dataset("rotation", shape=(0, 9), chunks=(1, 9), dtype=np.float32, append=True)
    axis_id = data_group.create_dataset("axis", shape=(0,), chunks=(1,), dtype=np.float32, append=True)

    episode_ends_counter = 0

    '''Generate the straight line path in joint space (10 directions)'''
    directions = fibonacci_sphere(10)
    for _ in tqdm(range(len(jnt_samples))):
        pos_init, rotmat_init = robot.fk(jnt_values=jnt_samples[_])

        for dir_id, direction in enumerate(directions):
            jnt_list, pos = [jnt_samples[_]], pos_init.copy()
            
            for step in range(MAX_WAYPOINT):
                pos_try = pos + direction * waypoint_interval
                jnt = robot.ik(tgt_pos=pos_try, tgt_rotmat=rotmat_init, seed_jnt_values=jnt_list[-1])

                if jnt is None:
                    break
                if step == 0:
                    '''save the initial position and rotation matrix'''
                    jnt_p.append(np.array(jnt_list[-1]).reshape(1, dof))
                    axis_id.append(np.array([dir_id], dtype=np.float32))
                    workspc_pos.append(np.array(pos).reshape(1, 3))
                    workspc_rotmat.append(np.array(rotmat_init).reshape(1, 9))
                    episode_ends_counter += 1

                pos = pos_try
                jnt_list.append(jnt)
                jnt_p.append(np.array(jnt).reshape(1, dof))
                axis_id.append(np.array([dir_id], dtype=np.float32))
                p, r = robot.fk(jnt_values=jnt)
                workspc_pos.append(np.array(p).reshape(1, 3))
                workspc_rotmat.append(np.array(r).reshape(1, 9))
            episode_ends_counter += len(jnt_list) - 1
            episode_ends_ds.append(np.array([episode_ends_counter], dtype=np.float32))
            print(f"Generated {len(jnt_list)} waypoints for direction {dir_id}.")
