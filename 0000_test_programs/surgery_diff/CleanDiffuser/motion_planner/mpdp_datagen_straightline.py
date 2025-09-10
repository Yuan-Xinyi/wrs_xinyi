# import wrs.robot_sim.manipulators.franka_research_3_arm.franka_research_3_arm as franka
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
import wrs.motion.probabilistic.rrt_connect_welding as rrtc_welding

from wrs.motion.trajectory.quintic import QuinticSpline
import matplotlib.pyplot as plt
import wrs.motion.trajectory.totg as toppra

import cv2
import time
import yaml
import numpy as np
import zarr
import os
from tqdm import tqdm

def visualize_anime_path(robot, path):
    class Data(object):
        def __init__(self):
            self.counter = 0
            self.path = path
            self.current_model = None
            self.current_frame = None  # 用于记录并移除上一帧的末端坐标系

    anime_data = Data()

    def update(robot, anime_data, task):
        if anime_data.counter >= len(anime_data.path):
            if anime_data.current_model:
                anime_data.current_model.detach()
            if anime_data.current_frame:
                anime_data.current_frame.detach()
            anime_data.counter = 0
            return task.again

        if anime_data.current_model:
            anime_data.current_model.detach()
        if anime_data.current_frame:
            anime_data.current_frame.detach()

        conf = anime_data.path[anime_data.counter]
        robot.goto_given_conf(conf)
        anime_data.current_model = robot.gen_meshmodel(alpha=1.0)
        anime_data.current_model.attach_to(base)

        ee_pos, ee_rotmat = robot.fk(conf)
        anime_data.current_frame = mcm.mgm.gen_frame(pos=ee_pos, rotmat=ee_rotmat)
        anime_data.current_frame.attach_to(base)

        anime_data.counter += 1
        return task.again

    def start_animation(task):
        taskMgr.doMethodLater(0.08, update, "update",
                              extraArgs=[robot, anime_data],
                              appendTask=True)
        return task.done

    taskMgr.doMethodLater(1.0, start_animation, "start_animation_delay")
    base.run()


def fibonacci_sphere(samples=20):
    """在球面上均匀生成 `samples` 个单位方向向量"""
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # 黄金角
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)


if __name__ == '__main__':
    MAX_WAYPOINT = 200
    MAX_TRY_TIME = 5.0
    MAX_TRAJ_NUM = 100000
    waypoint_interval = 0.01

    '''init the parameters'''
    from copy import copy
    current_file_dir = os.path.dirname(__file__)

    '''Initialize the world and robot'''
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = franka.FrankaResearch3(enable_cc=True)

    jnt_samples = np.load('cvt_joint_samples_fr3_scale5.npy')
    print(f"Total {len(jnt_samples)} joint configurations sampled.")
    print('--' * 100)

    '''simple visualization of the straight line path (20 directions)'''
    # directions = fibonacci_sphere(20)
    # for _ in tqdm(range(1)):  # 只看第一个初始点做动画
    #     pos_init, rotmat_init = robot.fk(jnt_values=jnt_samples[_])

    #     jnt_tracks = {}
    #     for dir_id, direction in enumerate(directions):
    #         jnt_list, pos = [jnt_samples[_]], pos_init.copy()

    #         for _ in range(MAX_WAYPOINT):
    #             pos_try = pos + direction * waypoint_interval
    #             jnt = robot.ik(tgt_pos=pos_try, tgt_rotmat=rotmat_init, seed_jnt_values=jnt_list[-1])

    #             if jnt is None:
    #                 print(f"IK failed for direction {dir_id} at pos {pos_try}.")
    #                 break
    #             pos = pos_try
    #             jnt_list.append(jnt)

    #         jnt_tracks[dir_id] = jnt_list
    #         print(f"Generated {len(jnt_list)} waypoints for direction {dir_id}.")

    # # visualize the generated joint paths
    # path = []
    # for dir_id in range(20):  # 可视化前5个方向
    #     path.extend(jnt_tracks[dir_id])
    # visualize_anime_path(robot, path)
    # # base.run()

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
