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



def intrerp_pos_path(cube_center, cube_size, offset, interp_axis, interp_num):
    pos_list = []
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if interp_axis not in axis_map:
        raise ValueError("interp_axis must be 'x', 'y', or 'z'")

    axis_idx = axis_map[interp_axis]
    base_pos = cube_center + np.array([-cube_size/2, cube_size/2, cube_size/2])
    offset_vec = np.zeros(3)
    offset_vec[1] = offset
    base_pos += offset_vec

    for i in range(interp_num):
        pos = base_pos.copy()
        pos[axis_idx] += (i * cube_size / interp_num) - offset
        pos_list.append(pos)

    return pos_list
    
def gen_jnt_list_from_pos_list(pos_list, robot, obstacle_list, base,
                                max_try_time=5.0, check_collision=True, visualize=False):
    jnt_list = []
    success_count = 0

    for pos in pos_list:
        jnt = None
        start_time = time.time()
        while jnt is None and time.time() - start_time < max_try_time:
            try:
                rotmat = rm.rotmat_from_euler(np.pi/2,0,0)
                j = robot.ik(tgt_pos=pos, tgt_rotmat=rotmat, seed_jnt_values = jnt_list[-1] if jnt_list else None)
                if j is None:
                    continue
                # robot.goto_given_conf(j)
                if check_collision and robot.cc.is_collided(obstacle_list=obstacle_list):
                    continue
                jnt = j
                success_count += 1
                if visualize:
                    mcm.mgm.gen_frame(pos=pos, rotmat=rotmat).attach_to(base)
                    # robot.gen_meshmodel(alpha=.2).attach_to(base)
                break
            except:
                break
        jnt_list.append(jnt)

    print(f"{'-'*40}\nSuccessfully solved IK for {success_count} / {len(pos_list)} positions.\n{'-'*40}")
    return [j for j in jnt_list if j is not None], success_count

def plot_joint_trajectories(jnt_list):
    jnt_array = np.array(jnt_list)
    if jnt_array.ndim != 2 or jnt_array.shape[1] != 7:
        raise ValueError("Expected jnt_list to be of shape (n, 7)")

    fig, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    fig.suptitle("Joint Value Trajectories (Scatter)", fontsize=16)

    for i in range(7):
        axes[i].scatter(range(len(jnt_array)), jnt_array[:, i], s=20)
        axes[i].set_ylabel(f"Joint {i+1}")
        axes[i].grid(True)

    axes[-1].set_xlabel("Index")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def discretize_joint_space(robot, n_intervals=None):
    sampled_jnts = []
    if n_intervals is None:
        n_intervals = np.linspace(6, 5, robot.n_dof, endpoint=False)
    print(f"Sampling Joint Space using the following joint granularity: {n_intervals.astype(int)}...")
    for i in range(robot.n_dof):
        sampled_jnts.append(
            np.linspace(robot.jnt_ranges[i][0], robot.jnt_ranges[i][1], int(n_intervals[i]+2))[1:-1])
    grid = np.meshgrid(*sampled_jnts)
    sampled_qs = np.vstack([x.ravel() for x in grid]).T
    
    return sampled_qs

def partiallydiscretize_joint_space(robot, n_intervals=None):
    sampled_jnts = []
    if n_intervals is None:
        n_intervals = np.linspace(6, 4, robot.n_dof - 1, endpoint=False)
        # n_intervals = np.linspace(4, 2, robot.n_dof - 1, endpoint=False)
    print(f"Sampling Joint Space using the following joint granularity (excluding last DOF): {n_intervals.astype(int)}...")

    # 对前 n-1 个关节离散采样
    for i in range(robot.n_dof - 1):
        sampled_jnts.append(
            np.linspace(robot.jnt_ranges[i][0], robot.jnt_ranges[i][1], int(n_intervals[i] + 2))[1:-1]
        )

    # 构造网格
    grid = np.meshgrid(*sampled_jnts)
    base_qs = np.vstack([x.ravel() for x in grid]).T

    # 最后一维填 0.0 或 np.nan
    last_column = np.zeros((base_qs.shape[0], 1))  # 或 np.full((..., 1), np.nan)
    sampled_qs = np.hstack((base_qs, last_column))

    return sampled_qs

def generate_random_curve_pos_list(num_points=32, num_control_points=6):
    """
    生成任意三维空间曲线的 pos_list 基于随机控制点的 B-spline 插值。

    参数：
        num_points: 曲线上采样的点数量
        num_control_points: 控制点数量
        seed: 随机种子，确保可重复

    返回：
        pos_list: np.ndarray, shape=(num_points, 3)
    """
    from scipy.interpolate import splprep, splev
    
    # 控制点范围可根据你的工作空间调整
    control_points = np.random.uniform(
        low=[0.4, -0.1, 0.2],
        high=[0.6, 0.1, 0.4],
        size=(num_control_points, 3)
    )

    # 拟合三维B样条曲线
    tck, _ = splprep(control_points.T, s=0)
    t_vals = np.linspace(0, 1, num_points)
    pos_list = np.array(splev(t_vals, tck)).T  # shape = (num_points, 3)

    return pos_list



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

    # jnt_samples = discretize_joint_space(robot)
    jnt_samples = partiallydiscretize_joint_space(robot)
    print(f"Total {len(jnt_samples)} joint configurations sampled.")
    print('--' * 100)

    '''curve line path generation'''
    n_points = 32
    theta = np.linspace(0, np.pi, n_points)  # 1/4 圆弧
    radius = 0.1
    center = np.array([0.4, 0.0, 0.3])

    pos_list = generate_random_curve_pos_list()
    
    curve = np.array(pos_list)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], label="3D Polynomial Curve")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    jnt_list, success = gen_jnt_list_from_pos_list(
        pos_list=pos_list,
        robot=robot,
        obstacle_list=[],      # 如有障碍物可填
        base=base,
        max_try_time=3.0,
        check_collision=False,  # 初期可以先关掉避免卡住
        visualize=True
    )

    # robot.goto_given_conf(jnt_list[0])
    # robot.gen_meshmodel(rgb=[0,1,0], alpha=0.3).attach_to(base)
    # robot.goto_given_conf(jnt_list[-1])
    # robot.gen_meshmodel(rgb=[0,0,1], alpha=0.3).attach_to(base)
    # base.run()
    visualize_anime_path(robot, jnt_list)

    '''simple visualization of the straight line path'''
    # # for _ in tqdm(range(len(jnt_samples))):
    # for _ in tqdm(range(1)):
    #     pos_init, rotmat_init = robot.fk(jnt_values=jnt_samples[_])
    #
    #     jnt_tracks = {}
    #     for axis in ['x', 'y', 'z']:
    #         axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    #         jnt_list, pos = [jnt_samples[_]], pos_init.copy()
    #
    #         for _ in range(MAX_WAYPOINT):
    #             pos_try = pos.copy(); pos_try[axis_idx] += waypoint_interval
    #             # print(f"Trying to solve IK for position {pos_try} on axis {axis}...")
    #             jnt = robot.ik(tgt_pos=pos_try, tgt_rotmat=rotmat_init, seed_jnt_values=jnt_list[-1])
    #
    #             if jnt is None:
    #                 print(f"IK failed for position {pos_try} on axis {axis}.")
    #                 break
    #             pos = pos_try
    #             jnt_list.append(jnt)
    #             # mcm.mgm.gen_frame(pos=pos, rotmat=rotmat_init).attach_to(base)
    #             # robot.goto_given_conf(jnt)
    #             # robot.gen_meshmodel(alpha=0.3).attach_to(base)
    #
    #         jnt_tracks[axis] = jnt_list
    #         print(f"Generated {len(jnt_list)} waypoints for axis {axis}.")
    #
    # # visualize the generated joint paths
    # path = []
    # for axis in ['x', 'z']:
    #     path.extend(jnt_tracks[axis])
    # visualize_anime_path(robot, path)
    # # base.run()

    '''dataset generation'''
    # # dataset_name = os.path.join('/home/lqin/zarr_datasets', f'straight_jntpath_partially.zarr')
    # dataset_name = os.path.join('/home/lqin/zarr_datasets', f'0607_simple_straight.zarr')
    # store = zarr.DirectoryStore(dataset_name)
    # root = zarr.group(store=store)
    # print('dataset created in:', dataset_name)    
    # meta_group = root.create_group("meta")
    # data_group = root.create_group("data")
    
    # dof = robot.n_dof
    # episode_ends_ds = meta_group.create_dataset("episode_ends", shape=(0,), chunks=(1,), dtype=np.float32, append=True)
    # jnt_p = data_group.create_dataset("jnt_pos", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
    # workspc_pos = data_group.create_dataset("position", shape=(0, 3), chunks=(1, 3), dtype=np.float32, append=True)
    # workspc_rotmat = data_group.create_dataset("rotation", shape=(0, 9), chunks=(1, 9), dtype=np.float32, append=True)
    # axis_id = data_group.create_dataset("axis", shape=(0,), chunks=(1,), dtype=np.float32, append=True)

    # episode_ends_counter = 0
    
    # '''Generate the straight line path in joint space'''
    # for _ in tqdm(range(len(jnt_samples))):
    # # for _ in tqdm(range(1)):
    #     pos_init, rotmat_init = robot.fk(jnt_values=jnt_samples[_])

    #     for axis in ['x', 'y', 'z']:
    #         axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    #         jnt_list, pos = [jnt_samples[_]], pos_init.copy()
            
    #         for _ in range(MAX_WAYPOINT):
    #             pos_try = pos.copy(); pos_try[axis_idx] += waypoint_interval
    #             # print(f"Trying to solve IK for position {pos_try} on axis {axis}...")
    #             jnt = robot.ik(tgt_pos=pos_try, tgt_rotmat=rotmat_init, seed_jnt_values=jnt_list[-1])

    #             if jnt is None:
    #                 # print(f"IK failed for position {pos_try} on axis {axis}.")
    #                 break
    #             if _ == 0:
    #                 '''save the initial position and rotation matrix'''
    #                 jnt_p.append(np.array(jnt_list[-1]).reshape(1, dof))
    #                 axis_id.append(np.array([axis_idx], dtype=np.float32))
    #                 workspc_pos.append(np.array(pos).reshape(1, 3))
    #                 workspc_rotmat.append(np.array(rotmat_init).reshape(1, 9))
    #                 episode_ends_counter += 1

    #             pos = pos_try
    #             jnt_list.append(jnt)
    #             jnt_p.append(np.array(jnt).reshape(1, dof))
    #             axis_id.append(np.array([axis_idx], dtype=np.float32))
    #             p, r = robot.fk(jnt_values=jnt)
    #             workspc_pos.append(np.array(p).reshape(1, 3))
    #             workspc_rotmat.append(np.array(r).reshape(1, 9))
    #         episode_ends_counter += len(jnt_list) - 1
    #         episode_ends_ds.append(np.array([episode_ends_counter], dtype=np.float32))
    #         print(f"Generated {len(jnt_list)} waypoints for axis {axis}.")


