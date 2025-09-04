# import wrs.robot_con.xarm_lite6.xarm_lite6_x as xarm_x
import wrs.robot_sim.robots.xarmlite6_wg.x6wg2 as xarm_s
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
import wrs.motion.probabilistic.rrt_connect_welding as rrtc_welding
from wrs.motion.trajectory.quintic import QuinticSpline
import matplotlib.pyplot as plt
import wrs.motion.trajectory.totg as toppra
import helper_functions as hf

import cv2
import time
import yaml
import numpy as np
import zarr
import os
from tqdm import tqdm


def gen_jnt_list_from_pos_list(init_jnt, pos_list, robot, obstacle_list, base,
                                max_try_time=5.0, check_collision=True, visualize=False):
    jnt_list = []
    success_count = 0
    _, rotmat = robot.fk(jnt_values=init_jnt)

    for pos in pos_list:
        jnt = None
        start_time = time.time()
        try:
            while jnt is None and time.time() - start_time < max_try_time:
                j = robot.ik(tgt_pos=pos, tgt_rotmat=rotmat, seed_jnt_values=jnt_list[-1] if jnt_list else None)
                if j is None:
                    raise RuntimeError("IK failed")
                robot.goto_given_conf(j)
                if check_collision and robot.cc.is_collided(obstacle_list=obstacle_list):
                    raise RuntimeError("Collision detected")
                jnt = j
                success_count += 1
                if visualize:
                    mcm.mgm.gen_frame(pos=pos, rotmat=rotmat).attach_to(base)
                    robot.gen_meshmodel(alpha=.2).attach_to(base)
                break
            if jnt is None:
                raise RuntimeError("Unknown failure")
            jnt_list.append(jnt)
        except Exception as e:
            # print(f"{'-'*40}\nAborting: {success_count} / {len(pos_list)} positions succeeded.\n{'-'*40}")
            return [j for j in jnt_list if j is not None], success_count

    # print(f"{'-'*40}\nSuccessfully solved IK for {success_count} / {len(pos_list)} positions.\n{'-'*40}")
    return jnt_list, success_count


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


def partiallydiscretize_joint_space(robot, n_intervals=None):
    sampled_jnts = []
    if n_intervals is None:
        n_intervals = np.linspace(6, 4, robot.n_dof - 1, endpoint=False)

    print('=' * 100)
    print(f"Sampling Joint Space using the following joint granularity (excluding last DOF): {n_intervals.astype(int)}...")

    for i in range(robot.n_dof - 1):
        sampled_jnts.append(
            np.linspace(robot.jnt_ranges[i][0], robot.jnt_ranges[i][1], int(n_intervals[i] + 2))[1:-1]
        )

    grid = np.meshgrid(*sampled_jnts)
    base_qs = np.vstack([x.ravel() for x in grid]).T

    last_column = np.zeros((base_qs.shape[0], 1))  # 或 np.full((..., 1), np.nan)
    sampled_qs = np.hstack((base_qs, last_column))

    return sampled_qs

def generate_random_cubic_curve(num_points=20, scale=0.1, center=np.array([0.4, 0.2, 0.3])):
    # 随机生成 4 个三维系数向量
    a = np.random.randn(3) * scale
    b = np.random.randn(3) * scale
    c = np.random.randn(3) * scale
    d = center
    coeffs = np.hstack([a, b, c, d])  # 展平为 shape=(12,)

    # 生成空间轨迹点
    t_vals = np.linspace(0, 1, num_points)
    points = [a * t**3 + b * t**2 + c * t + d for t in t_vals]

    return np.array(points), coeffs

if __name__ == "__main__":
    import sys
    # Get the parameters passed to the script
    if len(sys.argv) < 3:
        print("Please provide both 'id_start' and 'id_end' parameters.")
        sys.exit(1)

    # Read the parameters (in this case, traj_start and traj_end)
    id_start = int(sys.argv[1])  # The first parameter: trajectory start position
    id_end = int(sys.argv[2])    # The second parameter: trajectory end position

    print(f"Processing trajectory from {id_start} to {id_end}.")


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
    robot_s = xarm_s.XArmLite6WG2(enable_cc=True)

    '''dataset generation'''
    dataset_name = os.path.join('/home/lqin/zarr_datasets', f'0616_curvelineIK.zarr')
    dof = robot_s.n_dof
    if os.path.exists(dataset_name):
        root = zarr.open(dataset_name, mode='a')  # Open the dataset in append mode
        jnt_p = root['data']["jnt_pos"]
        workspc_pos = root['data']["position"]
        workspc_rotmat = root['data']["rotation"]
        coef = root['data']["coef"]
        episode_ends_ds = root['meta']["episode_ends"]
        episode_ends_counter = root['meta']['episode_ends'][-1]
        print('Dataset opened:', dataset_name)

    else:
        store = zarr.DirectoryStore(dataset_name)
        root = zarr.group(store=store)
        print('dataset created in:', dataset_name)    
        meta_group = root.create_group("meta")
        data_group = root.create_group("data")
        episode_ends_ds = meta_group.create_dataset("episode_ends", shape=(0,), chunks=(1,), dtype=np.float32, append=True)
        jnt_p = data_group.create_dataset("jnt_pos", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
        workspc_pos = data_group.create_dataset("position", shape=(0, 3), chunks=(1, 3), dtype=np.float32, append=True)
        workspc_rotmat = data_group.create_dataset("rotation", shape=(0, 9), chunks=(1, 9), dtype=np.float32, append=True)
        coef = data_group.create_dataset("coef", shape=(0, 12), chunks=(1, 12), dtype=np.float32, append=True)
        episode_ends_counter = 0

    jnt_samples = partiallydiscretize_joint_space(robot_s)
    print(f"Robot has {robot_s.n_dof} degree of freedoms, total {len(jnt_samples)} joint configurations sampled.")
    print('--' * 100)

    n = 50

    # Iterate through each joint sample
    for jnt_sample in jnt_samples[id_start:id_end]:
        pos_init, rotmat_init = robot_s.fk(jnt_values=jnt_sample)
        
        for _ in range(n):
            scale = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
            workspace_points, coeffs = generate_random_cubic_curve(num_points=64, scale=scale, center=pos_init)
            
            pos_list, success_count = gen_jnt_list_from_pos_list(init_jnt=jnt_samples[_],
                pos_list=workspace_points, robot=robot_s, obstacle_list=None, base=base,
                max_try_time=MAX_TRY_TIME, check_collision=True, visualize=False
            )
            
            # Save the generated trajectory data
            for jnt in pos_list:
                p, r = robot_s.fk(jnt_values=jnt)
                jnt_p.append(np.array(jnt).reshape(1, dof))
                workspc_pos.append(np.array(p).reshape(1, 3))
                workspc_rotmat.append(np.array(r).reshape(1, 9))
                coef.append(np.array(coeffs).reshape(1, 12))
            
            # Update the counter
            episode_ends_counter += len(pos_list)
            episode_ends_ds.append(np.array([episode_ends_counter], dtype=np.float32))
            
            print(f"Generated {len(pos_list)} waypoints for the cubic curve with scale: {scale}")

    '''visualize the generated joint paths'''
    hf.visualize_anime_path(base, robot_s, pos_list)


