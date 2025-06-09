import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline
import zarr
import helper_functions as helper
from scipy.interpolate import BSpline
import os
import sys
from ruckig import InputParameter, OutputParameter, Result, Ruckig
from copy import copy
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm

def extend_path_for_bspline(joint_path: np.ndarray, repeat: int = 2) -> np.ndarray:
    """
    Extend the joint path by repeating the first and last points to improve B-spline boundary smoothness.

    Parameters:
        joint_path (np.ndarray): Original joint path of shape (T, D), where T is the number of points and D is DoF.
        repeat (int): Number of times to repeat the start and end points.

    Returns:
        np.ndarray: Extended joint path of shape (T + 2 * repeat, D).
    """
    if joint_path.shape[0] < 2:
        raise ValueError("joint_path must have at least 2 points.")
    
    start_repeats = np.repeat(joint_path[0:1], repeat, axis=0)
    end_repeats = np.repeat(joint_path[-1:], repeat, axis=0)
    
    extended = np.concatenate([start_repeats, joint_path, end_repeats], axis=0)
    return extended

# Get the parameters passed to the script
if len(sys.argv) < 3:
    print("Please provide both 'id_start' and 'id_end' parameters.")
    sys.exit(1)

# Read the parameters (in this case, traj_start and traj_end)
id_start = int(sys.argv[1])  # The first parameter: trajectory start position
id_end = int(sys.argv[2])    # The second parameter: trajectory end position

print(f"Processing trajectory from {id_start} to {id_end}.")

ruckig_root = zarr.open('/home/lqin/zarr_datasets/0607_simple_straight.zarr', mode='r')
# ruckig_root = zarr.open('/home/lqin/zarr_datasets/straight_line_ruckig_jntpath.zarr', mode='r')

'''ruckig time optimal trajectory generation'''
dt = 0.01  # seconds
waypoints_num = 4  # number of waypoints for ruckig
base, robot, otg, inp, out = helper.initialize_ruckig(dt, waypoint_num=waypoints_num)

'''new the dataset'''
dataset_name = '/home/lqin/zarr_datasets/0607_simple_straight_paras.zarr'
dof = robot.n_dof
# Check if the dataset exists
if os.path.exists(dataset_name):
    root = zarr.open(dataset_name, mode='a')  # Open the dataset in append mode
    jnt_p, jnt_v, jnt_a = root['data']["jnt_pos"], root['data']["jnt_vel"], root['data']["jnt_acc"]
    control_points_ds = root['data']["control_points"]
    ruckig_episode_ends_ds = root['meta']["ruckig_episode_ends"]
    bspline_episode_ends_ds = root['meta']["bspline_episode_ends"]
    ruckig_episode_ends_counter = root['meta']['ruckig_episode_ends'][-1]
    bspline_episode_ends_counter = root['meta']['bspline_episode_ends'][-1]
    workspc_pos = root['data']["position"]
    workspc_rotmat = root['data']["rotation"]
    print('Dataset opened:', dataset_name)

else:
    store = zarr.DirectoryStore(dataset_name)
    root = zarr.group(store=store)  # Create a new dataset
    print('Dataset created in:', dataset_name)

    # Create meta and data groups if they don't exist
    meta_group = root.create_group("meta", overwrite=False)  # Ensure it won't overwrite existing group
    data_group = root.create_group("data", overwrite=False)

    # Create datasets, use append=True to allow adding data
    ruckig_episode_ends_ds = meta_group.create_dataset("ruckig_episode_ends", shape=(0,), chunks=(1,), dtype=np.float32, append=True)
    bspline_episode_ends_ds = meta_group.create_dataset("bspline_episode_ends", shape=(0,), chunks=(1,), dtype=np.float32, append=True)

    # Create datasets for joint positions, velocities, accelerations, and control points
    workspc_pos = data_group.create_dataset("position", shape=(0, 3), chunks=(1, 3), dtype=np.float32, append=True)
    workspc_rotmat = data_group.create_dataset("rotation", shape=(0, 9), chunks=(1, 9), dtype=np.float32, append=True)
    
    jnt_p = data_group.create_dataset("jnt_pos", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
    jnt_v = data_group.create_dataset("jnt_vel", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
    jnt_a = data_group.create_dataset("jnt_acc", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
    control_points_ds = data_group.create_dataset("control_points", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)

    # Initialize counters
    ruckig_episode_ends_counter = 0
    bspline_episode_ends_counter = 0


for traj_id in range(id_start, id_end):
# for traj_id in range(3):
    '''extract traj pos'''
    print('-' * 100)
    print(f'traj id: {traj_id}')

    # id
    traj_start = int(ruckig_root['meta']['episode_ends'][traj_id - 1]) if traj_id > 0 else 0
    traj_end = int(ruckig_root['meta']['episode_ends'][traj_id])

    # jnt pos and workspace pos
    jnt_pos_list = ruckig_root['data']['jnt_pos'][traj_start:traj_end]
    pos_list = ruckig_root['data']['position'][traj_start:traj_end]
    print(f'start conf: {jnt_pos_list[0]}, end conf: {jnt_pos_list[-1]}')
    print(f'start pos: {pos_list[0]}, end pos: {pos_list[-1]}')
    print(f'episode len: {len(jnt_pos_list)}')

    '''use ruckig to generate time optimal trajectory'''
    inp.current_position, inp.target_position = jnt_pos_list[0], jnt_pos_list[-1]
    waypoints = np.linspace(np.array(jnt_pos_list[0]), np.array(jnt_pos_list[-1]), waypoints_num)
    inp.intermediate_positions = waypoints

    # Generate the trajectory within the control loop
    first_output, out_list, jnt_path = None, [], []
    jnt_velpath = []
    jnt_accpath = []

    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)

        # print('\t'.join([f'{out.time:0.3f}'] + [f'{p:0.3f}' for p in out.new_position]))
        out_list.append(copy(out))
        jnt_path.append(np.array(out.new_position))
        jnt_velpath.append(np.array(out.new_velocity))
        jnt_accpath.append(np.array(out.new_acceleration))

        out.pass_to_input(inp)
        jnt_p.append(np.array(out.new_position).reshape(1, dof))
        jnt_v.append(np.array(out.new_velocity).reshape(1, dof))
        jnt_a.append(np.array(out.new_acceleration).reshape(1, dof))
        ruckig_episode_ends_counter += 1

        if not first_output:
            first_output = copy(out)
    ruckig_episode_ends_ds.append(np.array([ruckig_episode_ends_counter], dtype=np.int32))
    print(f'Trajectory generated with {len(jnt_path)} points.')
    print(f'episode_ends_counter: {ruckig_episode_ends_counter}')
    print(f'Trajectory duration: {first_output.trajectory.duration:0.4f} [s]')

    '''visualize the trajectory'''
    # from pathlib import Path
    # from plotter import Plotter
    
    # pdf_path = os.path.join('/home/lqin/zarr_datasets/log_0604', f'test.pdf')
    # if not os.path.exists(os.path.dirname(pdf_path)):
    #     os.makedirs(os.path.dirname(pdf_path))
    # Plotter.plot_trajectory(pdf_path, otg, inp, out_list, plot_jerk=False)
    # helper.workspace_plot(robot, jnt_path)
    # helper.visualize_anime_path(base, robot, jnt_path)

    '''construct b-spline'''
    jnt_path_array = np.array(jnt_path)
    # jnt_path_array = extend_path_for_bspline(jnt_array_org, repeat=2)  # Extend the path for better B-spline fitting
    T = len(jnt_path_array)
    t = np.linspace(0, (T - 1) * dt, T)
    num_joints = jnt_path_array.shape[1]  # reconstruct multiple joints
    s = np.linspace(0, 1, T)

    '''approximate b-spline'''
    '''b-spline parameter'''
    degree = 4
    num_ctrl_pts = 32
    _, org_c, _, _ = helper.Build_BSpline(jnt_path_array, num_ctrl_pts=num_ctrl_pts, degree=degree)

    '''print parameters and save into zarr'''
    print(f'c shape: {org_c.shape}')
    print('-' * 100)
    control_points_ds.append(np.array(org_c))
    bspline_episode_ends_counter += len(org_c)
    bspline_episode_ends_ds.append(np.array([bspline_episode_ends_counter], dtype=np.int32))

    '''test the b-spline reconstruction'''
    ctrl_points = np.linspace(0, 1, num_ctrl_pts)
    knots = np.linspace(0, 1, num_ctrl_pts - degree + 1)
    knots = np.concatenate(([0] * degree, knots, [1] * degree))
    spline = BSpline(knots, org_c, degree)

    T_total_list = [3.3, 4, 5]
    results = []
    for T_total_new in T_total_list:
        print(f"\nTesting with T_total = {T_total_new}s")
        result = (T_total_new, *helper.calculate_BSpline_wrt_T(spline, T_total_new))
        results.append(result)

    # jnt_velpath_array = extend_path_for_bspline(np.array(jnt_velpath), repeat=2)
    # jnt_accpath_array = extend_path_for_bspline(np.array(jnt_accpath), repeat=2)
    jnt_velpath_array = np.array(jnt_velpath)
    jnt_accpath_array = np.array(jnt_accpath)
    
    helper.plot_BSpline_wrt_org(jnt_path_array, jnt_velpath_array, jnt_accpath_array, t, results, overlay=True)


