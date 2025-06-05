import numpy as np
import zarr
import time
from copy import copy
from ruckig import InputParameter, OutputParameter, Result, Ruckig
import helper_functions as helper
import os
import concurrent.futures
from threading import Lock

# Initialize parameters and Zarr dataset
ruckig_root = zarr.open('/home/lqin/zarr_datasets/straight_jntpath_finegrained.zarr', mode='r')
episode_len = ruckig_root['meta']['episode_ends'][1:] - ruckig_root['meta']['episode_ends'][:-1]

# Initialize Ruckig parameters
dt = 0.01  # seconds
waypoints_num = 16  # number of waypoints for ruckig
base, robot, otg, inp, out = helper.initialize_ruckig(dt, waypoint_num=waypoints_num)

# Dataset to store results
dataset_name = '/home/lqin/zarr_datasets/straight_jntpath_finegrained_ruckig.zarr'
store = zarr.DirectoryStore(dataset_name)
root = zarr.group(store=store)
meta_group = root.create_group("meta")
data_group = root.create_group("data")
dof = robot.n_dof

# Create datasets for storing generated data
episode_ends_ds = meta_group.create_dataset("episode_ends", shape=(0,), chunks=(1,), dtype=np.float32, append=True)
jnt_p = data_group.create_dataset("jnt_pos", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
jnt_v = data_group.create_dataset("jnt_vel", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
jnt_a = data_group.create_dataset("jnt_acc", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)

# B-spline parameters
degree = 4
num_ctrl_pts = 16
ctrl_points = np.linspace(0, 1, num_ctrl_pts)
knots = np.linspace(0, 1, num_ctrl_pts - degree + 1)
knots = np.concatenate(([0] * degree, knots, [1] * degree))

# Lock for thread safety when writing to the failed file
lock = Lock()


# Function to process each trajectory
def process_trajectory(traj_id, completed_trajectories):
    """
    Process a single trajectory by generating a time-optimal trajectory using Ruckig.
    """
    # Lock to ensure thread-safe checking and adding completed trajectory
    with lock:
        if traj_id in completed_trajectories:
            print(f"Trajectory {traj_id} already processed, skipping.")
            return

        completed_trajectories.add(traj_id)  # Mark this trajectory as processed

    traj_start = int(ruckig_root['meta']['episode_ends'][traj_id - 1]) if traj_id > 0 else 0
    traj_end = int(ruckig_root['meta']['episode_ends'][traj_id])

    jnt_pos_list = ruckig_root['data']['jnt_pos'][traj_start:traj_end]
    pos_list = ruckig_root['data']['position'][traj_start:traj_end]

    print(f'Traj ID: {traj_id}, episode len: {len(jnt_pos_list)}')
    # print(f'start conf: {jnt_pos_list[0]}, end conf: {jnt_pos_list[-1]}')
    # print(f'start pos: {pos_list[0]}, end pos: {pos_list[-1]}')

    # Use Ruckig to generate time-optimal trajectory
    inp.current_position, inp.target_position = jnt_pos_list[0], jnt_pos_list[-1]
    waypoints = np.linspace(np.array(jnt_pos_list[0]), np.array(jnt_pos_list[-1]), waypoints_num)
    inp.intermediate_positions = waypoints

    # Generate the trajectory within the control loop
    first_output, out_list, jnt_path = None, [], []
    res = Result.Working
    episode_ends_counter = 0
    retry_count = 0
    max_retries = 10  # Maximum number of retries
    retry_delay = 2  # Delay between retries (seconds)

    # Accumulators to hold the trajectory data until processing is complete
    jnt_p_accum = []
    jnt_v_accum = []
    jnt_a_accum = []

    while res == Result.Working:
        try:
            res = otg.update(inp, out)

            # Accumulate output data (don't write yet)
            out_list.append(copy(out))
            jnt_path.append(np.array(out.new_position))

            out.pass_to_input(inp)
            jnt_p_accum.append(np.array(out.new_position).reshape(1, dof))
            jnt_v_accum.append(np.array(out.new_velocity).reshape(1, dof))
            jnt_a_accum.append(np.array(out.new_acceleration).reshape(1, dof))
            episode_ends_counter += 1

            if not first_output:
                first_output = copy(out)

        except Exception as e:
            print(f"Error encountered: {e}")
            print(f"Retrying in {retry_delay} seconds...")
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Max retries reached for trajectory {traj_id}. Skipping this trajectory.")
                break
            time.sleep(retry_delay)  # Delay before retrying

    # Ensure the trajectory was processed successfully before saving
    if first_output is not None:
        # After the whole trajectory is processed, write all accumulated data at once
        with lock:
            jnt_p.append(np.vstack(jnt_p_accum))
            jnt_v.append(np.vstack(jnt_v_accum))
            jnt_a.append(np.vstack(jnt_a_accum))
            episode_ends_ds.append(np.array([episode_ends_counter], dtype=np.int32))

        # print(f'Trajectory generated with {len(jnt_path)} waypoints.')
        # print(f'episode_ends_counter: {episode_ends_counter}')
        # print(f'Trajectory duration: {first_output.trajectory.duration:0.4f} [s]')
    else:
        print(f"Skipping trajectory {traj_id} due to errors.")


# Function to process all trajectories using multithreading
def process_all_trajectories():
    # Track completed trajectories
    completed_trajectories = set()

    # Use ThreadPoolExecutor to process trajectories in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for traj_id in range(len(ruckig_root['meta']['episode_ends'][:500])):
            futures.append(executor.submit(process_trajectory, traj_id, completed_trajectories))

        # Wait for all threads to finish
        for future in futures:
            future.result()


# Main script
if __name__ == "__main__":
    process_all_trajectories()
