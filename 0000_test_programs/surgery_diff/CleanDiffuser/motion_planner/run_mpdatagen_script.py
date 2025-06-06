import subprocess
import time
import os
import zarr
import json
import traceback

# This script runs a target Python script with specified trajectory start and end indices.

# Function to run the target script with traj_start and traj_end as parameters
def run_target_script(traj_start, traj_end):
    script_name = "mp_datagen_Waypoints2Ruckig2BSpline.py"
    params = [str(traj_start), str(traj_end)]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)

    result = subprocess.run(
        ['python', script_path] + params,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"Script executed successfully:\n{result.stdout}")
    else:
        raise RuntimeError(f"Subprocess failed with stderr:\n{result.stderr}")

# Open Zarr dataset
ruckig_root = zarr.open('/home/lqin/zarr_datasets/straight_jntpath_partially.zarr', mode='r')
total_trajectories = len(ruckig_root['meta']['episode_ends'][:])
print(f"Total trajectories in the dataset: {total_trajectories}")

# Define batches
traj_batch = 10
traj_start_end_pairs = [
    (start, min(start + traj_batch, total_trajectories))
    for start in range(0, total_trajectories, traj_batch)
]

# Log file
failed_log_file = "failed_trajs.jsonl"
# if dont exist, create it
if not os.path.exists(failed_log_file):
    with open(failed_log_file, "w") as f:
        f.write("[]")  # Initialize with an empty JSON array

# Run loop
for traj_start, traj_end in traj_start_end_pairs:
    print('=' * 100)
    print(f"Running target script with traj_start = {traj_start}, traj_end = {traj_end}")
    print('=' * 100)

    try:
        run_target_script(traj_start, traj_end)

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"Error occurred while running traj [{traj_start}, {traj_end}]: {e}")

        # Real-time append to jsonl
        with open(failed_log_file, "a") as f:
            record = {
                "traj_start": traj_start,
                "traj_end": traj_end,
                "error": err_msg
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    time.sleep(1)  # Optional delay
