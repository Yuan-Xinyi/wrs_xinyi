import subprocess
import time
import os
import zarr

# Function to run the target script with traj_start and traj_end as parameters
def run_target_script(traj_start, traj_end):
    # Define the script to call and the parameters to pass
    script_name = "mp_datagen_Waypoints2Ruckig2BSpline.py"  # The script to be called
    params = [str(traj_start), str(traj_end)]  # Convert the parameters to strings to pass them to the target script

    # Call the target Python script with parameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)  # Construct the full path to the script
    result = subprocess.run(['python', script_path] + params, capture_output=True, text=True)

    # Output the result of the script execution
    if result.returncode == 0:
        print(f"Script executed successfully: {result.stdout}")
    else:
        print(f"Error in script execution: {result.stderr}")

ruckig_root = zarr.open('/home/lqin/zarr_datasets/straight_jntpath_finegrained.zarr', mode='r')
# ruckig_root = zarr.open('/home/lqin/zarr_datasets/straight_jntpath.zarr', mode='r')
total_trajectories = len(ruckig_root['meta']['episode_ends'][:])
print(f"Total trajectories in the dataset: {total_trajectories}")

traj_batch = 10
traj_start_end_pairs = [(start, min(start + traj_batch, total_trajectories)) 
                        for start in range(0, total_trajectories, traj_batch)]

for traj_start, traj_end in traj_start_end_pairs:
    print('='*100)
    print(f"Running target script with traj_start = {traj_start}, traj_end = {traj_end}")
    print('='*100)
    run_target_script(traj_start, traj_end)
    time.sleep(1)  # Optional delay to avoid frequent calls
