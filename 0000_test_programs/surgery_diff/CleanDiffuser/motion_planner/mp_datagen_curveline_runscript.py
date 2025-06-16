import subprocess
import time

# Define the path to your main script
script_path = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/mp_datagen_curveline.py'

# Generate numbers from 0 to 2400 with an interval of 10
traj_indices = range(0, 2400, 10)

# Run the target script with different traj_start and traj_end parameters
for traj_start in traj_indices:
    traj_end = min(traj_start + 10, 2400)  # Ensure that traj_end doesn't exceed 2400

    print('=' * 100)
    print(f"Running target script with traj_start = {traj_start}, traj_end = {traj_end}")
    print('=' * 100)

    try:
        # Run the target script with the current traj_start and traj_end
        result = subprocess.run(
            ['python', script_path, str(traj_start), str(traj_end)],  # 传递参数
            capture_output=True,
            text=True
        )

        # Check the result and print
        if result.returncode == 0:
            print(f"Script executed successfully:\n{result.stdout}")
        else:
            print(f"Error occurred:\n{result.stderr}")

    except Exception as e:
        print(f"Error occurred while running traj [{traj_start}, {traj_end}]: {e}")
    
    time.sleep(1)  # Optional delay between each batch
