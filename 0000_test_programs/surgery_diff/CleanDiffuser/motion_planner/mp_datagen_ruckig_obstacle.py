# please install the ruckig library first: pip install ruckig
from ruckig import InputParameter, OutputParameter, Result, Ruckig
import numpy as np
import matplotlib.pyplot as plt
import os
import zarr
from tqdm import tqdm
from copy import copy

import wrs.robot_sim.manipulators.franka_research_3_arm.franka_research_3_arm as franka
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm

'''helper functions'''
import obstacle_utils as obstacle_utils

current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))

dataset_path = os.path.join('/home/lqin', 'zarr_datasets', 'franka_ruckig.zarr')
store = zarr.DirectoryStore(dataset_path)
root = zarr.group(store=store)
print('Current dataset created in:', dataset_path)

meta_group = root.create_group("meta")
data_group = root.create_group("data")
episode_ends_ds = meta_group.create_dataset("episode_ends", shape=(0,), chunks=(1,), dtype=np.float32, append=True)
goal_conf_ds = data_group.create_dataset("goal_conf", shape=(0, 7), chunks=(1, 7), dtype=np.float32, append=True)
jnt_p = data_group.create_dataset("jnt_pos", shape=(0, 7), chunks=(1, 7), dtype=np.float32, append=True)
jnt_v = data_group.create_dataset("jnt_vel", shape=(0, 7), chunks=(1, 7), dtype=np.float32, append=True)
jnt_a = data_group.create_dataset("jnt_acc", shape=(0, 7), chunks=(1, 7), dtype=np.float32, append=True)


def gen_start_goal_conf(robot):
    '''generate the start and goal conf'''
    start_conf = robot.rand_conf()
    goal_conf = robot.rand_conf()
    
    return start_conf, goal_conf

def initialize(sampling_interval):
    '''init the robot and world'''
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = franka.FrankaResearch3Arm(enable_cc=True)
    inp = InputParameter(robot.n_dof)
    out = OutputParameter(robot.n_dof)
    otg = Ruckig(robot.n_dof, sampling_interval)
    
    return base, robot, otg, inp, out

traj_num = 2000
sampling_interval = 0.001  # seconds

if __name__ == '__main__':
    # Initialize the world and robot
    base, robot, otg, inp, out = initialize(sampling_interval)
    
    for id in tqdm(range(traj_num)):
        # Generate random start and goal configurations
        print('-' * 100)
        start_conf = robot.rand_conf()
        goal_conf = robot.rand_conf()
        print('start configuration:', repr(start_conf))
        print('goal configuration:', repr(goal_conf))

        inp.current_position, inp.target_position = start_conf, goal_conf
        inp.current_velocity = rm.np.zeros(robot.n_dof) # support non-zero vel, zero for simplicity
        inp.current_acceleration = rm.np.zeros(robot.n_dof) # support non-zero acc, zero for simplicity
        inp.target_velocity = rm.np.zeros(robot.n_dof)
        inp.target_acceleration = rm.np.zeros(robot.n_dof)
        inp.min_position = robot.jnt_ranges[:, 0]
        inp.max_position = robot.jnt_ranges[:, 1]

        inp.max_velocity = rm.np.asarray([rm.pi * 2 / 3] * robot.n_dof)
        inp.max_acceleration = rm.np.asarray([rm.pi] * robot.n_dof)
        inp.max_jerk = rm.np.asarray([rm.pi * 2] * robot.n_dof)
    
        # print('\t'.join(['t'] + [str(i) for i in range(otg.degrees_of_freedom)]))
    
        # Generate the trajectory within the control loop
        first_output, out_list = None, []
        res = Result.Working
        while res == Result.Working:
            res = otg.update(inp, out)
            # print('\t'.join([f'{out.time:0.3f}'] + [f'{p:0.3f}' for p in out.new_position]))
            
            '''append the trajectory to the dataset'''
            assert (np.array(inp.target_position) == goal_conf).all(), 'goal_conf not equal to inp.target_position'
            goal_conf_ds.append(np.array(inp.target_position).reshape(1, 7))
            jnt_p.append(np.array(out.new_position).reshape(1, 7))
            jnt_v.append(np.array(out.new_velocity).reshape(1, 7))
            jnt_a.append(np.array(out.new_acceleration).reshape(1, 7))

            '''append the trajectory to the dataset'''
            out_list.append(copy(out))
            out.pass_to_input(inp)
    
            if not first_output:
                first_output = copy(out)
        
        episode_ends_ds.append(np.array([len(out_list)], dtype=np.int32))
        print(f'Calculation duration: {first_output.calculation_duration:0.1f} [µs]')
        print(f'Trajectory duration: {first_output.trajectory.duration:0.4f} [s]')
        print(f'Trajectory counter: {len(out_list)}')

        # Plot the trajectory
        from pathlib import Path
        from plotter import Plotter
        import os
    
        project_path = Path(__file__).parent.parent.absolute()
        log_dir = os.path.join('/home/lqin', 'zarr_datasets', 'log_0410')
        os.makedirs(log_dir, exist_ok=True)
        Plotter.plot_trajectory(
            os.path.join(log_dir, f'{id}_trajectory.pdf'),
            otg, inp, out_list,
            plot_jerk=False
        )