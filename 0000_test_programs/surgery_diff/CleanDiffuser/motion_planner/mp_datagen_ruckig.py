# please install the ruckig library first: pip install ruckig
from ruckig import InputParameter, Ruckig, Trajectory, Result
import numpy as np
import matplotlib.pyplot as plt
import os
import zarr
from tqdm import tqdm

import wrs.robot_sim.manipulators.franka_research_3_arm.franka_research_3_arm as franka
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm

current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))

dataset_dir = os.path.join(parent_dir, 'datasets')
dataset_name = os.path.join(dataset_dir, 'franka_mp_ruckig_1000hz_2.zarr')
store = zarr.DirectoryStore(dataset_name)
root = zarr.group(store=store)
print('dataset created in:', dataset_name)

meta_group = root.create_group("meta")
data_group = root.create_group("data")
episode_ends_ds = meta_group.create_dataset("episode_ends", shape=(0,), chunks=(1,), dtype=np.float32, append=True)
jnt_p = data_group.create_dataset("jnt_pos", shape=(0, 7), chunks=(1, 7), dtype=np.float32, append=True)
jnt_v = data_group.create_dataset("jnt_vel", shape=(0, 7), chunks=(1, 7), dtype=np.float32, append=True)
jnt_a = data_group.create_dataset("jnt_acc", shape=(0, 7), chunks=(1, 7), dtype=np.float32, append=True)

def gen_collision_free_start_goal(robot):
    '''generate the start and goal conf'''
    while True:
        start_conf = robot.rand_conf()
        goal_conf = robot.rand_conf()
        robot.goto_given_conf(jnt_values=start_conf)
        start_cc = robot.cc.is_collided()
        robot.goto_given_conf(jnt_values=goal_conf)
        goal_cc = robot.cc.is_collided()
        if not start_cc and not goal_cc:
            break
    return start_conf, goal_conf


if __name__ == '__main__':
    '''init the robot and world'''
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = franka.FrankaResearch3Arm(enable_cc=True)
    inp = InputParameter(robot.n_dof)

    traj_num = 1000
    episode_ends_counter = 0

    '''assign the default dynamic constraints'''
    for _ in tqdm(range(traj_num)):
        start_conf, goal_conf = gen_collision_free_start_goal(robot)
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

    
        # support different constraints for negative direction, default same magnitude as positive
        inp.min_velocity = [-value for value in inp.max_velocity]
        inp.min_acceleration = [-value for value in inp.max_acceleration]
    
        # We don't need to pass the control rate (cycle time) when using only offline features
        otg = Ruckig(robot.n_dof)
        trajectory = Trajectory(robot.n_dof)
    
        # Calculate the trajectory in an offline manner
        result = otg.calculate(inp, trajectory)
        if result == Result.ErrorInvalidInput:
            raise Exception('Invalid input!')
    
        print(f'Trajectory duration: {trajectory.duration:0.4f} [s]')

        '''start the ploting session'''
        sampling_interval = 0.001  # seconds
        time_points = np.linspace(0, trajectory.duration, 
                                num=int(trajectory.duration/sampling_interval)+1,
                                endpoint=True)
        episode_ends_counter += len(time_points)
        for t in time_points:
            pos, vel, acc = trajectory.at_time(t)
            jnt_p.append(np.array(pos).reshape(1, 7))
            jnt_v.append(np.array(vel).reshape(1, 7))
            jnt_a.append(np.array(acc).reshape(1, 7))
        episode_ends_ds.append(np.array([episode_ends_counter], dtype=np.int32))

    