# import wrs.robot_sim.manipulators.franka_research_3_arm.franka_research_3_arm as franka
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
import wrs.motion.probabilistic.rrt as rrt
import wrs.motion.probabilistic.rrt_connect as rrtc

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
MAX_TRY_TIME = 5.0

def visualize_start_goal_waypoints(robot, rrt, start_conf, goal_conf, obstacle_list=[]):
    robot.goto_given_conf(jnt_values=start_conf)
    robot.gen_meshmodel(rgb=rm.const.steel_blue, alpha=.3).attach_to(base)
    robot.goto_given_conf(jnt_values=goal_conf)
    robot.gen_meshmodel(rgb=[0,1,0], alpha=.3).attach_to(base)


    '''obstacle'''
    # base.run()
    path = rrt.plan(start_conf=start_conf,
                    goal_conf=goal_conf,
                    obstacle_list=obstacle_list,
                    ext_dist=.1,
                    max_time=300,
                    animation=False)
    
    if path is not None:
        for conf in path:
            robot.goto_given_conf(jnt_values=conf)
            robot.gen_meshmodel(alpha=.1).attach_to(base)

    base.run()


def visualize_anime(robot, rrt, start_conf, goal_conf):
    
    '''generate the dataset'''
    motion_data = rrt.plan(start_conf=start_conf,
                    goal_conf=goal_conf,
                    ext_dist=.1,
                    max_time=300,
                    animation=True)

    '''interpolate the path'''
    jv_array = rm.np.asarray(motion_data.jv_list)
    interp_x = rm.np.linspace(0, len(jv_array) - 1, 100)
    cs = QuinticSpline(range(len(motion_data.jv_list)), motion_data.jv_list)
    interp_confs = cs(interp_x)

    '''generate the time-optimal trajectory'''
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))
    ax1.plot(range(len(interp_confs)), interp_confs, '-o')
    interp_time, interp_confs1, interp_spds, interp_accs = toppra.generate_time_optimal_trajectory(interp_confs,
                                                                                            ctrl_freq=.05)
    ax2.plot(interp_time, interp_confs1, '-o')
    ax3.plot(interp_time, interp_spds, '-o')
    ax4.plot(interp_time, interp_accs, '-o')
    plt.show()

    _, interp_confs2, _, _ = toppra.generate_time_optimal_trajectory(motion_data.jv_list, ctrl_freq=.05)

    class Data(object):
        def __init__(self):
            self.counter = 0
            self.path1 = None
            self.path2 = None
            self.on_screen = []


    anime_data = Data()
    anime_data.path1 = interp_confs1
    anime_data.path2 = interp_confs2


    def update(robot, anime_data, task):
        if anime_data.counter >= len(anime_data.path1):
            for model in anime_data.on_screen:
                model.detach()
            anime_data.counter = 0
            anime_data.on_screen = []
        conf = anime_data.path1[anime_data.counter]
        robot.goto_given_conf(conf)
        model = robot.gen_meshmodel()
        model.attach_to(base)
        anime_data.on_screen.append(model)
        if anime_data.counter < len(anime_data.path2):
            conf = anime_data.path2[anime_data.counter]
            robot.goto_given_conf(conf)
            model = robot.gen_meshmodel()
            model.attach_to(base)
            anime_data.on_screen.append(model)
        anime_data.counter += 1
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                            extraArgs=[robot, anime_data],
                            appendTask=True)

    base.run()


def gen_toppra_traj(rrt, start_conf, goal_conf):
    
    '''generate the dataset'''
    motion_data = rrt.plan(start_conf=start_conf,
                    goal_conf=goal_conf,
                    ext_dist=.1,
                    max_time=45,
                    animation=True)

    '''interpolate the path'''
    if motion_data is None:
        return None, None, None, None
    else:
        jv_array = rm.np.asarray(motion_data.jv_list)
        interp_x = rm.np.linspace(0, len(jv_array) - 1, 100)
        cs = QuinticSpline(range(len(motion_data.jv_list)), motion_data.jv_list)
        interp_confs = cs(interp_x)

        '''generate the time-optimal trajectory'''
        interp_time, interp_confs, interp_spds, interp_accs = toppra.generate_time_optimal_trajectory(interp_confs,
                                                                                                ctrl_freq=.01)

        return interp_time, interp_confs, interp_spds, interp_accs


def gen_collision_free_start_goal(robot, obstacle_list=[]):
    '''generate the start and goal conf'''
    MAX_ITER = 100  
    for _ in range(MAX_ITER):
        start_conf = robot.rand_conf()
        goal_conf = robot.rand_conf()
        robot.goto_given_conf(jnt_values=start_conf)
        start_cc = robot.cc.is_collided(obstacle_list=obstacle_list)
        robot.goto_given_conf(jnt_values=goal_conf)
        goal_cc = robot.cc.is_collided(obstacle_list=obstacle_list)
        if not start_cc and not goal_cc:
            return start_conf, goal_conf
    return None, None

def generate_obstacle(xyz_lengths, pos, rotmat, rgb=None, alpha=None):
    # generate a single obstacle
    obj_cmodel = mcm.gen_box(xyz_lengths=xyz_lengths,
                             pos=pos, rotmat=rotmat)
    obj_cmodel.show_local_frame()
    obj_cmodel.show_cdmesh()
    obj_cmodel.attach_to(base)
    return obj_cmodel

def generate_multiple_obstacles(obstacle_num):
    # generate multiple obstacles
    obstacle_list = []
    obstacle_info = np.zeros(9, dtype=np.float32)
    for i in range(obstacle_num):
        xyz_lengths = rm.np.array([.05, .05, .05])
        pos = np.array([
            np.random.uniform(low=-.6, high=.6),
            np.random.uniform(low=-.2, high=.6),
            np.random.uniform(low=.0, high=.8)
        ])
        obstacle_info[i*3:i*3+3] = pos
        rotmat = np.eye(3)
        obj_cmodel = generate_obstacle(xyz_lengths=xyz_lengths, pos=pos, rotmat=rotmat)
        obstacle_list.append(obj_cmodel)
    return obstacle_list, obstacle_info

def generate_obstacle_confs(robot, obstacle_num=3):
    while True:
        obstacle_list, obstacle_info = generate_multiple_obstacles(obstacle_num)

        start_time = time.time()
        while time.time() - start_time < MAX_TRY_TIME:
            try:
                start_conf, goal_conf = gen_collision_free_start_goal(robot, obstacle_list)
                if start_conf is not None and goal_conf is not None:
                    return start_conf, goal_conf, obstacle_list, obstacle_info
            except Exception as e:
                break


if __name__ == '__main__':
    '''init the parameters'''
    from copy import copy
    plot = False
    current_file_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(os.path.dirname(__file__))

    '''path planning'''
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = franka.FrankaResearch3(enable_cc=True)
    rrt = rrtc.RRTConnect(robot)

    '''motion planning'''
    from ruckig import InputParameter, OutputParameter, Ruckig, Result
    inp = InputParameter(robot.n_dof)
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

    '''visualization'''
    # start_conf, goal_conf, obstacle_list = generate_obstacle_confs(robot, obstacle_num=6)  
    # visualize_start_goal_waypoints(robot, rrt, start_conf, goal_conf, obstacle_list=obstacle_list)
    # visualize_anime(robot, rrt, start_conf, goal_conf)


    '''initialize the dataset'''
    config_file = os.path.join(current_file_dir, 'config.yaml')
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    dataset_dir = os.path.join(parent_dir, 'datasets')
    dataset_name = os.path.join(dataset_dir, 'franka_kinodyn_obstacles_3.zarr')
    # dataset_name = os.path.join(dataset_dir, 'test.zarr')
    store = zarr.DirectoryStore(dataset_name)
    root = zarr.group(store=store)
    print('dataset created in:', dataset_name)

    meta_group = root.create_group("meta")
    data_group = root.create_group("data")
    dof = robot.n_dof
    episode_ends_ds = meta_group.create_dataset("episode_ends", shape=(0,), chunks=(1,), dtype=np.float32, append=True)
    jnt_p = data_group.create_dataset("jnt_pos", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
    jnt_v = data_group.create_dataset("jnt_vel", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
    jnt_a = data_group.create_dataset("jnt_acc", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
    obstacles = data_group.create_dataset("obstacles", shape=(0, 9), chunks=(1, 18), dtype=np.float32, append=True)

    episode_ends_counter = 0
    max_steps_per_episode = 2000
    for _ in tqdm(range(config['traj_num'])):
        start_conf, goal_conf, obstacle_list, obstacle_info = generate_obstacle_confs(robot, obstacle_num=3)  
        inp.current_position, inp.target_position = start_conf, goal_conf
        print('-'*100)
        print('start_conf:', start_conf)
        print('goal_conf:', goal_conf)
        
        for _ in range(3):
            
            path = rrt.plan(start_conf=start_conf,
                            goal_conf=goal_conf,
                            obstacle_list=obstacle_list,
                            ext_dist=.1,
                            max_time=300,
                            animation=False)
            
            if path is None:
                print('Failed to generate the trajectory')
                break

            otg = Ruckig(robot.n_dof, 0.01, len(path.jv_list))
            inp.intermediate_positions = path.jv_list
            out = OutputParameter(robot.n_dof, len(path.jv_list))
            step_counter = 0
            
            # Generate the trajectory within the control loop
            first_output, out_list = None, []
            res = Result.Working
            while res == Result.Working:
                res = otg.update(inp, out)
                # print('\t'.join([f'{out.time:0.3f}'] + [f'{p:0.3f}' for p in out.new_position]))
                out_list.append(copy(out))
                out.pass_to_input(inp)
                obstacles.append(obstacle_info.reshape(1, 9))
                jnt_p.append(np.array((out.new_position)).reshape(1, dof))
                jnt_v.append(np.array((out.new_velocity)).reshape(1, dof))
                jnt_a.append(np.array((out.new_acceleration)).reshape(1, dof))
                episode_ends_counter += 1
                step_counter += 1
                if not first_output:
                    first_output = copy(out)
                
                if step_counter > max_steps_per_episode:
                    break

            episode_ends_ds.append(np.array([episode_ends_counter], dtype=np.int32))
