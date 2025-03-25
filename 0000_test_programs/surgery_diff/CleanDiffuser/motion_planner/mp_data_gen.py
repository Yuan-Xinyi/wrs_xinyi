import wrs.robot_con.xarm_lite6.xarm_lite6_x as xarm_x
import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm_s
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
import wrs.motion.probabilistic.rrt as rrt

from wrs.motion.trajectory.quintic import QuinticSpline
import matplotlib.pyplot as plt
import wrs.motion.trajectory.topp_ra as toppra

import cv2
import time
import yaml
import numpy as np
import zarr
import os
from tqdm import tqdm

def visualize_start_goal_waypoints(robot, rrt, start_conf, goal_conf):
    robot.goto_given_conf(jnt_values=start_conf)
    robot.gen_meshmodel(rgb=rm.const.steel_blue, alpha=.3).attach_to(base)
    robot.goto_given_conf(jnt_values=goal_conf)
    robot.gen_meshmodel(rgb=[0,1,0], alpha=.3).attach_to(base)

    path = rrt.plan(start_conf=start_conf,
                    goal_conf=goal_conf,
                    ext_dist=.1,
                    max_time=300,
                    animation=True)
    print(path)
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
                                                                                                ctrl_freq=.05)

        return interp_time, interp_confs, interp_spds, interp_accs


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
    '''init the parameters'''
    plot = False
    current_file_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(os.path.dirname(__file__))

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = xarm_s.XArmLite6(enable_cc=True)
    rrt = rrt.RRT(robot)


    '''visualization'''
    # visualize_start_goal_waypoints(robot, rrt, start_conf, goal_conf)
    # visualize_anime(robot, rrt, start_conf, goal_conf)


    '''initialize the dataset'''
    config_file = os.path.join(current_file_dir, 'config.yaml')
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    dataset_dir = os.path.join(parent_dir, 'datasets')
    dataset_name = os.path.join(dataset_dir, 'xarm_toppra_mp.zarr')
    store = zarr.DirectoryStore(dataset_name)
    root = zarr.group(store=store)
    print('dataset created in:', dataset_name)

    meta_group = root.create_group("meta")
    data_group = root.create_group("data")
    episode_ends_ds = meta_group.create_dataset("episode_ends", shape=(0,), chunks=(1,), dtype=np.float32, append=True)
    start_conf_ds = data_group.create_dataset("start_confs", shape=(0, 1), chunks=(1, 1), dtype=np.float32, append=True)
    goal_conf_ds = data_group.create_dataset("goal_confs", shape=(0, 1), chunks=(1, 1), dtype=np.float32, append=True)
    jnt_t = data_group.create_dataset("interp_time", shape=(0, 1), chunks=(1, 1), dtype=np.float32, append=True)
    jnt_cfg = data_group.create_dataset("interp_confs", shape=(0, 6), chunks=(1, 6), dtype=np.float32, append=True)
    jnt_v = data_group.create_dataset("interp_spds", shape=(0, 6), chunks=(1, 6), dtype=np.float32, append=True)
    jnt_a = data_group.create_dataset("interp_accs", shape=(0, 6), chunks=(1, 6), dtype=np.float32, append=True)

    episode_ends_counter = 0
    for _ in tqdm(range(config['traj_num'])):
        start_conf, goal_conf = gen_collision_free_start_goal(robot)
        print('-'*100)
        print('start_conf:', start_conf)
        print('goal_conf:', goal_conf)
        for _ in range(10):
            tic  = time.time()
            interp_time, interp_confs, interp_spds, interp_accs = gen_toppra_traj(rrt, start_conf, goal_conf)
            if interp_time is None:
                break
            episode_ends_counter += len(interp_time)
            episode_ends_ds.append(np.array([episode_ends_counter], dtype=np.int32))
            start_conf_ds.append(start_conf.reshape(-1, 1))
            goal_conf_ds.append(goal_conf.reshape(-1, 1))
            jnt_t.append(interp_time.reshape(-1, 1))
            jnt_cfg.append(interp_confs)
            jnt_v.append(interp_spds)
            jnt_a.append(interp_accs)
            toc = time.time()
            print(f'current traj gen time cost: {toc - tic}')