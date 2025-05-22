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

def generate_ground_plane(size_x=2.0, size_y=2.0, thickness=0.001, z_level=0.0, color=[0.8, 0.8, 0.8], alpha=1.0):
    """
    在 z=0 添加一个灰色地面平面（非常薄的立方体）。
    """
    pos = np.array([0.0, 0.0, z_level - thickness / 2.0])  # 中心在 z=0
    size = np.array([size_x, size_y, thickness])
    rotmat = np.eye(3)

    ground = mcm.gen_box(
        xyz_lengths=size,
        pos=pos,
        rotmat=rotmat,
        rgb=color,
        alpha=alpha
    )
    ground.attach_to(base)
    return ground

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

def visualize_anime_diffusion(robot, path, start_conf, goal_conf):
    class Data(object):
        def __init__(self):
            self.counter = 0
            self.path = None
            self.current_model = None  # 当前帧的模型

    robot.goto_given_conf(jnt_values=start_conf)
    robot.gen_meshmodel(rgb=[0, 0, 1], alpha=.3).attach_to(base)
    robot.goto_given_conf(jnt_values=goal_conf)
    robot.gen_meshmodel(rgb=[0, 1, 0], alpha=.3).attach_to(base)

    anime_data = Data()
    anime_data.path = path

    def update(robot, anime_data, task):
        if anime_data.counter >= len(anime_data.path):
            if anime_data.current_model:
                anime_data.current_model.detach()  # 移除最后一帧的模型
            anime_data.counter = 0
            return task.done

        # 移除上一帧的模型
        if anime_data.current_model:
            anime_data.current_model.detach()

        # 更新机器人位置并生成当前帧的模型
        conf = anime_data.path[anime_data.counter]
        robot.goto_given_conf(conf)
        anime_data.current_model = robot.gen_meshmodel(alpha=1.0)
        anime_data.current_model.attach_to(base)

        anime_data.counter += 1
        return task.again

    taskMgr.doMethodLater(0.01, update, "update",
                        extraArgs=[robot, anime_data],
                        appendTask=True)


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

def draw_edges_of_cube(center, size=0.05, rotmat=np.eye(3), color=[0, 0, 0]):
    """
    用 stick 画出一个 cube 的 12 条边。
    """
    r = size / 2.0
    local_corners = np.array([
        [-r, -r, -r], [ r, -r, -r], [ r,  r, -r], [-r,  r, -r],
        [-r, -r,  r], [ r, -r,  r], [ r,  r,  r], [-r,  r,  r]
    ])
    global_corners = np.dot(local_corners, rotmat.T) + center

    edge_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for i, j in edge_pairs:
        spos = global_corners[i]
        epos = global_corners[j]
        stick = mcm.gen_stick(spos=spos, epos=epos, radius=0.0005, rgb=color, alpha=1.0)
        stick.attach_to(base)

def generate_obstacle(xyz_lengths, pos, rotmat, rgb=None, alpha=None):
    """生成一个立方体障碍物并显示在 base 中"""
    obj_cmodel = mcm.gen_box(xyz_lengths=xyz_lengths,
                             pos=pos,
                             rotmat=rotmat,
                             rgb=rgb if rgb is not None else [0.7, 0.7, 0.7],
                             alpha=alpha if alpha is not None else 0.5)
    # obj_cmodel.show_local_frame()
    # obj_cmodel.show_cdmesh()
    obj_cmodel.attach_to(base)
    return obj_cmodel

def generate_multiple_obstacles(cube_size=0.05, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """
    在 4x4 网格上生成四层塔状结构障碍物，整体绕 Z 轴旋转 -135°。
    """
    layers = [
        "1111111011001000",  # 第 0 层：底层几乎填满
        "1110110010000000",  # 第 1 层：缩小范围
        "1100100000000000",  # 第 2 层：仅左上和中心
        "1000000000000000"   # 第 3 层：仅左上角
    ]

    grid_size = 4  # 4x4 网格
    obstacle_list = []
    obstacle_positions = []

    # Z轴旋转 -135 度
    theta = np.radians(-135)
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    for z, layer in enumerate(layers):
        for i, char in enumerate(layer):
            if char == '1':
                x = i % grid_size
                y = grid_size - 1 - (i // grid_size)

                # 原始位置
                x_raw = x * cube_size
                y_raw = y * cube_size

                # 旋转后的坐标
                xy_rotated = np.dot(rot_z, np.array([x_raw, y_raw]))

                pos = np.array([
                    xy_rotated[0] + offset_x,
                    xy_rotated[1] + offset_y,
                    z * cube_size + cube_size / 2.0 + offset_z
                ])

                # 旋转后的朝向
                rotmat = np.eye(3)
                rotmat[:2, :2] = rot_z

                box = generate_obstacle(
                    xyz_lengths=np.array([cube_size] * 3),
                    pos=pos,
                    rotmat=rotmat
                )
                draw_edges_of_cube(center=pos, size=cube_size, rotmat=rotmat, color=[0, 0, 0])
                obstacle_list.append(box)
                obstacle_positions.extend(pos)

    obstacle_info = np.array(obstacle_positions, dtype=np.float32)
    return obstacle_list, obstacle_info


def generate_obstacle_confs(robot, obstacle_list, obstacle_info):
    """
    尝试生成一组无碰撞起始-目标姿态与障碍物集合（自带可视化）。
    """
    while True:
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
    generate_ground_plane(size_x=2, size_y=2, color=[0.9, 0.9, 0.9])
    mgm.gen_frame().attach_to(base)
    robot = franka.FrankaResearch3(enable_cc=True)
    robot.gen_meshmodel(alpha=1).attach_to(base)
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
    obstacle_list, obstacle_info = generate_multiple_obstacles(
        cube_size=0.15,
        offset_x=0.5,
        offset_y=0.35,
        offset_z=0.0
    )
    
    base.run()
    start_conf, goal_conf = generate_obstacle_confs(robot, obstacle_list, obstacle_info)
    visualize_start_goal_waypoints(robot, rrt, start_conf, goal_conf, obstacle_list=obstacle_list)
    visualize_anime(robot, rrt, start_conf, goal_conf)


    '''initialize the dataset'''
    config_file = os.path.join(current_file_dir, 'config','config.yaml')
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    dataset_dir = os.path.join(parent_dir, 'datasets')
    
    obstacle_num = 3
    obstacle_info_shape = 3*obstacle_num

    dataset_name = os.path.join(dataset_dir, f'franka_kinodyn_obstacles_{obstacle_num}.zarr')
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
    obstacles = data_group.create_dataset("obstacles", shape=(0, obstacle_info_shape), chunks=(1, obstacle_info_shape), dtype=np.float32, append=True)

    episode_ends_counter = 0
    max_steps_per_episode = 2000
    for _ in tqdm(range(config['traj_num'])):
        start_conf, goal_conf, obstacle_list, obstacle_info = generate_obstacle_confs(robot, obstacle_num=3)  
        inp.current_position, inp.target_position = start_conf, goal_conf
        print('-'*100)
        print('start_conf:', start_conf)
        print('goal_conf:', goal_conf)
        
        for _ in range(2):
            
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
                obstacles.append(obstacle_info.reshape(1, obstacle_info_shape))
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
