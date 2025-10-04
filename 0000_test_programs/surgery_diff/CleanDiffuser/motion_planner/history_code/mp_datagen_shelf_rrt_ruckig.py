# import wrs.robot_sim.manipulators.franka_research_3_arm.franka_research_3_arm as franka
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
import wrs.motion.probabilistic.rrt_connect_welding as rrtc_welding

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


def generate_ground_plane(size_x=2.0, size_y=2.0, thickness=0.005, z_level=0.0, color=[0.8, 0.8, 0.8], alpha=1.0):
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

def visualize_anime_path(robot, path, start_conf, goal_conf):
    class Data(object):
        def __init__(self):
            self.counter = 0
            self.path = None
            self.current_model = None  # 当前帧的模型

    # robot.goto_given_conf(jnt_values=start_conf)
    # robot.gen_meshmodel(rgb=[0, 0, 1], alpha=.3).attach_to(base)
    # robot.goto_given_conf(jnt_values=goal_conf)
    # robot.gen_meshmodel(rgb=[0, 1, 0], alpha=.3).attach_to(base)
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
    base.run()


def generate_shelf(base, width=0.6, depth=0.4, layer_height=0.3, layers=4, thickness=0.02,
                     rgb=[0.4, 0.3, 0.2], pos_offset=np.zeros(3), rot_theta_deg=0.0):
    """
    生成一个多层柜子（按每层高度构建），支持整体旋转和平移。
    :param base: 可视化 base
    :param width: 柜体宽度 沿 x
    :param depth: 柜体深度 沿 y
    :param layer_height: 每层高度
    :param layers: 层数
    :param thickness: 板材厚度
    :param rgb: 颜色
    :param pos_offset: 位置偏移 np.array([x, y, z])
    :param rot_theta_deg: 绕 Z 轴的旋转角度（单位：度）
    """
    obstacle_list, obstacle_info = [], []
    height = layer_height * layers
    theta = np.radians(rot_theta_deg)
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    def transform(pos_local):
        return np.dot(rot_z, pos_local) + pos_offset

    def attach_box(size, local_pos):
        box = mcm.gen_box(size, pos=transform(local_pos), rotmat=rot_z, rgb=rgb)
        obstacle_list.append(box)
        obstacle_info.append(np.concatenate([transform(local_pos), rm.rotmat_to_quaternion(rot_z)]))
        box.attach_to(base)

    # 各部分组件
    attach_box([width, depth, thickness], [0, 0, thickness / 2])                     # bottom
    attach_box([width, depth, thickness], [0, 0, height - thickness / 2])            # top
    attach_box([thickness, depth, height], [-width / 2 + thickness / 2, 0, height / 2])  # left
    attach_box([thickness, depth, height], [ width / 2 - thickness / 2, 0, height / 2])  # right
    attach_box([width - 2 * thickness, thickness, height],
               [0, -depth / 2 + thickness / 2, height / 2])                          # back

    for i in range(1, layers):
        z = i * layer_height
        attach_box([width - 2 * thickness, depth - thickness, thickness], [0, 0, z])  # shelves
    return obstacle_list, np.array(obstacle_info, dtype=np.float32)

# ruckig function
def initialize(sampling_interval, waypoint_num=10):
    '''init the robot and world'''
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = franka.FrankaResearch3(enable_cc=True)
    inp = InputParameter(robot.n_dof)
    out = OutputParameter(robot.n_dof, waypoint_num)
    otg = Ruckig(robot.n_dof, sampling_interval, waypoint_num)
    
    return base, robot, otg, inp, out

def intrerp_pos_path(cube_center, cube_size, offset, interp_axis, interp_num):
    pos_list = []
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if interp_axis not in axis_map:
        raise ValueError("interp_axis must be 'x', 'y', or 'z'")

    axis_idx = axis_map[interp_axis]
    base_pos = cube_center + np.array([-cube_size/2, cube_size/2, cube_size/2])
    offset_vec = np.zeros(3)
    offset_vec[1] = offset
    base_pos += offset_vec

    for i in range(interp_num):
        pos = base_pos.copy()
        pos[axis_idx] += (i * cube_size / interp_num) - offset
        pos_list.append(pos)

    return pos_list
    
def gen_jnt_list_from_pos_list(pos_list, robot, obstacle_list, base,
                                max_try_time=5.0, check_collision=True, visualize=False):
    jnt_list = []
    success_count = 0

    for pos in pos_list:
        jnt = None
        start_time = time.time()
        while jnt is None and time.time() - start_time < max_try_time:
            try:
                rotmat = rm.rotmat_from_euler(np.pi/2,0,0)
                j = robot.ik(tgt_pos=pos, tgt_rotmat=rotmat, seed_jnt_values = jnt_list[-1] if jnt_list else None)
                if j is None:
                    continue
                robot.goto_given_conf(j)
                if check_collision and robot.cc.is_collided(obstacle_list=obstacle_list):
                    continue
                jnt = j
                success_count += 1
                if visualize:
                    mcm.mgm.gen_frame(pos=pos, rotmat=rotmat).attach_to(base)
                    robot.gen_meshmodel(alpha=.2).attach_to(base)
                break
            except:
                break
        jnt_list.append(jnt)

    print(f"{'-'*40}\nSuccessfully solved IK for {success_count} / {len(pos_list)} positions.\n{'-'*40}")
    return [j for j in jnt_list if j is not None], success_count

def plot_joint_trajectories(jnt_list):
    jnt_array = np.array(jnt_list)
    if jnt_array.ndim != 2 or jnt_array.shape[1] != 7:
        raise ValueError("Expected jnt_list to be of shape (n, 7)")

    fig, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    fig.suptitle("Joint Value Trajectories (Scatter)", fontsize=16)

    for i in range(7):
        axes[i].scatter(range(len(jnt_array)), jnt_array[:, i], s=20)
        axes[i].set_ylabel(f"Joint {i+1}")
        axes[i].grid(True)

    axes[-1].set_xlabel("Index")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def get_local_corner_of_upper_second_layer(width, depth, thickness, layer_height, layers):
    """
    返回从上数第二层层板的右前上角的局部坐标。
    """
    z = (layers - 2) * layer_height  # 层板中心高度（第 layers-2 层）
    width_eff = width - 2 * thickness
    depth_eff = depth - thickness
    thickness_z = thickness

    local_corner = np.array([+width_eff / 2,
                             +depth_eff / 2,
                             z + thickness_z / 2])
    local_corner += np.array([0.0, -0.5, layer_height / 2])  # 局部坐标系的偏移
    return local_corner

if __name__ == '__main__':
    cube_size = 0.3
    offset = 0.02
    max_waypoint_num = 30
    MAX_TRY_TIME = 5.0
    traj_num = 2

    '''init the parameters'''
    from copy import copy
    current_file_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(os.path.dirname(__file__))

    '''Initialize the world and robot'''
    from ruckig import InputParameter, OutputParameter, Result, Ruckig
    sampling_interval = 0.01  # seconds
    base, robot, otg, inp, out = initialize(sampling_interval, waypoint_num=max_waypoint_num)
    inp.target_velocity = rm.np.zeros(robot.n_dof)
    inp.target_acceleration = rm.np.zeros(robot.n_dof)
    inp.min_position = robot.jnt_ranges[:, 0]
    inp.max_position = robot.jnt_ranges[:, 1]
    inp.max_velocity = rm.np.asarray([rm.pi * 2 / 3] * robot.n_dof)
    inp.max_acceleration = rm.np.asarray([rm.pi] * robot.n_dof)
    inp.max_jerk = rm.np.asarray([rm.pi * 2] * robot.n_dof)

    '''generate the obstacles'''
    obstacle_list, cube_info = generate_shelf(
        base,
        width=1.0,
        depth=0.4,
        layer_height=0.2,
        layers=5,
        pos_offset=np.array([0.8, 0.2, 0.0]),
        rot_theta_deg=90
    )
    start_pos = get_local_corner_of_upper_second_layer(
        width=1.0, depth=0.4, thickness=0.02, layer_height=0.2, layers=5
    )
    rotmat = rm.rotmat_from_euler(0,0,0)
    mcm.mgm.gen_frame(pos=start_pos, rotmat=rotmat).attach_to(base)
    ground = generate_ground_plane(size_x=2.5, size_y=2.5, color=[0.95, 0.95, 1.0])
    obstacle_list.append(ground)
    jnt = robot.ik(
        tgt_pos=start_pos, 
        tgt_rotmat=rotmat, 
        seed_jnt_values=None
    )
    robot.goto_given_conf(jnt)
    robot.gen_meshmodel(alpha=1.0).attach_to(base)
    # base.run()

    '''dataset generation'''
    # dataset_name = os.path.join('/home/lqin/zarr_datasets', f'fixed_traj.zarr')
    # store = zarr.DirectoryStore(dataset_name)
    # root = zarr.group(store=store)
    # print('dataset created in:', dataset_name)    
    # meta_group = root.create_group("meta")
    # data_group = root.create_group("data")
    # dof = robot.n_dof
    # episode_ends_ds = meta_group.create_dataset("episode_ends", shape=(0,), chunks=(1,), dtype=np.float32, append=True)
    # jnt_p = data_group.create_dataset("jnt_pos", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
    # jnt_v = data_group.create_dataset("jnt_vel", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
    # jnt_a = data_group.create_dataset("jnt_acc", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
    # episode_ends_counter = 0

    for waypoint_num in range(5, 31):
        '''generate the waypoints jnt configurations'''
        print('='*100)
        pos_list = intrerp_pos_path(
            cube_center=cube_info[7][:3],
            cube_size=cube_size,
            offset=offset,
            interp_axis='x',
            interp_num=waypoint_num
            )
        jnt_list, scc_count = gen_jnt_list_from_pos_list(
            pos_list, robot, 
            obstacle_list, base, 
            max_try_time=MAX_TRY_TIME,
            check_collision=True,
            visualize=False
            )
        # plot_joint_trajectories(jnt_list)
        visualize_anime_path(robot, jnt_list, jnt_list[0], jnt_list[-1])
        base.run()

        '''generate the trajectory'''
        inp.current_position, inp.target_position = jnt_list[0], jnt_list[-1]
        inp.intermediate_positions = jnt_list

        # Generate the trajectory within the control loop
        first_output, out_list, jnt_path = None, [], []
        res = Result.Working
        while res == Result.Working:
            res = otg.update(inp, out)
    
            # print('\t'.join([f'{out.time:0.3f}'] + [f'{p:0.3f}' for p in out.new_position]))
            out_list.append(copy(out))
            jnt_path.append(np.array((out.new_position)))
    
            out.pass_to_input(inp)
            jnt_p.append(np.array((out.new_position)).reshape(1, dof))
            jnt_v.append(np.array((out.new_velocity)).reshape(1, dof))
            jnt_a.append(np.array((out.new_acceleration)).reshape(1, dof))
            episode_ends_counter += 1

            if not first_output:
                first_output = copy(out)
        episode_ends_ds.append(np.array([episode_ends_counter], dtype=np.int32))
        print(f'Trajectory generated with {len(jnt_path)} waypoints.')
        print(f'episode_ends_counter: {episode_ends_counter}')
        print(f'Trajectory duration: {first_output.trajectory.duration:0.4f} [s]')
    
        # visualize the trajectory
        from pathlib import Path
        from plotter import Plotter
        pdf_path = os.path.join('/home/lqin/zarr_datasets/log_0524', f'traj_waypos{waypoint_num}.pdf')
        if not os.path.exists(os.path.dirname(pdf_path)):
            os.makedirs(os.path.dirname(pdf_path))
        Plotter.plot_trajectory(pdf_path, otg, inp, out_list, plot_jerk=False)
        visualize_anime_path(robot, jnt_path, jnt_path[0], jnt_path[-1])
