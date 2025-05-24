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

def generate_obstacle(xyz_lengths, pos, rotmat, rgb=None, alpha=None):
    """生成一个立方体障碍物并显示在 base 中"""
    obj_cmodel = mcm.gen_box(xyz_lengths=xyz_lengths,
                             pos=pos,
                             rotmat=rotmat,
                             rgb=rgb if rgb is not None else [0.3, 0.2, 0.1],
                             alpha=alpha if alpha is not None else 0.5)
    # obj_cmodel.show_local_frame()
    # obj_cmodel.show_cdmesh()
    obj_cmodel.attach_to(base)
    return obj_cmodel

def generate_multiple_obstacles(cube_size=0.05, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    layers = [
        "111110100",  # 第 0 层
        "110100000",  # 第 1 层
        "100000000"   # 第 2 层
    ]
    grid_size = 3
    theta = np.radians(180)
    rot_z = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    obstacle_list, obstacle_info = [], []

    for z, layer in enumerate(layers):
        for i, char in enumerate(layer):
            if char == '1':
                x, y = i % grid_size, grid_size - 1 - (i // grid_size)
                xy = np.dot(rot_z, [x * cube_size, y * cube_size])
                pos = np.array([xy[0] + offset_x, xy[1] + offset_y, z * cube_size + cube_size / 2 + offset_z])
                rotmat = np.eye(3); rotmat[:2, :2] = rot_z
                quat = rm.rotmat_to_quaternion(rotmat)
                obstacle_list.append(generate_obstacle(np.full(3, cube_size), pos, rotmat))
                obstacle_info.append(np.concatenate([pos, quat]))

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
                j = robot.ik(tgt_pos=pos, tgt_rotmat=rotmat)
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

    print(f"{'='*40}\nSuccessfully solved IK for {success_count} / {len(pos_list)} positions.\n{'='*40}")
    return [j for j in jnt_list if j is not None], success_count


if __name__ == '__main__':
    cube_size = 0.3
    offset = 0.02
    waypoint_num = 30
    MAX_TRY_TIME = 5.0

    '''init the parameters'''
    from copy import copy
    current_file_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(os.path.dirname(__file__))

    '''Initialize the world and robot'''
    from ruckig import InputParameter, OutputParameter, Result, Ruckig
    sampling_interval = 0.01  # seconds
    base, robot, otg, inp, out = initialize(sampling_interval, waypoint_num=waypoint_num)
    inp.target_velocity = rm.np.zeros(robot.n_dof)
    inp.target_acceleration = rm.np.zeros(robot.n_dof)
    inp.min_position = robot.jnt_ranges[:, 0]
    inp.max_position = robot.jnt_ranges[:, 1]
    inp.max_velocity = rm.np.asarray([rm.pi * 2 / 3] * robot.n_dof)
    inp.max_acceleration = rm.np.asarray([rm.pi] * robot.n_dof)
    inp.max_jerk = rm.np.asarray([rm.pi * 2] * robot.n_dof)

    '''generate the obstacles'''
    obstacle_list, cube_info = generate_multiple_obstacles(
        cube_size=cube_size,
        offset_x=0.75,
        offset_y=-.15,
        offset_z=0.0
    )
    ground = generate_ground_plane(size_x=2.5, size_y=2.5, color=[0.95, 0.95, 1.0])
    obstacle_list.append(ground)

    '''generate the waypoints jnt configurations'''
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
        visualize=True
        )
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
    plot_joint_trajectories(jnt_list)

    # visualize_anime_path(robot, jnt_list, jnt_list[0], jnt_list[-1])
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
 
        if not first_output:
            first_output = copy(out)
 
    print(f'Calculation duration: {first_output.calculation_duration:0.1f} [µs]')
    print(f'Trajectory duration: {first_output.trajectory.duration:0.4f} [s]')
 
    # Plot the trajectory
    from pathlib import Path
    from plotter import Plotter
    pdf_path = os.path.join(current_file_dir, 'tmp', '03_trajectory.pdf')
    if not os.path.exists(os.path.dirname(pdf_path)):
        os.makedirs(os.path.dirname(pdf_path))
    Plotter.plot_trajectory(pdf_path, otg, inp, out_list, plot_jerk=False)
    visualize_anime_path(robot, jnt_path, jnt_path[0], jnt_path[-1])
