# import wrs.robot_con.xarm_lite6.xarm_lite6_x as xarm_x
import wrs.robot_sim.robots.xarmlite6_wg.x6wg2 as xarm_s
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
import wrs.motion.probabilistic.rrt_connect_welding as rrtc_welding
from wrs.motion.trajectory.quintic import QuinticSpline
import matplotlib.pyplot as plt
import wrs.motion.trajectory.totg as toppra
import helper_functions as hf
import utils
import helper_functions as hf

import cv2
import time
import yaml
import numpy as np
import zarr
import os
from tqdm import tqdm


def _orthonormal_basis_of_axis(a: np.ndarray):
    a = a / (np.linalg.norm(a) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b = np.cross(a, tmp)
    nb = np.linalg.norm(b)
    if nb < 1e-12:
        tmp = np.array([0.0, 0.0, 1.0])
        b = np.cross(a, tmp)
        nb = np.linalg.norm(b)
    b = b / (nb + 1e-12)
    c = np.cross(a, b)
    c = c / (np.linalg.norm(c) + 1e-12)
    return a, b, c

def _rot_from_tool_z(u: np.ndarray, a_axis: np.ndarray):
    # 构造 R 的第三列为 u（工具 z 轴）；x 取 a×u 方向，y = u×x
    au = np.cross(a_axis, u)
    n = np.linalg.norm(au)
    if n < 1e-12:
        tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(u, tmp)
        x = x / (np.linalg.norm(x) + 1e-12)
    else:
        x = au / n
    y = np.cross(u, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    R = np.stack([x, y, u], axis=1)
    return R

def _sample_rotations_in_cone(a_axis, alpha_max_rad, n_alpha=3, n_psi=12, bias_u=None):
    a, b, c = _orthonormal_basis_of_axis(np.asarray(a_axis, dtype=float))
    # 采样 alpha（包含 0 和 alpha_max）
    alphas = [min(alpha_max_rad, max(0.0, 0.0))] if n_alpha <= 1 else np.linspace(0.0, alpha_max_rad, n_alpha)
    # 采样 psi 覆盖 [0, 2π)
    psis = np.linspace(0.0, 2*np.pi, num=max(n_psi, 1), endpoint=False)

    candidates = []
    for alpha in alphas:
        for psi in psis:
            u = np.cos(alpha)*a + np.sin(alpha)*(b*np.cos(psi) + c*np.sin(psi))
            u = u / (np.linalg.norm(u) + 1e-12)
            candidates.append(u)

    if bias_u is not None:
        bu = bias_u / (np.linalg.norm(bias_u) + 1e-12)
        candidates.sort(key=lambda u: np.arccos(np.clip(np.dot(u, bu), -1.0, 1.0)))
    else:
        candidates.sort(key=lambda u: np.arccos(np.clip(np.dot(u, a), -1.0, 1.0)))

    R_list = [_rot_from_tool_z(u, a_axis=a) for u in candidates]
    return R_list


def gen_jnt_list_from_pos_list_relaxed(
        init_jnt, pos_list, robot, obstacle_list, base,
        alpha_max_rad,                 # 允许的最大偏转角（相对初始姿态的工具轴）
        n_alpha=3, n_psi=12,
        tool_axis='z',
        max_try_time=5.0,
        check_collision=True,
        visualize=False):
    """
    放宽姿态为“相对初始姿态的旋转锥”：
    对每个目标点，在以初始工具轴为锥轴、半角 alpha_max_rad 的锥内采样若干候选姿态求 IK。
    返回:
        jnt_list: 成功解的关节角列表
        success_count: 成功点数
        deviations: 偏差角列表 (弧度)，第一个元素为 0.0
    """
    jnt_list = []
    success_count = 0
    deviations = []

    # 固定锥轴 = 初始工具轴方向
    _, rot0 = robot.fk(jnt_values=init_jnt)
    if tool_axis == 'z':
        u_ref = rot0[:, 2]
    elif tool_axis == 'x':
        u_ref = rot0[:, 0]
    elif tool_axis == 'y':
        u_ref = rot0[:, 1]
    else:
        raise ValueError("tool_axis must be one of {'x','y','z'}")

    # 偏差计算时用
    prev_u = None

    for pos in pos_list:
        jnt = None
        start_time = time.time()

        # 在固定锥轴 u_ref 下采样候选姿态
        R_cands = _sample_rotations_in_cone(
            a_axis=u_ref,
            alpha_max_rad=alpha_max_rad,
            n_alpha=n_alpha,
            n_psi=n_psi
        )

        for R in R_cands:
            if time.time() - start_time >= max_try_time:
                break
            seed = jnt_list[-1] if jnt_list else init_jnt
            j = robot.ik(tgt_pos=pos, tgt_rotmat=R, seed_jnt_values=seed)
            if j is None:
                continue
            robot.goto_given_conf(j)
            if check_collision and robot.cc.is_collided(obstacle_list=obstacle_list):
                continue

            # 成功
            jnt = j
            success_count += 1
            jnt_list.append(jnt)

            # 当前真实工具轴方向
            _, R_now = robot.fk(jnt_values=jnt)
            if tool_axis == 'z':
                u_now = R_now[:, 2]
            elif tool_axis == 'x':
                u_now = R_now[:, 0]
            else:
                u_now = R_now[:, 1]
            u_now = u_now / (np.linalg.norm(u_now) + 1e-12)

            # 偏差角计算
            if prev_u is None:
                deviations.append(0.0)
            else:
                dot = np.clip(np.dot(prev_u, u_now), -1.0, 1.0)
                theta = float(np.arccos(dot))
                deviations.append(theta)

            prev_u = u_now

            if visualize:
                mcm.mgm.gen_frame(pos=pos, rotmat=R_now).attach_to(base)
                robot.gen_meshmodel(alpha=.2).attach_to(base)
            break

        if jnt is None:
            return [j for j in jnt_list if j is not None], success_count, deviations

    return jnt_list, success_count, deviations



MAX_TRY_TIME = 100.0
alpha_max_rad = np.deg2rad(10.0)
waypoints_num = 32

if __name__ == "__main__":
    '''Initialize the world and robot'''
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot_s = xarm_s.XArmLite6WG2(enable_cc=True)

    init_jnt = robot_s.rand_conf()
    pos_init, rotmat_init = robot_s.fk(jnt_values=init_jnt)
    scale = np.random.choice([0.1, 0.2, 0.3])
    scale = 0.1
    success = False

    while not success:
        workspace_points, coeffs = utils.generate_random_cubic_curve(num_points=waypoints_num, scale=scale, center=pos_init)
        jnt_list, success_count, deviations = gen_jnt_list_from_pos_list_relaxed(
            init_jnt=init_jnt,
            pos_list=workspace_points,
            robot=robot_s,
            obstacle_list=None,
            base=base,
            alpha_max_rad=alpha_max_rad,
            max_try_time=MAX_TRY_TIME,
            check_collision=True,
            visualize=False
        )
        print(f"Generated {len(jnt_list)} waypoints for the cubic curve with scale: {scale}")
        success = (success_count == waypoints_num)

    assert len(jnt_list) == len(workspace_points), f"Error: only {len(jnt_list)} waypoints for {len(workspace_points)}"
    print(f"Average deviations: {np.mean(deviations):.2f} \nwith detailed statistics: {[f'{d:.2f}' for d in deviations]}")
    hf.workspace_plot(robot_s, jnt_list)
    hf.visualize_static_path(base, robot_s, jnt_list)
    