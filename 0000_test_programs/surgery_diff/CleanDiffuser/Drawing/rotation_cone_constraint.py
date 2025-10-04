# import wrs.robot_con.xarm_lite6.xarm_lite6_x as xarm_x
import wrs.robot_sim.robots.xarmlite6_wg.x6wg2 as xarm_s
import wrs.robot_sim.robots.cobotta.cobotta as cbt
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
from scipy.spatial.transform import Rotation as sR


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

def sample_rotations_in_cone(a_axis, alpha_max_rad, n_alpha=3, n_psi=12, bias_u=None):
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

from scipy.spatial.transform import Rotation as sR
import numpy as np
import time

def _sample_similar_rotations(R_prev, max_rot_diff, n_samples=20):
    """
    在上一帧旋转矩阵 R_prev 的基础上，采样若干旋转矩阵，
    每个姿态与 R_prev 的旋转差异 <= max_rot_diff。
    """
    R_list = []
    for _ in range(n_samples):
        # 随机旋转轴
        axis = np.random.randn(3)
        axis /= (np.linalg.norm(axis) + 1e-12)
        # 随机旋转角度
        theta = np.random.uniform(0, max_rot_diff)
        # 构造旋转
        R_delta = sR.from_rotvec(axis * theta).as_matrix()
        # 新姿态
        R_cand = R_prev @ R_delta
        R_list.append(R_cand)
    return R_list


def gen_jnt_list_from_pos_list_relaxed(
        init_jnt, pos_list, robot, obstacle_list, base,
        alpha_max_rad,
        n_alpha=3, n_psi=12,
        tool_axis='z',
        plane_normal=None,            
        max_try_time=20.0,
        max_rot_diff=np.deg2rad(15.0),   # 最大允许旋转差异
        check_collision=True,
        visualize=False):
    """
    在平面法线约束下的 relaxed 逆解轨迹生成：
    - 第一个点：基于 plane_normal/tool_axis 做 cone 采样；
    - 后续点：基于 R_prev 邻域做采样，确保旋转差异 ≤ max_rot_diff。
    """
    jnt_list = []
    success_count = 0
    deviations = []
    R_prev = None

    # 1. 确定锥轴（只用于第一个点）
    if plane_normal is not None:
        a_axis = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    else:
        _, rot0 = robot.fk(jnt_values=init_jnt)
        if tool_axis == 'z':
            a_axis = rot0[:, 2]
        elif tool_axis == 'x':
            a_axis = rot0[:, 0]
        elif tool_axis == 'y':
            a_axis = rot0[:, 1]
        else:
            raise ValueError("tool_axis must be one of {'x','y','z'}")
        a_axis = a_axis / (np.linalg.norm(a_axis) + 1e-12)

    # 2. 遍历所有目标位置
    for idx, pos in enumerate(pos_list):
        jnt = None
        start_time = time.time()

        # 采样候选旋转
        if idx == 0:  # 第一个点：用锥体采样
            R_cands = sample_rotations_in_cone(
                a_axis=a_axis,
                alpha_max_rad=alpha_max_rad,
                n_alpha=n_alpha,
                n_psi=n_psi
            )
        else:  # 后续点：在 R_prev 附近采样
            R_cands = _sample_similar_rotations(R_prev, max_rot_diff, n_samples=30)
        
        jnt_list = []
        for R_cand in R_cands:
            if time.time() - start_time >= max_try_time:
                break
            
            '''debug'''
            jnt = robot.ik(tgt_pos=pos, tgt_rotmat=R_cand)
            if jnt is None:
                continue
            else:
                jnt_list.append(jnt)
                print('ik soulution found! as {}'.format(jnt))
        return jnt_list

    #         seed = jnt_list[-1] if jnt_list else init_jnt
    #         j = robot.ik(tgt_pos=pos, tgt_rotmat=R_cand, seed_jnt_values=seed)
    #         if j is None:
    #             continue
    #         robot.goto_given_conf(j)
    #         if check_collision and robot.cc.is_collided(obstacle_list=obstacle_list):
    #             continue

    #         # 旋转差异（与上一帧比）
    #         if R_prev is not None:
    #             R_delta = sR.from_matrix(R_prev.T @ R_cand)
    #             theta = R_delta.magnitude()
    #         else:
    #             theta = 0.0

    #         # 成功
    #         jnt = j
    #         success_count += 1
    #         jnt_list.append(jnt)
    #         deviations.append(theta)
    #         R_prev = R_cand.copy()

    #         if visualize:
    #             mcm.mgm.gen_frame(pos=pos, rotmat=R_cand).attach_to(base)
    #             robot.gen_meshmodel(alpha=.2).attach_to(base)
    #         break

    #     if jnt is None:  # 当前点失败，提前结束
    #         return [j for j in jnt_list if j is not None], success_count, deviations

    # return jnt_list, success_count, deviations

def visualize_workspace_points_in_world(base, points, radius=0.01, rgb=(1,0,0)):
    for p in points:
        sphere = mgm.gen_sphere(pos=p, radius=radius, rgb=rgb)
        sphere.attach_to(base)

def plane_normal_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    根据三个点计算平面的法向量（归一化）。
    参数:
        p1, p2, p3: np.ndarray, shape=(3,)
    返回:
        n: np.ndarray, shape=(3,), 单位法向量
    """
    v1 = np.asarray(p2) - np.asarray(p1)
    v2 = np.asarray(p3) - np.asarray(p1)
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("Error: The three points are collinear or too close to each other.")
    return n / norm

def _generate_paper_plane_from_axis(a_axis, center, size_x=0.3, size_y=0.3, thickness=0.001,
                                   color=[1.0, 1.0, 1.0], alpha=0.7):
    """
    根据平面法线 a_axis 生成一张矩形纸平面。

    参数:
        a_axis: 平面法向 (3,)
        center: 平面中心 (3,)
        size_x, size_y: 平面的长和宽
        thickness: 厚度
        color: 颜色
        alpha: 透明度
    """
    normal = a_axis / (np.linalg.norm(a_axis) + 1e-12)

    # 找到和 normal 正交的一个方向作为 x 轴
    tmp = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_dir = np.cross(normal, tmp)
    x_dir /= (np.linalg.norm(x_dir) + 1e-12)

    # y 轴 = normal × x_dir
    y_dir = np.cross(normal, x_dir)
    y_dir /= (np.linalg.norm(y_dir) + 1e-12)

    # 构造旋转矩阵
    rotmat = np.stack([x_dir, y_dir, normal], axis=1)

    # 平面位置（稍微往 normal 反方向移半个厚度，保证中心落在平面上）
    pos = center - normal * (thickness / 2.0)

    # 生成薄片
    size = np.array([size_x, size_y, thickness])
    paper = mcm.gen_box(
        xyz_lengths=size,
        pos=pos,
        rotmat=rotmat,
        rgb=color,
        alpha=alpha
    )
    paper.attach_to(base)
    return paper


MAX_TRY_TIME = 100.0
alpha_max_rad = np.deg2rad(10.0)
waypoints_num = 8
scale = 0.05
success = False

if __name__ == "__main__":
    '''Initialize the world and robot'''
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot_s = xarm_s.XArmLite6WG2(enable_cc=True)
    # robot_s = cbt.Cobotta(enable_cc=True)
    
    while not success:
        is_collided = True

        while is_collided:
            print("Initial configuration in collision, sampling a new one...")
            init_jnt = robot_s.rand_conf()
            is_collided = robot_s.cc.is_collided(obstacle_list=[])

        pos_init, rotmat_init = robot_s.fk(jnt_values=init_jnt)
        robot_s.goto_given_conf(init_jnt)
        robot_s.gen_meshmodel().attach_to(base)

        # workspace_points, coeffs = utils.generate_random_cubic_curve(num_points=waypoints_num, scale=scale, center=pos_init)
        workspace_points, coeffs = utils.generate_random_cubic_curve(
                                                                    num_points=waypoints_num,
                                                                    scale=scale,
                                                                    start=pos_init,      # 保证起点是当前正向运动学的位置
                                                                    equal_arc=True
                                                                )

        # visualize_workspace_points_in_world(base, workspace_points, radius=0.01, rgb=(1,1,0))
        # base.run()
        a_axis = plane_normal_from_points(workspace_points[0], workspace_points[1], workspace_points[2])
        center = np.mean(workspace_points, axis=0)  # 取曲线点的平均值作为纸中心
        _generate_paper_plane_from_axis(a_axis, center, size_x=0.4, size_y=0.4, thickness=0.001,
                                        color=[1.0, 1.0, 1.0], alpha=0.7)
        jnt_list, success_count, deviations = gen_jnt_list_from_pos_list_relaxed(
            init_jnt=init_jnt,
            pos_list=workspace_points,
            robot=robot_s,
            obstacle_list=None,
            base=base,
            alpha_max_rad=np.deg2rad(10),
            max_rot_diff=np.deg2rad(10),  # 限制相邻姿态旋转差异 ≤ 10°
            check_collision=True,
            visualize=False
        )
        print(f"Generated {len(jnt_list)} waypoints for the cubic curve with scale: {scale}")
        success = (success_count == waypoints_num)

    visualize_workspace_points_in_world(base, workspace_points, radius=0.01, rgb=(1,1,0))
    assert len(jnt_list) == len(workspace_points), f"Error: only {len(jnt_list)} waypoints for {len(workspace_points)}"
    print(f"Average deviations: {np.rad2deg(np.mean(deviations)):.2f} \nwith detailed statistics: {[f'{np.rad2deg(d):.2f}' for d in deviations]}")
    hf.workspace_plot(robot_s, jnt_list)

    # hf.visualize_static_path(base, robot_s, jnt_list)
    hf.visualize_anime_path(base, robot_s, jnt_list)
    