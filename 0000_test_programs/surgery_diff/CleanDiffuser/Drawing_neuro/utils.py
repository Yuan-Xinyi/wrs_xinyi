import time
import yaml
import numpy as np
from wrs import wd, rm, mcm
import matplotlib.pyplot as plt

def gen_jnt_list_from_pos_list(init_jnt, pos_list, robot, obstacle_list, base,
                                max_try_time=5.0, check_collision=True, visualize=False):
    jnt_list = []
    success_count = 0
    _, rotmat = robot.fk(jnt_values=init_jnt)

    for pos in pos_list:
        jnt = None
        start_time = time.time()
        try:
            while jnt is None and time.time() - start_time < max_try_time:
                j = robot.ik(tgt_pos=pos, tgt_rotmat=rotmat, seed_jnt_values=jnt_list[-1] if jnt_list else None)
                if j is None:
                    raise RuntimeError("IK failed")
                robot.goto_given_conf(j)
                if check_collision and robot.cc.is_collided(obstacle_list=obstacle_list):
                    raise RuntimeError("Collision detected")
                jnt = j
                success_count += 1
                if visualize:
                    mcm.mgm.gen_frame(pos=pos, rotmat=rotmat).attach_to(base)
                    robot.gen_meshmodel(alpha=.2).attach_to(base)
                break
            if jnt is None:
                raise RuntimeError("Unknown failure")
            jnt_list.append(jnt)
        except Exception as e:
            # print(f"{'-'*40}\nAborting: {success_count} / {len(pos_list)} positions succeeded.\n{'-'*40}")
            return [j for j in jnt_list if j is not None], success_count

    # print(f"{'-'*40}\nSuccessfully solved IK for {success_count} / {len(pos_list)} positions.\n{'-'*40}")
    return jnt_list, success_count


def plot_joint_and_workspace(robot, *jnt_lists, labels=None):
    arrays = [np.array(j) for j in jnt_lists]
    n_joints, T = arrays[0].shape[1], arrays[0].shape[0]

    if labels is None:
        labels = [f"Traj {i+1}" for i in range(len(arrays))]
    colors = plt.cm.tab10.colors

    # 创建子图：关节轨迹 (n_joints) + 工作空间 (3个轴)
    fig, axes = plt.subplots(n_joints + 3, 1, figsize=(10, 2*(n_joints+3)), sharex=True)
    fig.suptitle("Joint & Workspace Trajectories", fontsize=16)

    # === Joint trajectories ===
    for j in range(n_joints):
        for i, arr in enumerate(arrays):
            axes[j].plot(range(T), arr[:, j],
                         marker="o", markersize=3, linewidth=1.2,
                         color=colors[i % len(colors)],
                         label=labels[i] if j == 0 else None,
                         linestyle='-' if i == 0 else '--')
        axes[j].set_ylabel(f"J{j+1}")
        axes[j].grid(True)

    # === Workspace trajectories (X/Y/Z) ===
    pos_lists = [[robot.fk(jnt)[0] for jnt in path] for path in jnt_lists]
    for axis in range(3):
        ax_idx = n_joints + axis
        for i, pos_arr in enumerate(pos_lists):
            arr = np.array(pos_arr)
            axes[ax_idx].plot(range(arr.shape[0]), arr[:, axis],
                              color=colors[i % len(colors)],
                              label=labels[i] if axis == 0 else None,
                              linestyle='-' if i == 0 else '--')
        axes[ax_idx].set_ylabel(["X","Y","Z"][axis])
        axes[ax_idx].grid(True)

    axes[-1].set_xlabel("Time Step")
    axes[0].legend(loc="upper right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()



def partiallydiscretize_joint_space(robot, n_intervals=None):
    sampled_jnts = []
    if n_intervals is None:
        n_intervals = np.linspace(6, 4, robot.n_dof - 1, endpoint=False)

    print('=' * 100)
    print(f"Sampling Joint Space using the following joint granularity (excluding last DOF): {n_intervals.astype(int)}...")

    for i in range(robot.n_dof - 1):
        sampled_jnts.append(
            np.linspace(robot.jnt_ranges[i][0], robot.jnt_ranges[i][1], int(n_intervals[i] + 2))[1:-1]
        )

    grid = np.meshgrid(*sampled_jnts)
    base_qs = np.vstack([x.ravel() for x in grid]).T

    last_column = np.zeros((base_qs.shape[0], 1))  # 或 np.full((..., 1), np.nan)
    sampled_qs = np.hstack((base_qs, last_column))

    return sampled_qs

import numpy as np

def generate_random_cubic_curve(num_points=20, scale=0.1, start=None, equal_arc=False, resolution=1000):
    """
    随机生成一条三次多项式曲线，并在空间中采样点。
    起点锚定为 start（如果给定）。
    
    参数:
        num_points (int): 采样点数量
        scale (float): 随机系数缩放
        start (np.ndarray): 曲线起点 (3,)
        equal_arc (bool): 是否按弧长等间距采样
        resolution (int): 高分辨率采样数量，用于近似计算弧长

    返回:
        points (np.ndarray): shape=(num_points, 3)，空间轨迹点
        coeffs (np.ndarray): shape=(12,)，曲线系数
    """
    if start is None:
        start = np.array([0.4, 0.2, 0.3])  # 默认中心

    # 随机生成三次多项式的系数向量
    a = np.random.randn(3) * scale
    b = np.random.randn(3) * scale
    c = np.random.randn(3) * scale
    d = start
    coeffs = np.hstack([a, b, c, d])

    # 定义曲线函数
    def curve_func(t):
        return a * t**3 + b * t**2 + c * t + d

    if not equal_arc:
        t_vals = np.linspace(0, 1, num_points)
        points = np.array([curve_func(t) for t in t_vals])
    else:
        ts = np.linspace(0, 1, resolution)
        pts = np.array([curve_func(t) for t in ts])
        seglens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cumlen = np.insert(np.cumsum(seglens), 0, 0.0)
        total_len = cumlen[-1]

        target_lens = np.linspace(0, total_len, num_points)
        points = []
        for tl in target_lens:
            idx = np.searchsorted(cumlen, tl)
            if idx == 0:
                points.append(pts[0])
            elif idx >= len(ts):
                points.append(pts[-1])
            else:
                ratio = (tl - cumlen[idx-1]) / (cumlen[idx] - cumlen[idx-1] + 1e-12)
                p = pts[idx-1] * (1-ratio) + pts[idx] * ratio
                points.append(p)
        points = np.array(points)

    # 确保第一个点就是 start
    points[0] = start

    return points, coeffs
