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

def generate_random_cubic_curve(num_points=20, scale=0.1, center=np.array([0.4, 0.2, 0.3])):
    # 随机生成 4 个三维系数向量
    a = np.random.randn(3) * scale
    b = np.random.randn(3) * scale
    c = np.random.randn(3) * scale
    d = center
    coeffs = np.hstack([a, b, c, d])  # 展平为 shape=(12,)

    # 生成空间轨迹点
    t_vals = np.linspace(0, 1, num_points)
    points = [a * t**3 + b * t**2 + c * t + d for t in t_vals]

    return np.array(points), coeffs