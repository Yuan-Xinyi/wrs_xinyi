import numpy as np
import matplotlib.pyplot as plt

import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from wrs import wd, rm
import wrs.modeling.geometric_model as mgm

# 初始化世界与机器人
robot = franka.FrankaResearch3(enable_cc=True)
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)

def generate_jnt_path(axis, num_points, max_attempts=100):
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    for _ in range(max_attempts):
        q_start = robot.rand_conf()
        pos_init, rotmat = robot.fk(jnt_values=q_start)
        jnt_list = [q_start]
        pos_list = [pos_init]
        pos = pos_init.copy()
        for i in range(1, num_points):
            pos[axis_idx] += 0.01
            jnt = robot.ik(tgt_pos=pos, tgt_rotmat=rotmat, seed_jnt_values=jnt_list[-1])
            if jnt is None:
                break
            jnt_list.append(jnt)
            pos_list.append(pos.copy())
        if len(jnt_list) == num_points:
            return jnt_list, pos_list
    raise RuntimeError(f"Failed to generate a valid joint path after {max_attempts} attempts.")

# RBF核函数
def rbf_kernel(t1, t2, lengthscale=1.0, variance=1.0):
    sqdist = np.subtract.outer(t1, t2) ** 2
    return variance * np.exp(-0.5 * sqdist / lengthscale ** 2)

# GPMP轨迹优化（简化）
def gpmp_trajectory_optimize(gth_jnt_path, robot, num_points=32, lengthscale=0.1, variance=1.0, noise=1e-6):
    num_joints = robot.n_dof
    time_steps = np.linspace(0, 1, num_points)
    Y = np.array(gth_jnt_path)  # shape: (T, 7)
    K = rbf_kernel(time_steps, time_steps, lengthscale, variance)
    K_inv = np.linalg.inv(K + noise * np.eye(num_points))
    q_gp = []
    for j in range(num_joints):
        y = Y[:, j]
        alpha = K_inv @ y
        mean = K @ alpha
        q_gp.append(mean)
    return np.stack(q_gp, axis=1)

def traj_comparison_multi(*joint_seqs, labels=None):
    n_seqs = len(joint_seqs)
    T = joint_seqs[0].shape[0]
    if labels is None:
        labels = [f"Traj {i+1}" for i in range(n_seqs)]
    time = np.arange(T)
    colors = plt.cm.tab10.colors
    fig, axs = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    fig.suptitle("Multi-Trajectory Joint Comparison", fontsize=16)
    for j in range(7):
        for i, seq in enumerate(joint_seqs):
            axs[j].plot(time, seq[:, j], label=labels[i],
                        color=colors[i % len(colors)], linestyle='-' if i == 0 else '--')
        axs[j].set_ylabel(f"Joint {j+1}")
        axs[j].grid(True)
        if j == 0:
            axs[j].legend(loc="upper right")
    axs[-1].set_xlabel("Time Step")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def workspace_plot_multi(robot, *jnt_paths, labels=None):
    T = jnt_paths[0].shape[0]
    if labels is None:
        labels = [f"Traj {i+1}" for i in range(len(jnt_paths))]
    colors = plt.cm.tab10.colors
    pos_lists = []
    for path in jnt_paths:
        pos_list = [robot.fk(jnt)[0] for jnt in path]
        pos_lists.append(np.array(pos_list))
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis in range(3):
        for i, pos_arr in enumerate(pos_lists):
            axs[axis].plot(pos_arr[:, axis], label=labels[i], color=colors[i % len(colors)], linestyle='-' if i == 0 else '--')
        axs[axis].set_ylabel(['X', 'Y', 'Z'][axis])
        axs[axis].grid(True)
        if axis == 0:
            axs[axis].legend()
    axs[-1].set_xlabel("Time Steps")
    fig.suptitle("Workspace Trajectories - End Effector Positions (X/Y/Z)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# 运行主程序
gth_jnt_path, path_points = generate_jnt_path('x', 32)
init_traj = np.zeros((32, robot.n_dof))
q_gp = gpmp_trajectory_optimize(init_traj, robot)

similarity = np.mean(np.linalg.norm(q_gp - gth_jnt_path, axis=1))
print(f"GPMP Trajectory average l2 distance: {similarity:.4f}")

traj_comparison_multi(np.array(gth_jnt_path), q_gp, labels=["Ground Truth", "GPMP Trajectory"])
workspace_plot_multi(robot, np.array(gth_jnt_path), q_gp, labels=["Ground Truth", "GPMP Trajectory"])

robot.goto_given_conf(gth_jnt_path[0])
robot.gen_meshmodel(rgb=[0,1,0], alpha=0.3).attach_to(base)
robot.goto_given_conf(gth_jnt_path[-1])
robot.gen_meshmodel(rgb=[0,1,0], alpha=0.3).attach_to(base)

robot.goto_given_conf(q_gp[0])
robot.gen_meshmodel(rgb=[0,0,1], alpha=0.3).attach_to(base)
robot.goto_given_conf(q_gp[-1])
robot.gen_meshmodel(rgb=[0,0,1], alpha=0.3).attach_to(base)

base.run()
