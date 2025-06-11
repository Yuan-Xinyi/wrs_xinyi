import numpy as np
import torch
from torch.optim import LBFGS
import matplotlib.pyplot as plt

import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm

# 初始化机器人和可视化环境
robot = franka.FrankaResearch3(enable_cc=True)
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)

# 生成路径点
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

def calculate_rot_error(rot1, rot2):
    delta = rm.delta_w_between_rotmat(rot1, rot2)
    return np.linalg.norm(delta)

# PyTorch Cost Function
def torch_cost_fn(q_all, target_positions, weight_smooth=1e-4, weight_rot_smooth=1e-4):
    loss = torch.tensor(0.0, dtype=torch.float32)
    rot_prev = None
    for i in range(len(target_positions)):
        q_i = q_all[i]
        pos_i, rot_i = robot.fk(jnt_values=q_i.detach().cpu().numpy())
        pos_i = torch.tensor(pos_i, dtype=torch.float32)
        target = torch.tensor(target_positions[i], dtype=torch.float32)
        loss += 1*torch.norm(pos_i - target) ** 2
        # loss += torch.norm(pos_i - target, p=1)  # l1 norm

        if i > 0:
            loss += weight_smooth * torch.norm(q_all[i] - q_all[i-1]) ** 2
            rot_dist = calculate_rot_error(rot_prev, rot_i)
            loss += weight_rot_smooth * rot_dist ** 2
        rot_prev = rot_i
    return loss

# 可视化工具函数
def traj_comparison_multi(*joint_seqs, labels=None):
    n_seqs = len(joint_seqs)
    assert n_seqs >= 2
    T = joint_seqs[0].shape[0]
    for seq in joint_seqs:
        assert seq.shape == (T, 7)
    if labels is None:
        labels = [f"Traj {i+1}" for i in range(n_seqs)]
    time = np.arange(T)
    colors = plt.cm.tab10.colors
    fig, axs = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    fig.suptitle("Multi-Trajectory Joint Comparison", fontsize=16)
    for j in range(7):
        for i, seq in enumerate(joint_seqs):
            axs[j].plot(time, seq[:, j], label=labels[i],
                        color=colors[i % len(colors)],
                        linestyle='-' if i == 0 else '--')
        axs[j].set_ylabel(f"Joint {j+1}")
        axs[j].grid(True)
        if j == 0:
            axs[j].legend(loc="upper right")
    axs[-1].set_xlabel("Time Step")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def workspace_plot_multi(robot, *jnt_paths, labels=None):
    n_paths = len(jnt_paths)
    T = jnt_paths[0].shape[0]
    if labels is None:
        labels = [f"Traj {i+1}" for i in range(n_paths)]
    colors = plt.cm.tab10.colors
    pos_lists = []
    for path in jnt_paths:
        pos_list = []
        for jnt in path:
            pos, _ = robot.fk(jnt)
            pos_list.append(pos)
        pos_lists.append(np.array(pos_list))
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes_labels = ['X', 'Y', 'Z']
    for axis in range(3):
        for i, pos_arr in enumerate(pos_lists):
            axs[axis].plot(pos_arr[:, axis], label=labels[i],
                           color=colors[i % len(colors)],
                           linestyle='-' if i == 0 else '--')
        axs[axis].set_ylabel(f'Position ({axes_labels[axis]})')
        axs[axis].set_title(f'{axes_labels[axis]} Axis')
        axs[axis].grid(True)
        if axis == 0:
            axs[axis].legend()
    axs[-1].set_xlabel("Time Steps")
    fig.suptitle("Workspace Trajectories - End Effector Positions (X/Y/Z)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# 参数设置
num_points = 32
num_joints = robot.n_dof
gth_jnt_path, path_points = generate_jnt_path('x', num_points)
q_gt = np.array(gth_jnt_path)
q_init = q_gt + np.random.normal(scale=0.05, size=(num_points, num_joints))
q_torch = torch.tensor(q_init, dtype=torch.float32, requires_grad=True)

# 优化
optimizer = LBFGS([q_torch], max_iter=100, lr=1.0, line_search_fn='strong_wolfe')

def closure():
    optimizer.zero_grad()
    loss = torch_cost_fn(q_torch, path_points)
    loss.backward()
    return loss

import time
start = time.time()
optimizer.step(closure)
end = time.time()
print(f"L-BFGS optimization finished in {end - start:.2f} seconds.")

# 结果对比
q_optimized = q_torch.detach().numpy()
similarity = np.mean(np.linalg.norm(q_optimized - q_gt, axis=1))
print(f"Avg L2 difference vs ground truth: {similarity:.4f}")
traj_comparison_multi(q_gt, q_optimized, q_init, labels=["Ground Truth", "Optimized", "Init"])
workspace_plot_multi(robot, q_gt, q_optimized, labels=["Ground Truth", "Optimized"])

# 可视化机器人起止位姿
robot.goto_given_conf(q_gt[0])
robot.gen_meshmodel(rgb=[0,1,0], alpha=0.3).attach_to(base)
robot.goto_given_conf(q_gt[-1])
robot.gen_meshmodel(rgb=[0,1,0], alpha=0.3).attach_to(base)
robot.goto_given_conf(q_optimized[0])
robot.gen_meshmodel(rgb=[0,0,1], alpha=0.3).attach_to(base)
robot.goto_given_conf(q_optimized[-1])
robot.gen_meshmodel(rgb=[0,0,1], alpha=0.3).attach_to(base)
base.run()
