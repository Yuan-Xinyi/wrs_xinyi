import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm

robot = franka.FrankaResearch3(enable_cc=True)
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)

import time
fk_call_count = 0
fk_total_time = 0.0

def generate_jnt_path(axis, num_points, max_attempts=100):
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]

    for _ in range(max_attempts):
        q_start = robot.rand_conf()
        pos_init, rotmat = robot.fk(jnt_values=q_start)

        jnt_list = [q_start]
        pos_list = [pos_init]

        pos = pos_init.copy()  # 初始化后直接增量更新

        for i in range(1, num_points):
            pos[axis_idx] += 0.01  # 原地修改，节省拷贝
            jnt = robot.ik(tgt_pos=pos, tgt_rotmat=rotmat, seed_jnt_values=jnt_list[-1])
            if jnt is None:
                break
            jnt_list.append(jnt)
            pos_list.append(pos.copy())  # 防止 pos 被覆盖

        if len(jnt_list) == num_points:
            return jnt_list, pos_list

    raise RuntimeError(f"Failed to generate a valid joint path after {max_attempts} attempts.")

def calculate_rot_error(rot1, rot2):
    delta = rm.delta_w_between_rotmat(rot1, rot2)
    return np.linalg.norm(delta)

# 代价函数：末端误差 + 平滑项
# def cost_fn(q_all, path_points, num_joints, weight_smooth=1e-2, weight_rot_smooth=1e-1):
#     q_all = q_all.reshape(len(path_points), num_joints)
#     loss = 0.0
#     rot_prev = None
#     for i, (q, x_desired) in enumerate(zip(q_all, path_points)):
#         x, rot = robot.fk(jnt_values=q)
#         loss += np.linalg.norm(x - x_desired)**2  # 末端位置误差

#         if i > 0:
#             loss += weight_smooth * np.linalg.norm(q - q_all[i-1])**2  # 关节平滑
#             rot_dist = calculate_rot_error(rot_prev, rot)
#             loss += weight_rot_smooth * rot_dist**2  # 旋转平滑

#         rot_prev = rot

#     return loss
def cost_fn(q_all, path_points, num_joints, weight_smooth=1e-2, weight_rot_smooth=1e-1):
    global fk_total_time, fk_call_count  # 用于累计时间和调用次数

    q_all = q_all.reshape(len(path_points), num_joints)
    loss = 0.0
    rot_prev = None

    for i, (q, x_desired) in enumerate(zip(q_all, path_points)):
        t0 = time.time()
        x, rot = robot.fk(jnt_values=q)
        fk_total_time += time.time() - t0
        fk_call_count += 1

        loss += np.linalg.norm(x - x_desired)**2
        if i > 0:
            loss += weight_smooth * np.linalg.norm(q - q_all[i-1])**2
            rot_dist = calculate_rot_error(rot_prev, rot)
            loss += weight_rot_smooth * rot_dist**2
        rot_prev = rot

    return loss



def traj_comparison_multi(*joint_seqs, labels=None):
    """
    对比多个关节轨迹，每个轨迹形状应为 (T, 7)
    参数:
        *joint_seqs: 任意个 np.ndarray 形状 (T, 7)
        labels: list[str]，每个轨迹的标签（默认为 ["Traj 1", "Traj 2", ...] 
    """
    n_seqs = len(joint_seqs)
    assert n_seqs >= 2, "需要至少两个轨迹用于比较"
    T = joint_seqs[0].shape[0]
    
    for seq in joint_seqs:
        assert seq.shape == (T, 7), "每个输入轨迹必须是 (T, 7)"

    if labels is None:
        labels = [f"Traj {i+1}" for i in range(n_seqs)]
    assert len(labels) == n_seqs, "标签数量必须和轨迹数量一致"

    time = np.arange(T)
    colors = plt.cm.tab10.colors  # 支持最多10个颜色

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
    assert n_paths >= 1, "至少需要一条轨迹"
    T = jnt_paths[0].shape[0]
    for path in jnt_paths:
        assert path.shape == (T, robot.n_dof), "轨迹尺寸必须一致"

    if labels is None:
        labels = [f"Traj {i+1}" for i in range(n_paths)]
    assert len(labels) == n_paths, "标签数量需与轨迹数量一致"

    colors = plt.cm.tab10.colors

    # 计算每条轨迹的末端位置列表
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
            axs[axis].plot(
                pos_arr[:, axis],
                label=labels[i],
                color=colors[i % len(colors)],
                linestyle='-' if i == 0 else '--'
            )
        axs[axis].set_ylabel(f'Position ({axes_labels[axis]})')
        axs[axis].set_title(f'{axes_labels[axis]} Axis')
        axs[axis].grid(True)
        if axis == 0:
            axs[axis].legend()

    axs[-1].set_xlabel("Time Steps")
    fig.suptitle("Workspace Trajectories - End Effector Positions (X/Y/Z)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # 参数
    num_joints = robot.n_dof
    num_points = 32
    gth_jnt_path, path_points = generate_jnt_path('x', num_points)  # 生成关节路径
    q_init = np.zeros((num_points, num_joints))  # 初始猜测
    # q_init = np.array(gth_jnt_path) + np.random.normal(scale=0.05, size=(num_points, num_joints))


    # joint limits
    q_min = robot.jnt_ranges[:, 0]
    q_max = robot.jnt_ranges[:, 1]
    bounds = [(q_min[i % num_joints], q_max[i % num_joints]) for i in range(num_points * num_joints)]

    # 优化
    import time
    start_time = time.time()
    res = minimize(
        cost_fn,
        q_init.flatten(),
        args=(path_points, num_joints),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True, 'maxiter': 1000, 'gtol': 1e-4}
    )
    end_time = time.time()
    print(f"Optimization took {end_time - start_time:.2f} seconds")

    q_traj = res.x.reshape(num_points, num_joints)
    similarity = np.mean(np.linalg.norm(q_traj - gth_jnt_path, axis=1))  # 平均 L2，越小越相似
    print(f"Optimization completed with average l2 norm: {similarity:.4f}")
    traj_comparison_multi(np.array(gth_jnt_path), q_traj, q_init, labels=["Ground Truth", "Optimized", "Init Guess"])
    workspace_plot_multi(robot, np.array(gth_jnt_path), q_traj, labels=["Ground Truth", "Optimized"])

    print("===== Timing Analysis =====")
    print(f"Total FK calls: {fk_call_count}")
    print(f"Total FK time: {fk_total_time:.4f} sec")
    print(f"Average FK time per call: {fk_total_time / fk_call_count:.8f} sec")
    print("===========================")

    robot.goto_given_conf(gth_jnt_path[0])
    robot.gen_meshmodel(rgb=[0,1,0], alpha=0.5).attach_to(base)
    robot.goto_given_conf(gth_jnt_path[-1])
    robot.gen_meshmodel(rgb=[0,1,0], alpha=0.5).attach_to(base)

    robot.goto_given_conf(q_traj[0])
    robot.gen_meshmodel(rgb=[0,0,1], alpha=0.5).attach_to(base)
    robot.goto_given_conf(q_traj[-1])
    robot.gen_meshmodel(rgb=[0,0,1], alpha=0.5).attach_to(base)
    base.run()