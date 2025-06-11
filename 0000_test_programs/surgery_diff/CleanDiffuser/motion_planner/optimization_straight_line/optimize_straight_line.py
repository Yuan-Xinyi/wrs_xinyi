import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm

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
def cost_fn(q_all, path_points, num_joints, weight_smooth=1e-2, weight_rot_smooth=1e-1):
    q_all = q_all.reshape(len(path_points), num_joints)
    loss = 0.0
    rot_prev = None
    for i, (q, x_desired) in enumerate(zip(q_all, path_points)):
        x, rot = robot.fk(jnt_values=q)
        loss += np.linalg.norm(x - x_desired)**2  # 末端位置误差

        if i > 0:
            loss += weight_smooth * np.linalg.norm(q - q_all[i-1])**2  # 关节平滑
            rot_dist = calculate_rot_error(rot_prev, rot)
            loss += weight_rot_smooth * rot_dist**2  # 旋转平滑

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
    # gth_jnt_path, path_points = generate_jnt_path('x', num_points)  # 生成关节路径

    data = np.load("ik_traj_dataset.npy", allow_pickle=True)
    print(f"Loaded dataset with {len(data)} samples.")

    start_positions = [item["start_pos"] for item in data]
    goal_positions = [item["goal_pos"] for item in data]
    start_joints = [item["start_jnt"] for item in data]
    traj_lengths = [item["traj_len"] for item in data]
    print(f'gth average length: {np.mean(traj_lengths):.4f} m')

    '''random initial joint conf test'''
    init_pos_diff = []
    traj_len_list = []

    success_count = 0
    longer_traj_count = 0

    ts = time.time()

    import json

    output_file = open("result_per_sample.json", "w")

    # 统计量初始化
    init_pos_diff = []
    traj_len_list = []
    success_count = 0
    longer_traj_count = 0
    ts = time.time()

    for id in tqdm(range(len(data))):
    # for id in tqdm(range(2)):
        total_start = time.time()
        q_init = np.array([robot.rand_conf() for _ in range(num_points)])
        q_min = robot.jnt_ranges[:, 0]
        q_max = robot.jnt_ranges[:, 1]
        bounds = [(q_min[i % num_joints], q_max[i % num_joints]) for i in range(num_points * num_joints)]

        disp = np.abs(data[id]['goal_pos'] - data[id]['start_pos'])
        axis_idx = np.argmax(disp)

        path_points = np.tile(data[id]['start_pos'], (num_points, 1))
        path_points[:, axis_idx] = np.linspace(data[id]['start_pos'][axis_idx], data[id]['goal_pos'][axis_idx], num_points)

        # 优化开始
        opt_start = time.time()
        res = minimize(
            cost_fn,
            q_init.flatten(),
            args=(path_points, num_joints),
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': False, 'maxiter': 5000, 'gtol': 1e-5}
        )
        opt_end = time.time()

        opt_time = opt_end - opt_start
        total_time = time.time() - total_start

        q_traj = res.x.reshape(num_points, num_joints)
        pos_guess, rot = robot.fk(jnt_values=q_traj[0])
        real_jnt = robot.ik(tgt_pos=data[id]['start_pos'], tgt_rotmat=rot)

        traj_len = 0.0
        init_diff = float(np.linalg.norm(pos_guess - data[id]['start_pos']))
        interp_time = 0.0

        if real_jnt is not None:
            success_count += 1
            jnt_list = [real_jnt]
            gth_pos = data[id]['start_pos'].copy()

            interp_start = time.time()
            for _ in range(200):
                gth_pos[axis_idx] += 0.01
                jnt = robot.ik(tgt_pos=gth_pos, tgt_rotmat=rot, seed_jnt_values=jnt_list[-1])
                if jnt is not None:
                    jnt_list.append(jnt)
                else:
                    break
            interp_end = time.time()
            interp_time = interp_end - interp_start

            if len(jnt_list) > 1:
                pos_goal = robot.fk(jnt_values=jnt_list[-1])[0]
                traj_len = (len(jnt_list) - 1) * 0.01
                if traj_len > data[id]['traj_len']:
                    longer_traj_count += 1
                traj_len_list.append(traj_len)
                init_pos_diff.append(init_diff)

        # 写入 JSON 一行
        sample_result = {
            "sample_id": id,
            "traj_len": traj_len,
            "init_pos_diff": init_diff,
            "opt_time": opt_time,
            "interp_time": interp_time,
            "total_time": total_time
        }
        output_file.write(json.dumps(sample_result) + "\n")

    output_file.close()

    # 统计打印
    time_cost = time.time() - ts
    print(f"Time cost: {time_cost:.2f} seconds")
    print(success_count)
    print(f'Average time per sample: {time_cost / success_count:.4f} seconds')
    print(f"Total successful samples: {success_count / len(data) * 100:.2f}%")
    print(f"Average initial position difference: {np.sum(init_pos_diff) / len(data):.4f}")
    print(f"Average trajectory length: {np.sum(traj_len_list) / len(data):.4f}")
    print(f"Longer trajectory count: {longer_traj_count / len(data) * 100:.2f}%")
