import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

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

def cost_fn(q_all, path_points, num_joints, weight_smooth=1e-2, weight_rot_smooth=1e-1):
    q_all = q_all.reshape(len(path_points), num_joints)
    loss = 0.0
    rot_prev = None
    for i, (q, x_desired) in enumerate(zip(q_all, path_points)):
        x, rot = robot.fk(jnt_values=q)
        loss += np.linalg.norm(x - x_desired)**2
        if i > 0:
            loss += weight_smooth * np.linalg.norm(q - q_all[i-1])**2
            rot_dist = calculate_rot_error(rot_prev, rot)
            loss += weight_rot_smooth * rot_dist**2
        rot_prev = rot
    return loss

# === 固定 ground truth 轨迹 ===
num_points = 32
num_joints = robot.n_dof
gth_jnt_path, path_points = generate_jnt_path('x', num_points)
path_points = list(path_points)  # 确保可索引

# === 多次尝试不同的随机初始化 q_init，比较优化时间 ===
num_trials = 5
results = []

for trial in range(num_trials):
    np.random.seed(trial)  # 每次不同的随机初始值
    # q_init = np.array(gth_jnt_path) + np.random.normal(scale=0.05*trial, size=(num_points, num_joints))
    q_init = np.zeros((num_points, num_joints))

    q_min = robot.jnt_ranges[:, 0]
    q_max = robot.jnt_ranges[:, 1]
    bounds = [(q_min[i % num_joints], q_max[i % num_joints]) for i in range(num_points * num_joints)]

    start_time = time.time()
    res = minimize(
        cost_fn,
        q_init.flatten(),
        args=(path_points, num_joints),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': False, 'maxiter': 1000, 'gtol': 1e-4}
    )
    end_time = time.time()

    q_traj = res.x.reshape(num_points, num_joints)
    avg_l2 = np.mean(np.linalg.norm(q_traj - gth_jnt_path, axis=1))

    print(f"[Trial {trial}] Time: {end_time - start_time:.2f}s | Avg L2 error: {avg_l2:.4f}")
    results.append((trial, end_time - start_time, avg_l2))

# === 可视化结果 ===
trials, times, errors = zip(*results)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(trials, times, 'g-', label="Time (s)")
ax2.plot(trials, errors, 'b--', label="Avg L2 Error")

ax1.set_xlabel("Trial Index")
ax1.set_ylabel("Time (s)", color='g')
ax2.set_ylabel("Avg L2 Error", color='b')
ax1.set_title("Effect of Random Init q_init on Optimization Time and Accuracy")

plt.grid(True)
plt.show()
