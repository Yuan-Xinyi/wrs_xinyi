import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# 使用 Zarr 读取数据
import zarr
root = zarr.open('/home/lqin/zarr_datasets/franka_ruckig_100hz_fixgoal.zarr', mode='r')
traj_id = 100
traj_start = int(np.sum(root['meta']['episode_ends'][:traj_id]))
traj_end = int(np.sum(root['meta']['episode_ends'][:traj_id + 1]))
jnt_pos_list = root['data']['jnt_pos'][traj_start:traj_end]
jnt_vel_list = root['data']['jnt_vel'][traj_start:traj_end]
jnt_acc_list = root['data']['jnt_acc'][traj_start:traj_end]

# 模拟时间序列和关节位置轨迹
T = len(jnt_pos_list)
dt = 0.01
t = np.linspace(0, (T - 1) * dt, T)
x = jnt_pos_list[:, 0]

# 手动构建 B-Spline：选用三次样条 k=3
degree = 5
num_ctrl_pts = 100  # 我们希望的控制点数量
s = np.linspace(0, 1, T)

# 计算控制点：我们在数据上均匀采样
ctrl_points = np.linspace(0, 1, num_ctrl_pts)
ctrl_values = np.interp(ctrl_points, s, x)

# 手动指定 knots（确保满足 B-spline 条件）
knots = np.concatenate([
    np.zeros(degree),  # 前面的重复 knots
    np.linspace(0, 1, num_ctrl_pts - degree + 1),
    np.ones(degree)    # 后面的重复 knots
])

# 构建 B-spline
spline = BSpline(knots, ctrl_values, degree)

# 相位细分用于评估
s_fine = np.linspace(0, 1, 1000)
q_s = spline(s_fine)

# 计算 B-Spline 导数
dq_ds = spline.derivative(1)(s_fine)
d2q_ds2 = spline.derivative(2)(s_fine)
d3q_ds3 = spline.derivative(3)(s_fine)

# 将导数映射回时间空间（s -> t）
T_total = t[-1] - t[0]
r_s = 1 / T_total
q_t = q_s
dq_dt = dq_ds * r_s
d2q_dt2 = d2q_ds2 * (r_s ** 2)
d3q_dt3 = d3q_ds3 * (r_s ** 3)

t_fine = s_fine * T_total

# 绘图
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# 位置（Position）
axs[0].plot(t, jnt_pos_list[:, 0], 'o', label='Original Position', markersize=4)
axs[0].plot(t_fine, q_t, label='B-Spline Fit', linewidth=2)
axs[0].plot(ctrl_points * T_total, ctrl_values, 'x--', color='red', label='Control Points', markersize=8)
axs[0].set_title("Position $q(t)$")
axs[0].set_ylabel("Position")
axs[0].legend()

# 速度（Velocity）
axs[1].plot(t, jnt_vel_list[:, 0], 'o', label='Original Velocity', markersize=4)
axs[1].plot(t_fine, dq_dt, label='B-Spline Velocity', linewidth=2)
axs[1].set_title("Velocity $\\dot{q}(t)$")
axs[1].set_ylabel("Velocity")
axs[1].legend()

# 加速度（Acceleration）
axs[2].plot(t, jnt_acc_list[:, 0], 'o', label='Original Acceleration', markersize=4)
axs[2].plot(t_fine, d2q_dt2, label='B-Spline Acceleration', linewidth=2)
axs[2].set_title("Acceleration $\\ddot{q}(t)$")
axs[2].set_ylabel("Acceleration")
axs[2].legend()

# 抖动（Jerk）
dddqaxis = np.diff(jnt_acc_list, axis=0, prepend=jnt_acc_list[0, 0]) / dt
dddqaxis[0, :] = 0.0
dddqaxis[-1, :] = 0.0

axs[3].plot(t[:-1], dddqaxis[:-1, 0], 'o', label='Original Jerk', markersize=4)
axs[3].plot(t_fine, d3q_dt3, label='B-Spline Jerk', linewidth=2)
axs[3].set_title("Jerk $\\dddot{q}(t)$")
axs[3].set_ylabel("Jerk")
axs[3].set_xlabel("Time [s]")
axs[3].legend()

plt.tight_layout()
plt.show()
