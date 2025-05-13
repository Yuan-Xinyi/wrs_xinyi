'''direct b-spline'''
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import make_lsq_spline

# # 使用 Zarr 读取数据
# import zarr
# root = zarr.open('/home/lqin/zarr_datasets/franka_ruckig_100hz_fixgoal.zarr', mode='r')
# traj_id = 12
# traj_start = int(np.sum(root['meta']['episode_ends'][:traj_id]))
# traj_end = int(np.sum(root['meta']['episode_ends'][:traj_id + 1]))
# jnt_pos_list = root['data']['jnt_pos'][traj_start:traj_end]
# jnt_vel_list = root['data']['jnt_vel'][traj_start:traj_end]
# jnt_acc_list = root['data']['jnt_acc'][traj_start:traj_end]

# # 模拟时间序列和关节位置轨迹
# T = len(jnt_pos_list)
# dt = 0.01
# t = np.linspace(0, (T - 1) * dt, T)
# num_joints = jnt_pos_list.shape[1]  # 多个关节

# # 参数化变量 s
# s = np.linspace(0, 1, T)

# # 设置 B-Spline 参数
# degree = 4
# num_ctrl_pts = 64
# ctrl_points = np.linspace(0, 1, num_ctrl_pts)
# knots = np.linspace(0, 1, num_ctrl_pts - degree + 1)
# knots = np.concatenate(([0] * degree, knots, [1] * degree))

# # 多维 B-Spline 拟合
# ctrl_values_list = []

# for j in range(num_joints):
#     x = jnt_pos_list[:, j]
#     ctrl_values = np.interp(ctrl_points, s, x)
#     ctrl_values[0] = x[0]  # 确保起始对齐
#     ctrl_values[-1] = x[-1]  # 确保终止对齐
#     ctrl_values_list.append(ctrl_values)

# # 将控制点转为二维数组（每列是一个关节）
# ctrl_values_matrix = np.vstack(ctrl_values_list).T

# # 构建多维 B-Spline
# spline = make_lsq_spline(s, jnt_pos_list, knots, degree)

# # 相位细分用于评估
# s_fine = np.linspace(0, 1, 1000)
# q_s = spline(s_fine)

# # 计算 B-Spline 导数（多维）
# dq_ds = spline.derivative(1)(s_fine)
# d2q_ds2 = spline.derivative(2)(s_fine)
# d3q_ds3 = spline.derivative(3)(s_fine)

# # 将导数映射回时间空间（s -> t）
# T_total = t[-1] - t[0]
# r_s = 1 / T_total
# q_t = q_s
# dq_dt = dq_ds * r_s
# d2q_dt2 = d2q_ds2 * (r_s ** 2)
# d3q_dt3 = d3q_ds3 * (r_s ** 3)

# t_fine = s_fine * T_total

# # 绘制：每个关节单独绘制
# fig, axs = plt.subplots(num_joints, 4, figsize=(16, 4 * num_joints), sharex=True)

# for j in range(num_joints):
#     # 位置（Position）
#     axs[j, 0].plot(t, jnt_pos_list[:, j], 'o', label='Original Position', markersize=4)
#     axs[j, 0].plot(t_fine, q_t[:, j], label='B-Spline Fit', linewidth=2)
#     axs[j, 0].plot(ctrl_points * T_total, ctrl_values_matrix[:, j], 'x--', color='red', label='Control Points', markersize=8)
#     axs[j, 0].set_title(f"Position $q_{j}(t)$")
#     axs[j, 0].set_ylabel("Position")
#     axs[j, 0].legend()

#     # 速度（Velocity）
#     axs[j, 1].plot(t, jnt_vel_list[:, j], 'o', label='Original Velocity', markersize=4)
#     axs[j, 1].plot(t_fine, dq_dt[:, j], label='B-Spline Velocity', linewidth=2)
#     axs[j, 1].set_title(f"Velocity $\\dot{{q}}_{j}(t)$")
#     axs[j, 1].set_ylabel("Velocity")
#     axs[j, 1].legend()

#     # 加速度（Acceleration）
#     axs[j, 2].plot(t, jnt_acc_list[:, j], 'o', label='Original Acceleration', markersize=4)
#     axs[j, 2].plot(t_fine, d2q_dt2[:, j], label='B-Spline Acceleration', linewidth=2)
#     axs[j, 2].set_title(f"Acceleration $\\ddot{{q}}_{j}(t)$")
#     axs[j, 2].set_ylabel("Acceleration")
#     axs[j, 2].legend()

#     # 抖动（Jerk）
#     dddqaxis = np.diff(jnt_acc_list[:, j], axis=0, prepend=jnt_acc_list[0, j]) / dt
#     dddqaxis[0] = 0.0
#     dddqaxis[-1] = 0.0

#     axs[j, 3].plot(t[1:-1], dddqaxis[1:-1], 'o', label='Original Jerk', markersize=4)
#     axs[j, 3].plot(t_fine, d3q_dt3[:, j], label='B-Spline Jerk', linewidth=2)
#     axs[j, 3].set_title(f"Jerk $\\dddot{{q}}_{j}(t)$")
#     axs[j, 3].set_ylabel("Jerk")
#     axs[j, 3].legend()

# plt.tight_layout()
# plt.show()



'''reconstruct b-spline'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline
from scipy.interpolate import make_lsq_spline, BSpline

# 使用 Zarr 读取数据
import zarr
root = zarr.open('/home/lqin/zarr_datasets/franka_ruckig_100hz_fixgoal.zarr', mode='r')
traj_id = 12
traj_start = int(np.sum(root['meta']['episode_ends'][:traj_id]))
traj_end = int(np.sum(root['meta']['episode_ends'][:traj_id + 1]))
jnt_pos_list = root['data']['jnt_pos'][traj_start:traj_end]
jnt_vel_list = root['data']['jnt_vel'][traj_start:traj_end]
jnt_acc_list = root['data']['jnt_acc'][traj_start:traj_end]

# 模拟时间序列和关节位置轨迹
T = len(jnt_pos_list)
dt = 0.01
t = np.linspace(0, (T - 1) * dt, T)
num_joints = jnt_pos_list.shape[1]  # 多个关节

# 参数化变量 s
s = np.linspace(0, 1, T)

# 设置 B-Spline 参数
degree = 4
num_ctrl_pts = 64
ctrl_points = np.linspace(0, 1, num_ctrl_pts)
knots = np.linspace(0, 1, num_ctrl_pts - degree + 1)
knots = np.concatenate(([0] * degree, knots, [1] * degree))

# 多维 B-Spline 拟合
ctrl_values_list = []

for j in range(num_joints):
    x = jnt_pos_list[:, j]
    ctrl_values = np.interp(ctrl_points, s, x)
    ctrl_values[0] = x[0]  # 确保起始对齐
    ctrl_values[-1] = x[-1]  # 确保终止对齐
    ctrl_values_list.append(ctrl_values)

# 将控制点转为二维数组（每列是一个关节）
ctrl_values_matrix = np.vstack(ctrl_values_list).T

# 构建多维 B-Spline
spline = make_lsq_spline(s, jnt_pos_list, knots, degree)

'''recons b-spline'''
original_c = spline.c.copy()
original_knots = spline.t.copy()
original_degree = spline.k

# 通过控制点重建 B-Spline
spline = BSpline(original_knots, original_c, original_degree)

# 相位细分用于评估
s_fine = np.linspace(0, 1, 1000)
q_s = spline(s_fine)

# 计算 B-Spline 导数（多维）
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

# 绘制：每个关节单独绘制
fig, axs = plt.subplots(num_joints, 4, figsize=(16, 4 * num_joints), sharex=True)

for j in range(num_joints):
    # 位置（Position）
    axs[j, 0].plot(t, jnt_pos_list[:, j], 'o', label='Original Position', markersize=4)
    axs[j, 0].plot(t_fine, q_t[:, j], label='B-Spline Fit', linewidth=2)
    axs[j, 0].plot(ctrl_points * T_total, ctrl_values_matrix[:, j], 'x--', color='red', label='Control Points', markersize=8)
    axs[j, 0].set_title(f"Position $q_{j}(t)$")
    axs[j, 0].set_ylabel("Position")
    axs[j, 0].legend()

    # 速度（Velocity）
    axs[j, 1].plot(t, jnt_vel_list[:, j], 'o', label='Original Velocity', markersize=4)
    axs[j, 1].plot(t_fine, dq_dt[:, j], label='B-Spline Velocity', linewidth=2)
    axs[j, 1].set_title(f"Velocity $\\dot{{q}}_{j}(t)$")
    axs[j, 1].set_ylabel("Velocity")
    axs[j, 1].legend()

    # 加速度（Acceleration）
    axs[j, 2].plot(t, jnt_acc_list[:, j], 'o', label='Original Acceleration', markersize=4)
    axs[j, 2].plot(t_fine, d2q_dt2[:, j], label='B-Spline Acceleration', linewidth=2)
    axs[j, 2].set_title(f"Acceleration $\\ddot{{q}}_{j}(t)$")
    axs[j, 2].set_ylabel("Acceleration")
    axs[j, 2].legend()

    # 抖动（Jerk）
    dddqaxis = np.diff(jnt_acc_list[:, j], axis=0, prepend=jnt_acc_list[0, j]) / dt
    dddqaxis[0] = 0.0
    dddqaxis[-1] = 0.0

    axs[j, 3].plot(t[1:-1], dddqaxis[1:-1], 'o', label='Original Jerk', markersize=4)
    axs[j, 3].plot(t_fine, d3q_dt3[:, j], label='B-Spline Jerk', linewidth=2)
    axs[j, 3].set_title(f"Jerk $\\dddot{{q}}_{j}(t)$")
    axs[j, 3].set_ylabel("Jerk")
    axs[j, 3].legend()

plt.tight_layout()
plt.show()