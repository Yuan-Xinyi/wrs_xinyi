# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.interpolate import splprep, splev
# # from scipy.interpolate import UnivariateSpline


# # import zarr
# # import numpy as np
# # import matplotlib.pyplot as plt

# # import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
# # from wrs import wd, rm, mcm
# # import wrs.modeling.geometric_model as mgm
# # robot_s = franka.FrankaResearch3(enable_cc=True)
# # base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
# # mgm.gen_frame().attach_to(base)

# # '''splprep code test'''
# # # # 原始数据点
# # # x = np.linspace(0, 6, 7)
# # # y = np.sin(x)

# # # # 拟合三次样条曲线
# # # tck, u = splprep([x, y], s=0, k=3)
# # # t, c, k = tck

# # # # 使用splev插值生成曲线点
# # # u_fine = np.linspace(0, 1, 200)
# # # x_smooth, y_smooth = splev(u_fine, tck)

# # # # 控制点
# # # ctrl_x, ctrl_y = c[0], c[1]

# # # # 计算样条在每个节点t位置对应的点（注意t中有重复的节点，用unique处理避免重复采样）
# # # t_unique = np.unique(t)
# # # x_knots, y_knots = splev(np.clip((t_unique - t.min()) / (t.max() - t.min()), 0, 1), tck)

# # # # 绘图
# # # plt.figure(figsize=(10, 6))
# # # plt.plot(x, y, 'o', label='Original Data Points', markersize=8)
# # # plt.plot(x_smooth, y_smooth, 'r-', label='B-Spline Curve', linewidth=2)
# # # plt.plot(ctrl_x, ctrl_y, 'x--', label='Control Points', color='purple', markersize=8)
# # # plt.plot(x_knots, y_knots, 's', label='Knot Points (t)', color='green', markersize=7)
# # # for i, (xk, yk) in enumerate(zip(x_knots, y_knots)):
# # #     plt.text(xk + 0.1, yk, f't[{i}]', fontsize=9, color='green')
# # # plt.title("B-Spline Curve with Control Points and Knots")
# # # plt.xlabel("x")
# # # plt.ylabel("y")
# # # plt.legend()
# # # plt.grid(True)
# # # plt.axis("equal")
# # # plt.savefig("b_spline_curve.png", dpi=600, bbox_inches='tight')
# # # # plt.tight_layout()
# # # plt.show()

# # '''3rd order spline code test'''
# # root = zarr.open('/home/lqin/zarr_datasets/franka_ruckig_100hz_fixgoal.zarr', mode='r')
# # # root = zarr.open('/home/lqin/zarr_datasets/franka_ruckig.zarr', mode='r')

# # traj_id =100
# # traj_start = int(np.sum(root['meta']['episode_ends'][:traj_id]))
# # traj_end = int(np.sum(root['meta']['episode_ends'][:traj_id + 1]))
# # jnt_pos_list = root['data']['jnt_pos'][traj_start:traj_end]
# # jnt_vel_list = root['data']['jnt_vel'][traj_start:traj_end]
# # jnt_acc_list = root['data']['jnt_acc'][traj_start:traj_end]
# # goal_conf = root['data']['goal_conf'][traj_start:traj_end]

# # T = len(jnt_pos_list)
# # t = np.linspace(0, 0.01 * (T - 1), T)  # 真实时间序列


# # # B-spline 拟合
# # tck, u = splprep([t, jnt_pos_list[:,0]], s=0, k=5)  # 3次样条曲线

# # # 使用更密集的点进行评估
# # # u_fine = np.linspace(0, 1, 1000)
# # # t_fine, pos = splev(u_fine, tck, der=0)
# # # vel = splev(u_fine, tck, der=1)[1]
# # # acc = splev(u_fine, tck, der=2)[1]
# # # jerk = splev(u_fine, tck, der=3)[1]
# # # 使用更密集的点进行评估
# # u_fine = np.linspace(0, 1, 1000)
# # t_fine, pos = splev(u_fine, tck, der=0)
# # vel = splev(u_fine, tck, der=1)[1]
# # acc = splev(u_fine, tck, der=2)[1]
# # jerk = splev(u_fine, tck, der=3)[1]

# # # 时间缩放
# # scale = t[-1] - t[0]
# # vel /= scale
# # acc /= scale**2
# # jerk /= scale**3


# # # 绘图
# # fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)


# # axs[0].set_title("Position")
# # axs[0].set_ylabel("Position [units]")
# # axs[0].plot(t, jnt_pos_list[:,0], 'o', label='Original Data Points', markersize=3)
# # axs[0].plot(t_fine, pos)
# # axs[0].legend()

# # axs[1].plot(t_fine, vel)
# # axs[1].set_title("Velocity")
# # axs[1].set_ylabel("Velocity [units/s]")
# # axs[1].plot(t, jnt_vel_list[:,0], 'o', label='Original Data Points', markersize=3)
# # axs[1].legend()

# # axs[2].plot(t_fine, acc)
# # axs[2].set_title("Acceleration")
# # axs[2].set_ylabel("Acceleration [units/s²]")
# # axs[2].plot(t, jnt_acc_list[:,0], 'o', label='Original Data Points', markersize=3)
# # axs[2].legend()

# # axs[3].plot(t_fine, jerk)
# # axs[3].set_title("Jerk")
# # axs[3].set_ylabel("Jerk [units/s³]")
# # axs[3].set_xlabel("Time [s]")
# # axs[3].plot(t[:-1], jnt_acc_list[1:,0]-jnt_acc_list[:-1,0], 'o', label='Original Data Points', markersize=3)
# # axs[3].legend()

# # plt.tight_layout()
# # plt.show()

'''univariateSpline code test'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import zarr

# 读取 zarr 数据
root = zarr.open('/home/lqin/zarr_datasets/franka_ruckig_100hz_fixgoal.zarr', mode='r')
traj_id = 100
traj_start = int(np.sum(root['meta']['episode_ends'][:traj_id]))
traj_end = int(np.sum(root['meta']['episode_ends'][:traj_id + 1]))

jnt_pos_list = root['data']['jnt_pos'][traj_start:traj_end]
jnt_vel_list = root['data']['jnt_vel'][traj_start:traj_end]
jnt_acc_list = root['data']['jnt_acc'][traj_start:traj_end]

# 生成真实时间序列
T = len(jnt_pos_list)
dt = 0.01  # 时间间隔
t = np.linspace(0, dt * (T - 1), T)

# 选择一个关节维度进行拟合
joint_index = 0
x = jnt_pos_list[:, joint_index]

# # # 用 UnivariateSpline 对 x(t) 拟合
# # spline = UnivariateSpline(t, x, s=1e-4, k=5)  # s 可以调整平滑度，k=5 表示五次 B-spline

# # # 构造高分辨率时间轴并求导
# # t_fine = np.linspace(t[0], t[-1], 1000)
# # pos = spline(t_fine)
# # vel = spline.derivative(1)(t_fine)
# # acc = spline.derivative(2)(t_fine)
# # jerk = spline.derivative(3)(t_fine)

# # # 计算原始数据的 jerk 差分估计
# # jerk_numeric = np.diff(jnt_acc_list[:, joint_index]) / dt
# # t_jerk = t[1:-1]

# # # 绘图对比
# # fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# # axs[0].plot(t, x, 'o', markersize=3, label='Original')
# # axs[0].plot(t_fine, pos, label='Spline Fit')
# # axs[0].set_title("Position")
# # axs[0].set_ylabel("Position")
# # axs[0].legend()

# # axs[1].plot(t, jnt_vel_list[:, joint_index], 'o', markersize=3, label='Original')
# # axs[1].plot(t_fine, vel, label='Spline Derivative')
# # axs[1].set_title("Velocity")
# # axs[1].set_ylabel("Velocity")
# # axs[1].legend()

# # axs[2].plot(t, jnt_acc_list[:, joint_index], 'o', markersize=3, label='Original')
# # axs[2].plot(t_fine, acc, label='Spline Derivative')
# # axs[2].set_title("Acceleration")
# # axs[2].set_ylabel("Acceleration")
# # axs[2].legend()

# # axs[3].plot(t_jerk, jerk_numeric[:-1], 'o', markersize=3, label='Numeric Jerk')
# # axs[3].plot(t_fine, jerk, label='Spline Derivative')
# # axs[3].set_title("Jerk")
# # axs[3].set_ylabel("Jerk")
# # axs[3].set_xlabel("Time [s]")
# # axs[3].legend()

# # plt.tight_layout()
# # plt.show()

# '''test'''
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import UnivariateSpline

# # 假设你已经有如下输入
# # jnt_pos_list = np.array([...])  # shape = (T, D)，D = 7 for 7 DoF
# dt = 0.01
# T, D = jnt_pos_list.shape
# t = np.linspace(0, dt * (T - 1), T)

# # 设置拟合精度和平滑参数
# spline_order = 5
# smoothness = 1e-4

# # 高分辨率时间轴用于可视化
# t_fine = np.linspace(t[0], t[-1], 1000)

# # 初始化结果容器
# pos_all = np.zeros((1000, D))
# vel_all = np.zeros((1000, D))
# acc_all = np.zeros((1000, D))
# jerk_all = np.zeros((1000, D))

# # 对每个关节拟合 UnivariateSpline
# splines = []
# for d in range(D):
#     spline = UnivariateSpline(t, jnt_pos_list[:, d], s=smoothness, k=spline_order)
#     splines.append(spline)

#     pos_all[:, d] = spline(t_fine)
#     vel_all[:, d] = spline.derivative(1)(t_fine)
#     acc_all[:, d] = spline.derivative(2)(t_fine)
#     jerk_all[:, d] = spline.derivative(3)(t_fine)

# # 示例：绘制第一个关节的结果
# joint_index = 0

# fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# axs[0].plot(t, jnt_pos_list[:, joint_index], 'o', markersize=3, label='Original')
# axs[0].plot(t_fine, pos_all[:, joint_index], label='Spline Fit')
# axs[0].set_title("Joint Position")
# axs[0].set_ylabel("Position")
# axs[0].legend()

# axs[1].plot(t_fine, vel_all[:, joint_index], label='Velocity')
# axs[1].set_title("Joint Velocity")
# axs[1].set_ylabel("Velocity")
# axs[1].legend()

# axs[2].plot(t_fine, acc_all[:, joint_index], label='Acceleration')
# axs[2].set_title("Joint Acceleration")
# axs[2].set_ylabel("Acceleration")
# axs[2].legend()

# axs[3].plot(t_fine, jerk_all[:, joint_index], label='Jerk')
# axs[3].set_title("Joint Jerk")
# axs[3].set_ylabel("Jerk")
# axs[3].set_xlabel("Time [s]")
# axs[3].legend()

# plt.tight_layout()
# plt.show()

# === Step 1: 轨迹拟合为 B-spline，获取控制点和节点 ===
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# 使用 splprep 拟合 B-spline 参数化曲线 (t(u), x(u))
tck, u = splprep([t, x], s=0, k=3)
u_fine = np.linspace(0, 1, T)
t_interp, x_interp = splev(u_fine, tck, der=0)

# === Step 2: 用差分方式估计导数 ===
# 中心差分估计速度、加速度、jerk
vel = np.zeros_like(x_interp)
acc = np.zeros_like(x_interp)
jerk = np.zeros_like(x_interp)

for i in range(1, T-1):
    vel[i] = (x_interp[i+1] - x_interp[i-1]) / (2 * dt)
    acc[i] = (x_interp[i+1] - 2*x_interp[i] + x_interp[i-1]) / (dt**2)

for i in range(2, T-2):
    jerk[i] = (x_interp[i+2] - 2*x_interp[i+1] + 2*x_interp[i-1] - x_interp[i-2]) / (2 * dt**3)

# === Step 3: 可视化 ===
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].plot(t_interp, x_interp, label='Position')
axs[0].set_title("Position")
axs[0].set_ylabel("x(t)")
axs[0].legend()

axs[1].plot(t_interp, vel, label='Velocity')
axs[1].set_title("Velocity (Central Difference)")
axs[1].set_ylabel("dx/dt")
axs[1].legend()

axs[2].plot(t_interp, acc, label='Acceleration')
axs[2].set_title("Acceleration (Central Difference)")
axs[2].set_ylabel("d²x/dt²")
axs[2].legend()

axs[3].plot(t_interp, jerk, label='Jerk')
axs[3].set_title("Jerk (Central Difference)")
axs[3].set_ylabel("d³x/dt³")
axs[3].set_xlabel("Time [s]")
axs[3].legend()

plt.tight_layout()
plt.show()

