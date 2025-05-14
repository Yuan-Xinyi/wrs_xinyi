import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import zarr
from scipy.optimize import minimize

# 使用 Zarr 读取数据
root = zarr.open('/home/lqin/zarr_datasets/franka_ruckig_100hz.zarr', mode='r')
traj_id = 0
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

# 设置多项式阶数
degree = 8  # 提高阶数以满足额外的约束

# 绘制：每个关节单独绘制
fig, axs = plt.subplots(num_joints, 4, figsize=(16, 4 * num_joints), sharex=True)

def constrained_polynomial_fit(t, y, degree):
    """
    带有约束的多项式拟合：限制位置、速度、加速度和抖动在起始和末尾为0。
    """
    # 初始猜测：普通多项式拟合
    p_init = np.polyfit(t, y, degree)
    
    # 优化目标：最小化拟合误差
    def objective(p):
        poly = np.poly1d(p)
        residuals = np.sum((poly(t) - y) ** 2)
        return residuals

    # 约束：起始和终止的速度、加速度、抖动和位置
    def position_constraint(p):
        poly = np.poly1d(p)
        return [
            poly(t[0]) - y[0],  # 起始位置
            poly(t[-1]) - y[-1]  # 终止位置
        ]
    
    def velocity_constraint(p):
        poly = np.poly1d(p)
        return [
            np.polyder(poly, 1)(t[0]),  # 起始速度
            np.polyder(poly, 1)(t[-1])  # 终止速度
        ]
    
    def acceleration_constraint(p):
        poly = np.poly1d(p)
        return [
            np.polyder(poly, 2)(t[0]),  # 起始加速度
            np.polyder(poly, 2)(t[-1])  # 终止加速度
        ]
    
    def jerk_constraint(p):
        poly = np.poly1d(p)
        return [
            np.polyder(poly, 3)(t[0]),  # 起始抖动（Jerk）
            np.polyder(poly, 3)(t[-1])  # 终止抖动（Jerk）
        ]
    
    # 优化：加入约束
    constraints = [
        {"type": "eq", "fun": position_constraint},
        {"type": "eq", "fun": velocity_constraint},
        {"type": "eq", "fun": acceleration_constraint},
        {"type": "eq", "fun": jerk_constraint}
    ]

    result = minimize(objective, p_init, constraints=constraints, method='SLSQP')
    return np.poly1d(result.x)

for j in range(num_joints):
    # 数据
    y = jnt_pos_list[:, j]
    
    # 使用带约束的多项式拟合
    poly = constrained_polynomial_fit(t, y, degree)
    y_fit = poly(t)
    
    # 计算导数（速度，加速度，抖动）
    dy_dt = np.polyder(poly, 1)(t)
    d2y_dt2 = np.polyder(poly, 2)(t)
    d3y_dt3 = np.polyder(poly, 3)(t)
    
    # 绘制原始数据
    axs[j, 0].plot(t, jnt_pos_list[:, j], 'o', label='Original Position', markersize=4)
    axs[j, 0].plot(t, y_fit, label='Polynomial Fit (Constrained)', linewidth=2)
    axs[j, 0].set_title(f"Position $q_{j}(t)$")
    axs[j, 0].set_ylabel("Position")
    axs[j, 0].legend()

    # 速度（Velocity）
    axs[j, 1].plot(t, jnt_vel_list[:, j], 'o', label='Original Velocity', markersize=4)
    axs[j, 1].plot(t, dy_dt, label='Constrained Velocity', linewidth=2)
    axs[j, 1].set_title(f"Velocity $\\dot{{q}}_{j}(t)$")
    axs[j, 1].set_ylabel("Velocity")
    axs[j, 1].legend()

    # 加速度（Acceleration）
    axs[j, 2].plot(t, jnt_acc_list[:, j], 'o', label='Original Acceleration', markersize=4)
    axs[j, 2].plot(t, d2y_dt2, label='Constrained Acceleration', linewidth=2)
    axs[j, 2].set_title(f"Acceleration $\\ddot{{q}}_{j}(t)$")
    axs[j, 2].set_ylabel("Acceleration")
    axs[j, 2].legend()

    # 抖动（Jerk）
    dddqaxis = np.diff(jnt_acc_list[:, j], axis=0, prepend=jnt_acc_list[0, j]) / dt
    axs[j, 3].plot(t[:-1], dddqaxis[:-1], 'o', label='Original Jerk', markersize=4)
    axs[j, 3].plot(t, d3y_dt3, label='Constrained Jerk', linewidth=2)
    axs[j, 3].set_title(f"Jerk $\\dddot{{q}}_{j}(t)$")
    axs[j, 3].set_ylabel("Jerk")
    axs[j, 3].legend()

plt.tight_layout()
plt.show()
