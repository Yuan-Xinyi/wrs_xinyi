import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import zarr

# 使用 Zarr 读取数据
root = zarr.open('/home/lqin/zarr_datasets/franka_ruckig_100hz.zarr', mode='r')
traj_id = 19
traj_start = int(np.sum(root['meta']['episode_ends'][:traj_id]))
traj_end = int(np.sum(root['meta']['episode_ends'][:traj_id + 1]))
jnt_pos_list = root['data']['jnt_pos'][traj_start:traj_end]

# 模拟时间序列和关节位置轨迹
T = len(jnt_pos_list)
dt = 0.01
t = np.linspace(0, (T - 1) * dt, T)
s = np.linspace(0, 1, T)  # 参数化变量 s
num_joints = jnt_pos_list.shape[1]  # 多个关节
degree = 8  # 多项式阶数

def constrained_polynomial_fit(s, y, degree):
    """
    基于 s 参数化 0-1 的多项式拟合，带约束：位置、速度、加速度和抖动在起始和末尾为 0。
    """
    p_init = np.polyfit(s, y, degree)
    
    def objective(p):
        poly = np.poly1d(p)
        residuals = np.sum((poly(s) - y) ** 2)
        return residuals

    # 约束：起始和终止的速度、加速度、抖动和位置
    def position_constraint(p):
        poly = np.poly1d(p)
        return [poly(0) - y[0], poly(1) - y[-1]]

    def velocity_constraint(p):
        poly = np.poly1d(p)
        return [np.polyder(poly, 1)(0), np.polyder(poly, 1)(1)]

    def acceleration_constraint(p):
        poly = np.poly1d(p)
        return [np.polyder(poly, 2)(0), np.polyder(poly, 2)(1)]

    def jerk_constraint(p):
        poly = np.poly1d(p)
        return [np.polyder(poly, 3)(0), np.polyder(poly, 3)(1)]

    constraints = [
        {"type": "eq", "fun": position_constraint},
        {"type": "eq", "fun": velocity_constraint},
        {"type": "eq", "fun": acceleration_constraint},
        {"type": "eq", "fun": jerk_constraint}
    ]

    result = minimize(objective, p_init, constraints=constraints, method='SLSQP')
    return np.poly1d(result.x)

# 绘制：每个关节单独绘制
fig, axs = plt.subplots(num_joints, 4, figsize=(16, 4 * num_joints), sharex=True)

for j in range(num_joints):
    # 数据
    y = jnt_pos_list[:, j]

    # 在 s 上拟合多项式
    poly_s = constrained_polynomial_fit(s, y, degree)
    poly_s = np.poly1d(np.poly1d(poly_s).coefficients)
    
    # 生成新的时间序列（例如，时间加倍）
    T_new = 0.5 * (t[-1] - t[0])
    t_new = np.linspace(0, T_new, int(T_new / dt))
    s_new = np.linspace(0, 1, len(t_new))  # 新的 s

    # 根据 s_new 计算新的曲线
    y_fit = poly_s(s_new)
    dy_ds = np.polyder(poly_s, 1)(s_new)
    d2y_ds2 = np.polyder(poly_s, 2)(s_new)
    d3y_ds3 = np.polyder(poly_s, 3)(s_new)

    # 映射到时间
    dy_dt = dy_ds / T_new
    d2y_dt2 = d2y_ds2 / (T_new ** 2)
    d3q_dt3 = d3y_ds3 / (T_new ** 3)
    
    # 绘制原始数据 vs 参数化拟合
    axs[j, 0].plot(t, y, 'o', label='Original Position', markersize=4, color='gray')
    axs[j, 0].plot(t_new, y_fit, label='Parametric Fit', linewidth=2, color='red')
    axs[j, 0].set_title(f"Position $q_{j}(t)$")
    axs[j, 0].set_ylabel("Position")
    axs[j, 0].legend()

    # 速度（Velocity）
    axs[j, 1].plot(t, np.gradient(y, dt), 'o', label='Original Velocity', markersize=4, color='gray')
    axs[j, 1].plot(t_new, dy_dt, label='Parametric Velocity', linewidth=2, color='red')
    axs[j, 1].set_title(f"Velocity $\\dot{{q}}_{j}(t)$")
    axs[j, 1].set_ylabel("Velocity")
    axs[j, 1].legend()

    # 加速度（Acceleration）
    axs[j, 2].plot(t, np.gradient(np.gradient(y, dt), dt), 'o', label='Original Acceleration', markersize=4, color='gray')
    axs[j, 2].plot(t_new, d2y_dt2, label='Parametric Acceleration', linewidth=2, color='red')
    axs[j, 2].set_title(f"Acceleration $\\ddot{{q}}_{j}(t)$")
    axs[j, 2].set_ylabel("Acceleration")
    axs[j, 2].legend()

    # 抖动（Jerk）
    jerk_original = np.gradient(np.gradient(np.gradient(y, dt), dt), dt)
    axs[j, 3].plot(t, jerk_original, 'o', label='Original Jerk', markersize=4, color='gray')
    axs[j, 3].plot(t_new, d3q_dt3, label='Parametric Jerk', linewidth=2, color='red')
    axs[j, 3].set_title(f"Jerk $\\dddot{{q}}_{j}(t)$")
    axs[j, 3].set_ylabel("Jerk")
    axs[j, 3].legend()

plt.tight_layout()
plt.show()
