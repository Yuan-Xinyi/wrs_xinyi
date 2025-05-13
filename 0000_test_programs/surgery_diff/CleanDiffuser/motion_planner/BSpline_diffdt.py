import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline
import zarr

def build_bspline(jnt_pos_list, num_ctrl_pts=64, degree=4):
    """
    构建 B-Spline 并返回 B-Spline 对象及其控制点信息。
    """
    T = len(jnt_pos_list)
    s = np.linspace(0, 1, T)
    
    # 设置 B-Spline 参数
    knots = np.linspace(0, 1, num_ctrl_pts - degree + 1)
    knots = np.concatenate(([0] * degree, knots, [1] * degree))
    
    # 构建 B-Spline
    spline = make_lsq_spline(s, jnt_pos_list, knots, degree)
    return spline, spline.c, spline.t, spline.k

def update_bspline_with_new_time(spline, T_total_new):
    """
    根据新的 T_total 和 dt 更新 B-Spline 曲线及其导数。
    """
    s_fine = np.linspace(0, 1, 1000)
    q_s = spline(s_fine)
    dq_ds = spline.derivative(1)(s_fine)
    d2q_ds2 = spline.derivative(2)(s_fine)
    d3q_ds3 = spline.derivative(3)(s_fine)

    # 时间映射
    r_s = 1 / T_total_new
    q_t = q_s
    dq_dt = dq_ds * r_s
    d2q_dt2 = d2q_ds2 * (r_s ** 2)
    d3q_dt3 = d3q_ds3 * (r_s ** 3)
    t_fine = s_fine * T_total_new

    return t_fine, q_t, dq_dt, d2q_dt2, d3q_dt3

def plot_bspline(jnt_pos_list, jnt_vel_list, jnt_acc_list, t, results, overlay=True):
    """
    绘制 B-Spline 及其导数，比较原始数据和 B-Spline 曲线。
    """
    num_joints = jnt_pos_list.shape[1]

    if overlay:
        fig, axs = plt.subplots(num_joints, 4, figsize=(16, 4 * num_joints), sharex=True)
        colors = ['r', 'g', 'b', 'm', 'c']

        for j in range(num_joints):
            # 绘制原始数据（仅一次）
            axs[j, 0].plot(t, jnt_pos_list[:, j], 'o', label='Original Position', markersize=4, color='gray', alpha=0.5)
            axs[j, 1].plot(t, jnt_vel_list[:, j], 'o', label='Original Velocity', markersize=4, color='gray', alpha=0.5)
            axs[j, 2].plot(t, jnt_acc_list[:, j], 'o', label='Original Acceleration', markersize=4, color='gray', alpha=0.5)
            dddqaxis = np.diff(jnt_acc_list[:, j], axis=0, prepend=jnt_acc_list[0, j]) / (t[1] - t[0])
            axs[j, 3].plot(t[:-1], dddqaxis[:-1], 'o', label='Original Jerk', markersize=4, color='gray', alpha=0.5)

        for idx, (T_total, t_fine, q_t, dq_dt, d2q_dt2, d3q_dt3) in enumerate(results):
            color = colors[idx % len(colors)]
            for j in range(num_joints):
                axs[j, 0].plot(t_fine, q_t[:, j], label=f'B-Spline (T={T_total}s)', color=color, linewidth=2)
                axs[j, 1].plot(t_fine, dq_dt[:, j], label=f'B-Spline Velocity (T={T_total}s)', color=color)
                axs[j, 2].plot(t_fine, d2q_dt2[:, j], label=f'B-Spline Acceleration (T={T_total}s)', color=color)
                axs[j, 3].plot(t_fine, d3q_dt3[:, j], label=f'B-Spline Jerk (T={T_total}s)', color=color)

    else:
        for idx, (T_total, t_fine, q_t, dq_dt, d2q_dt2, d3q_dt3) in enumerate(results):
            fig, axs = plt.subplots(num_joints, 4, figsize=(16, 4 * num_joints), sharex=True)
            for j in range(num_joints):
                # 绘制原始数据
                axs[j, 0].plot(t, jnt_pos_list[:, j], 'o', label='Original Position', markersize=4)
                axs[j, 1].plot(t, jnt_vel_list[:, j], 'o', label='Original Velocity', markersize=4)
                axs[j, 2].plot(t, jnt_acc_list[:, j], 'o', label='Original Acceleration', markersize=4)
                dddqaxis = np.diff(jnt_acc_list[:, j], axis=0, prepend=jnt_acc_list[0, j]) / (t[1] - t[0])
                axs[j, 3].plot(t[:-1], dddqaxis[:-1], 'o', label='Original Jerk', markersize=4)

                # 绘制 B-Spline 数据
                axs[j, 0].plot(t_fine, q_t[:, j], label=f'B-Spline (T={T_total}s)', linewidth=2)
                axs[j, 1].plot(t_fine, dq_dt[:, j], label='B-Spline Velocity')
                axs[j, 2].plot(t_fine, d2q_dt2[:, j], label='B-Spline Acceleration')
                axs[j, 3].plot(t_fine, d3q_dt3[:, j], label='B-Spline Jerk')

    for ax in axs.flat:
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_bspline_only(results, num_joints, overlay=True):
    """
    绘制 B-Spline 及其导数（无原始数据），比较不同 T_total。
    """
    if overlay:
        fig, axs = plt.subplots(num_joints, 4, figsize=(16, 4 * num_joints), sharex=True)
        colors = ['r', 'g', 'b', 'm', 'c']

        for idx, (T_total, t_fine, q_t, dq_dt, d2q_dt2, d3q_dt3) in enumerate(results):
            color = colors[idx % len(colors)]
            for j in range(num_joints):
                axs[j, 0].plot(t_fine, q_t[:, j], label=f'B-Spline (T={T_total}s)', color=color, linewidth=2)
                axs[j, 0].set_title(f"Position $q_{j}(t)$")
                axs[j, 0].set_ylabel("Position")
                axs[j, 0].legend()

                axs[j, 1].plot(t_fine, dq_dt[:, j], label=f'B-Spline Velocity (T={T_total}s)', color=color)
                axs[j, 1].set_title(f"Velocity $\\dot{{q}}_{j}(t)$")
                axs[j, 1].set_ylabel("Velocity")
                axs[j, 1].legend()

                axs[j, 2].plot(t_fine, d2q_dt2[:, j], label=f'B-Spline Acceleration (T={T_total}s)', color=color)
                axs[j, 2].set_title(f"Acceleration $\\ddot{{q}}_{j}(t)$")
                axs[j, 2].set_ylabel("Acceleration")
                axs[j, 2].legend()

                axs[j, 3].plot(t_fine, d3q_dt3[:, j], label=f'B-Spline Jerk (T={T_total}s)', color=color)
                axs[j, 3].set_title(f"Jerk $\\dddot{{q}}_{j}(t)$")
                axs[j, 3].set_ylabel("Jerk")
                axs[j, 3].legend()

    else:
        for idx, (T_total, t_fine, q_t, dq_dt, d2q_dt2, d3q_dt3) in enumerate(results):
            fig, axs = plt.subplots(num_joints, 4, figsize=(16, 4 * num_joints), sharex=True)
            for j in range(num_joints):
                axs[j, 0].plot(t_fine, q_t[:, j], label=f'B-Spline (T={T_total}s)', linewidth=2)
                axs[j, 0].set_title(f"Position $q_{j}(t)$")
                axs[j, 0].set_ylabel("Position")
                axs[j, 0].legend()

                axs[j, 1].plot(t_fine, dq_dt[:, j], label='B-Spline Velocity')
                axs[j, 1].set_title(f"Velocity $\\dot{{q}}_{j}(t)$")
                axs[j, 1].set_ylabel("Velocity")
                axs[j, 1].legend()

                axs[j, 2].plot(t_fine, d2q_dt2[:, j], label='B-Spline Acceleration')
                axs[j, 2].set_title(f"Acceleration $\\ddot{{q}}_{j}(t)$")
                axs[j, 2].set_ylabel("Acceleration")
                axs[j, 2].legend()

                axs[j, 3].plot(t_fine, d3q_dt3[:, j], label='B-Spline Jerk')
                axs[j, 3].set_title(f"Jerk $\\dddot{{q}}_{j}(t)$")
                axs[j, 3].set_ylabel("Jerk")
                axs[j, 3].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    mode = 'bspline_from_control_points'  # 'bspline_from_control_points' or 'bspline_from_traj'
    if mode == 'bspline_from_traj':
        # 测试：加载数据并测试不同的 T_total 和 dt
        root = zarr.open('/home/lqin/zarr_datasets/franka_ruckig_100hz_fixgoal.zarr', mode='r')
        traj_id = 1
        traj_start = int(np.sum(root['meta']['episode_ends'][:traj_id]))
        traj_end = int(np.sum(root['meta']['episode_ends'][:traj_id + 1]))
        jnt_pos_list = root['data']['jnt_pos'][traj_start:traj_end]
        print(f"Trajectory ID: {traj_id}, Start: {repr(jnt_pos_list[0])}, End: {repr(jnt_pos_list[-1])}")
        jnt_vel_list = root['data']['jnt_vel'][traj_start:traj_end]
        jnt_acc_list = root['data']['jnt_acc'][traj_start:traj_end]

        # 原始时间序列
        T = len(jnt_pos_list)
        dt = 0.01  # 原始 dt
        t = np.linspace(0, (T - 1) * dt, T)

        # 构建 B-Spline
        _, org_c, _, _ = build_bspline(jnt_pos_list)
        print(repr(org_c))
        print(f"first cp: {repr(org_c[0])}, last cp: {repr(org_c[-1])}")

        # reconstruct B-Spline
        degree = 4
        num_ctrl_pts = 64
        ctrl_points = np.linspace(0, 1, num_ctrl_pts)
        knots = np.linspace(0, 1, num_ctrl_pts - degree + 1)
        knots = np.concatenate(([0] * degree, knots, [1] * degree))
        spline = BSpline(knots, org_c, degree)

        # 测试不同的 T_total
        T_total_list = [3.5, 5, 8, 10]
        results = []
        for T_total_new in T_total_list:
            print(f"\nTesting with T_total = {T_total_new}s")
            result = (T_total_new, *update_bspline_with_new_time(spline, T_total_new))
            results.append(result)

        # 绘制（叠加或独立绘制）
        # plot_bspline(jnt_pos_list, jnt_vel_list, jnt_acc_list, t, results, overlay=True)
        plot_bspline_only(results, 7, overlay=True)
    elif mode == 'bspline_from_control_points':
        # 测试：加载数据并测试不同的 T_total 和 dt
        root = zarr.open('/home/lqin/zarr_datasets/franka_ruckig_100hz_bspline.zarr', mode='r')
        traj_id = 0
        traj_end = int(root['meta']['episode_ends'][traj_id])
        control_points = root['data']['control_points'][traj_end-64:traj_end]

        # reconstruct B-Spline
        degree = 4
        num_ctrl_pts = 64
        ctrl_points = np.linspace(0, 1, num_ctrl_pts)
        knots = np.linspace(0, 1, num_ctrl_pts - degree + 1)
        knots = np.concatenate(([0] * degree, knots, [1] * degree))
        spline = BSpline(knots, control_points, degree)

        # 测试不同的 T_total
        T_total_list = [3.5, 5, 8, 10]
        results = []
        for T_total_new in T_total_list:
            print(f"\nTesting with T_total = {T_total_new}s")
            result = (T_total_new, *update_bspline_with_new_time(spline, T_total_new))
            results.append(result)

        # 绘制（叠加或独立绘制）
        # plot_bspline(jnt_pos_list, jnt_vel_list, jnt_acc_list, t, results, overlay=True)
        plot_bspline_only(results, 7, overlay=True)
    else:
        raise ValueError("Invalid mode. Choose 'bspline_from_control_points' or 'bspline_from_traj'.")
