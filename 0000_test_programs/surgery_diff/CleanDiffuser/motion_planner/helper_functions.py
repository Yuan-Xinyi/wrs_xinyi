import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline

from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from ruckig import InputParameter, OutputParameter, Result, Ruckig

'''B-Spline related functions'''
def Build_BSpline(jnt_pos_list, num_ctrl_pts=64, degree=4):
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


def calculate_BSpline_wrt_T(spline, T_total_new):
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


def plot_BSpline_wrt_org(jnt_pos_list, jnt_vel_list, jnt_acc_list, t, results, overlay=True):
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
            axs[j, 2].plot(t, jnt_acc_list[:, j], 'o', label='Original Acceleration', markersize=4, color='gray',
                           alpha=0.5)
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


def plot_BSpline_wrt_T(results, num_joints, overlay=True):
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

def Time2BSpline(spline, T_total_new):
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

def initialize_ruckig(sampling_interval, waypoint_num=10):
    '''init the robot and world'''
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = franka.FrankaResearch3(enable_cc=True)
    inp = InputParameter(robot.n_dof)
    out = OutputParameter(robot.n_dof, waypoint_num)
    otg = Ruckig(robot.n_dof, sampling_interval, waypoint_num)

    inp.target_velocity = rm.np.zeros(robot.n_dof)
    inp.target_acceleration = rm.np.zeros(robot.n_dof)
    inp.min_position = robot.jnt_ranges[:, 0]
    inp.max_position = robot.jnt_ranges[:, 1]
    inp.max_velocity = rm.np.asarray([rm.pi * 2 / 3] * robot.n_dof)
    inp.max_acceleration = rm.np.asarray([rm.pi] * robot.n_dof)
    inp.max_jerk = rm.np.asarray([rm.pi * 2] * robot.n_dof)

    return base, robot, otg, inp, out

def visualize_anime_path(base, robot, path, start_conf=None, goal_conf=None, f=0.01):
    class Data(object):
        def __init__(self):
            self.counter = 0
            self.path = None
            self.current_model = None

    if start_conf is not None and goal_conf is not None:
        robot.goto_given_conf(jnt_values=start_conf)
        robot.gen_meshmodel(rgb=[0, 0, 1], alpha=.2).attach_to(base)
        robot.goto_given_conf(jnt_values=goal_conf)
        robot.gen_meshmodel(rgb=[0, 1, 0], alpha=.2).attach_to(base)

    anime_data = Data()
    anime_data.path = path

    def update(robot, anime_data, task):
        if anime_data.counter >= len(anime_data.path):
            if anime_data.current_model:
                anime_data.current_model.detach()
            anime_data.counter = 0
            return task.done

        if anime_data.current_model:
            anime_data.current_model.detach()

        conf = anime_data.path[anime_data.counter]
        robot.goto_given_conf(conf)
        anime_data.current_model = robot.gen_meshmodel(alpha=0.7)
        anime_data.current_model.attach_to(base)

        anime_data.counter += 1
        task.delayTime = f  # ✅ 设置下一帧的延迟
        return task.again

    # ✅ 第一次执行在 5 秒后启动
    taskMgr.doMethodLater(5.0, update, "update",
                          extraArgs=[robot, anime_data],
                          appendTask=True)

    base.run()



def workspace_plot(robot, jnt_path):
    # 计算关节路径对应的末端执行器的位置
    pos_list = []
    for jnt_conf in jnt_path:
        pos, _ = robot.fk(jnt_conf)  # 获取末端执行器的位置 pos
        pos_list.append(pos)

    # 转换为 numpy 数组
    pos_list = np.array(pos_list)

    # 创建一个 8x8 的图形
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # 绘制 X, Y, Z 轴的轨迹
    axs[0].plot(pos_list[:, 0], label='X', color='r')
    axs[0].set_title('X Axis')
    axs[0].set_ylabel('Position (X)')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(pos_list[:, 1], label='Y', color='g')
    axs[1].set_title('Y Axis')
    axs[1].set_ylabel('Position (Y)')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(pos_list[:, 2], label='Z', color='b')
    axs[2].set_title('Z Axis')
    axs[2].set_ylabel('Position (Z)')
    axs[2].set_xlabel('Time Steps')
    axs[2].grid(True)
    axs[2].legend()

    # 设置整体图的标题
    fig.suptitle("Workspace Trajectory - X, Y, Z Axes", fontsize=16)

    # 自动调整布局，防止标签重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 显示图形
    plt.show()
