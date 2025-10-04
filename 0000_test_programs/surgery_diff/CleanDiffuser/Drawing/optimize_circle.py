import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from wrs import wd, rm, mcm
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
# import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm6
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6
from wrs import wd, rm
import wrs.modeling.geometric_model as mgm
import rotation_cone_constraint as cone_constraint

# 初始化机器人和场景
# robot = franka.FrankaResearch3(enable_cc=True)
robot = xarm6.XArmLite6Miller(enable_cc=True) 
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)
# robot.goto_home_conf()
# robot.gen_meshmodel().attach_to(base)
# print(robot.fk(jnt_values=robot.get_jnt_values())[0])
# base.run()

# 桌面参数
table_size = np.array([1.5, 1.5, 0.05])   # 桌子大小 (长x宽x厚度)
table_pos  = np.array([0.6, 0, -0.025])   # 放在机器人前面

table = mcm.gen_box(xyz_lengths=table_size,
                    pos=table_pos,
                    rgb=np.array([0.6, 0.4, 0.2]),   # 棕色
                    alpha=1)
table.attach_to(base)

# 画纸参数
paper_size = np.array([1.0, 1.0, 0.002])  # 画纸大小
paper_pos  = table_pos.copy()

# 画纸中心位置：x方向 = 一半宽度，这样左边缘刚好在 x=0
paper_pos[0] = paper_size[0] / 2.0
paper_pos[1] = 0.0
paper_pos[2] = table_pos[2] + table_size[2]/2 + paper_size[2]/2  # 放在桌面上

paper = mcm.gen_box(xyz_lengths=paper_size,
                    pos=paper_pos,
                    rgb=np.array([1, 1, 1]),         # 白色
                    alpha=1)
paper.attach_to(base)

'''visualize'''
# cc_jnt = [-3.1149012366937407, -1.4309495062531088, 4.284456928656966, 
#           0.27762536280644234, 0.4952422567010116, -2.1941828629110525]
# robot.goto_given_conf(cc_jnt)
# pos,rot = robot.fk(jnt_values=cc_jnt)
# mgm.gen_frame(pos=pos, rotmat=rot).attach_to(base)
# print(robot.is_collided(obstacle_list=[table, paper], toggle_contacts=True, toggle_dbg=False)[0])
# robot.gen_meshmodel(toggle_cdprim=True).attach_to(base)
# # robot.show_cdprim()
# base.run()

import time
fk_call_count = 0
fk_total_time = 0.0

def generate_circle_path(
        robot,
        table, paper,
        radius=0.1,
        num_points=50,
        center=None,
        max_attempts=100,
        alpha_max_rad=np.deg2rad(30),
        max_rot_diff=np.deg2rad(30),
        n_alpha=3,
        n_psi=12,
        visualize=False):
    """
    生成松弛旋转约束的圆轨迹
    - 起始点 cone 采样 (基于平面法向/工具轴)
    - 后续点 基于 R_prev 在邻域采样
    """
    thetas = np.linspace(0, 2*np.pi, num_points, endpoint=False)

    for _ in range(max_attempts):
        pos_list = []
        for theta in thetas:
            pos = circle_center.copy()
            pos[0] += radius * np.cos(theta)
            pos[1] += radius * np.sin(theta)
            pos_list.append(pos)
            plane_normal = np.array([0, 0, -1])

        # 6. 使用 relaxed 逆解生成路径
        jnt_list = cone_constraint.gen_jnt_list_from_pos_list_relaxed(
            init_jnt=None,
            pos_list=pos_list,
            robot=robot,
            obstacle_list=[],  # [table, paper]
            base=None,  # 不可视化时可设 None
            alpha_max_rad=alpha_max_rad,
            n_alpha=n_alpha,
            n_psi=n_psi,
            max_rot_diff=max_rot_diff,
            check_collision=True,
            visualize=visualize,
            plane_normal=plane_normal
        )
        print('Generated joint path length:', len(jnt_list))
        for jnt in jnt_list:
            robot.goto_given_conf(jnt)
            if not robot.is_collided(obstacle_list=[table, paper], toggle_contacts=False, toggle_dbg=False):
                robot.gen_meshmodel(rgb=[0,0,1], alpha=0.3).attach_to(base)
            else:
                robot.gen_meshmodel(rgb=[1,0,0], alpha=0.3).attach_to(base)
        base.run()

    print("Failed to generate circle after attempts.")
    return None, None



def calculate_rot_error(rot1, rot2):
    delta = rm.delta_w_between_rotmat(rot1, rot2)
    return np.linalg.norm(delta)

# 代价函数：末端误差 + 平滑项
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
    n_seqs = len(joint_seqs)
    T = joint_seqs[0].shape[0]
    if labels is None:
        labels = [f"Traj {i+1}" for i in range(n_seqs)]
    time = np.arange(T)
    colors = plt.cm.tab10.colors

    fig, axs = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    fig.suptitle("Multi-Trajectory Joint Comparison", fontsize=16)

    for j in range(7):
        for i, seq in enumerate(joint_seqs):
            axs[j].plot(time, seq[:, j], label=labels[i],
                        color=colors[i % len(colors)], linestyle='-' if i == 0 else '--')
        axs[j].set_ylabel(f"Joint {j+1}")
        axs[j].grid(True)
        if j == 0:
            axs[j].legend(loc="upper right")

    axs[-1].set_xlabel("Time Step")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def workspace_plot_multi(robot, *jnt_paths, labels=None):
    T = jnt_paths[0].shape[0]
    if labels is None:
        labels = [f"Traj {i+1}" for i in range(len(jnt_paths))]
    colors = plt.cm.tab10.colors
    pos_lists = []
    for path in jnt_paths:
        pos_list = [robot.fk(jnt)[0] for jnt in path]
        pos_lists.append(np.array(pos_list))

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes_labels = ['X', 'Y', 'Z']
    for axis in range(3):
        for i, pos_arr in enumerate(pos_lists):
            axs[axis].plot(pos_arr[:, axis],
                           label=labels[i],
                           color=colors[i % len(colors)],
                           linestyle='-' if i == 0 else '--')
        axs[axis].set_ylabel(f'{axes_labels[axis]} Axis')
        axs[axis].grid(True)
        if axis == 0:
            axs[axis].legend()

    axs[-1].set_xlabel("Time Steps")
    fig.suptitle("Workspace Trajectories - End Effector Positions (X/Y/Z)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_rotation_cone(base, apex, a_axis, alpha_max_rad,
                            height=0.2, n_alpha=3, n_psi=12,
                            line_radius=0.002):
    """
    可视化旋转锥和旋转候选姿态
    :param base: PandaWorld/Scene
    :param apex: 锥顶 (一般是末端执行器位置)
    :param a_axis: 锥体主轴方向 (例如纸面法向)
    :param alpha_max_rad: 锥体半角 (rad)
    :param height: 锥体高度
    :param n_alpha: alpha 层数 (决定cone里候选点的层数)
    :param n_psi: 每层的采样数
    """
    a_axis = a_axis / (np.linalg.norm(a_axis)+1e-12)
    center = apex + a_axis * height

    # 局部基向量
    tmp = np.array([1,0,0]) if abs(a_axis[0]) < 0.9 else np.array([0,1,0])
    x_dir = np.cross(a_axis, tmp); x_dir /= np.linalg.norm(x_dir)
    y_dir = np.cross(a_axis, x_dir); y_dir /= np.linalg.norm(y_dir)

    # cone 底面
    r = np.tan(alpha_max_rad) * height
    angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
    circle_pts = []
    for theta in angles:
        p = center + r*(np.cos(theta)*x_dir + np.sin(theta)*y_dir)
        circle_pts.append(p)
        mgm.gen_sphere(pos=p, radius=0.005, rgb=[1,0,0]).attach_to(base)

    # 底面边
    for i in range(len(circle_pts)):
        p1, p2 = circle_pts[i], circle_pts[(i+1)%len(circle_pts)]
        mgm.gen_stick(spos=p1, epos=p2, radius=line_radius, rgb=[0,0,1]).attach_to(base)

    # 锥体侧边
    for p in circle_pts:
        mgm.gen_stick(spos=apex, epos=p, radius=line_radius, rgb=[0,1,0]).attach_to(base)

    # ===== 候选旋转矩阵 (来自旋转锥) =====
    R_list = cone_constraint.sample_rotations_in_cone(a_axis=a_axis,
                                       alpha_max_rad=alpha_max_rad,
                                       n_alpha=n_alpha,
                                       n_psi=n_psi)

    for R in R_list:
        # 在 apex 位置画候选姿态坐标系
        mgm.gen_frame(pos=apex, rotmat=R).attach_to(base)
        jnt = robot.ik(tgt_pos=apex, tgt_rotmat=R)
        if jnt is not None:
            robot.goto_given_conf(jnt)
            robot.gen_meshmodel(rgb=[1,0,1], alpha=0.3).attach_to(base)

    return R_list





if __name__ == "__main__":
    num_joints = robot.n_dof
    num_points = 15

    # === 生成圆轨迹 ===
    paper_surface_z = paper_pos[2] + paper_size[2]/2
    circle_center = np.array([0.2, 0.0, paper_surface_z + 0.01])
    mgm.gen_sphere(radius=0.005, pos=circle_center, rgb=[0,1,0], alpha=1).attach_to(base)
    # visualize_rotation_cone(base, apex=circle_center,
    #                         a_axis=np.array([0,0,-1]),   # 垂直于xy平面
    #                         alpha_max_rad=np.deg2rad(30),
    #                         height=0.15)
    # base.run()
    gth_jnt_path, pos_list = generate_circle_path(radius=0.1,
                                                  robot=robot,
                                                  table=table,
                                                  paper=paper,
                                                  num_points=num_points,
                                                  center=circle_center)
    for pos in pos_list:
        sphere = mgm.gen_sphere(radius=0.005, pos=pos, rgb=[1,0,0], alpha=1)
        sphere.attach_to(base)
    
    import helper_functions as helper
    helper.visualize_anime_path(base, robot, gth_jnt_path)

    # 初始猜测
    q_init = np.zeros((num_points, num_joints))

    # 关节约束
    q_min = robot.jnt_ranges[:, 0]
    q_max = robot.jnt_ranges[:, 1]
    bounds = [(q_min[i % num_joints], q_max[i % num_joints]) for i in range(num_points * num_joints)]

    # 优化
    start_time = time.time()
    res = minimize(
        cost_fn,
        q_init.flatten(),
        args=(pos_list, num_joints),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True, 'maxiter': 500, 'gtol': 1e-2}
    )
    end_time = time.time()
    print(f"Optimization took {end_time - start_time:.2f} seconds")

    q_traj = res.x.reshape(num_points, num_joints)
    similarity = np.mean(np.linalg.norm(q_traj - gth_jnt_path, axis=1))
    print(f"Optimization completed with average l2 norm: {similarity:.4f}")

    # 可视化对比
    traj_comparison_multi(np.array(gth_jnt_path), q_traj, q_init, labels=["Ground Truth Circle", "Optimized", "Init Guess"])
    workspace_plot_multi(robot, np.array(gth_jnt_path), q_traj, labels=["Ground Truth Circle", "Optimized"])

    # 性能分析
    print("===== Timing Analysis =====")
    print(f"Total FK calls: {fk_call_count}")
    print(f"Total FK time: {fk_total_time:.4f} sec")
    print(f"Average FK time per call: {fk_total_time / fk_call_count:.8f} sec")
    print("===========================")

    # 场景里显示起点和终点
    robot.goto_given_conf(gth_jnt_path[0])
    robot.gen_meshmodel(rgb=[0,1,0], alpha=0.5).attach_to(base)
    robot.goto_given_conf(gth_jnt_path[-1])
    robot.gen_meshmodel(rgb=[0,1,0], alpha=0.5).attach_to(base)

    robot.goto_given_conf(q_traj[0])
    robot.gen_meshmodel(rgb=[0,0,1], alpha=0.5).attach_to(base)
    robot.goto_given_conf(q_traj[-1])
    robot.gen_meshmodel(rgb=[0,0,1], alpha=0.5).attach_to(base)

    base.run()
