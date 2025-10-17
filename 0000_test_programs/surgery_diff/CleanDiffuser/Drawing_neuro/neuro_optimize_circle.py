'''wrs reliance'''
from wrs import rm, mcm
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as robot
import wrs.modeling.geometric_model as mgm

'''self modules'''
import rotation_cone_constraint as cone_constraint

'''global variables'''
import time
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


'''initialize robot and scene'''
xarm_gpu = xarm6_gpu.XArmLite6GPU()
xarm = 
jnt = xarm.robot.rand_conf()
xarm.robot.goto_given_conf(jnt_values=jnt)
xarm.robot.gen_stickmodel().attach_to(xarm.base)

'''table'''
table_size = np.array([1.5, 1.5, 0.05]) 
table_pos  = np.array([0.6, 0, -0.025])

table = mcm.gen_box(xyz_lengths=table_size,
                    pos=table_pos,
                    rgb=np.array([0.6, 0.4, 0.2]),
                    alpha=1)
table.attach_to(xarm.base)

'''paper'''
paper_size = np.array([1.0, 1.0, 0.002])
paper_pos  = table_pos.copy()
paper_pos[0] = paper_size[0] / 2.0
paper_pos[1] = 0.0
paper_pos[2] = table_pos[2] + table_size[2]/2 + paper_size[2]/2 
paper = mcm.gen_box(xyz_lengths=paper_size,
                    pos=paper_pos,
                    rgb=np.array([1, 1, 1]),
                    alpha=1)
paper.attach_to(xarm.base)


def randomize_circle(paper_pos, paper_size, num_points=50, margin=0.02, visualize=False):
    """randomly generate a circle on the paper surface"""
    cx, cy, cz = paper_pos
    lx, ly, lz = paper_size
    z = cz + lz / 2
    x_min, x_max = cx - lx/2 + margin, cx + lx/2 - margin
    y_min, y_max = cy - ly/2 + margin, cy + ly/2 - margin

    # Randomly generate circle center and maximum radius constraints
    center = np.array([np.random.uniform(x_min, x_max),
                       np.random.uniform(y_min, y_max),
                       z])
    max_r = min(center[0] - x_min, x_max - center[0],
                center[1] - y_min, y_max - center[1])
    r = np.random.uniform(0.05, max_r)
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    pos_list = np.stack([center[0] + r*np.cos(theta),
                         center[1] + r*np.sin(theta),
                         np.full(num_points, z)], axis=1)

    if visualize:
        [mgm.gen_sphere(p, 0.003, [1,0,0]).attach_to(base) for p in pos_list]
        mgm.gen_sphere(center, 0.005, [0,1,0]).attach_to(base)
        for i in range(num_points):
            mgm.gen_stick(pos_list[i], pos_list[(i+1)%num_points], 0.0015, [0,0,1]).attach_to(base)

    return pos_list, {'center': center, 'radius': r, 'normal': [0,0,1], 'plane_z': z}



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

# GPMP轨迹优化（平滑版）
def gpmp_smooth_trajectory(jnt_path, lengthscale=0.1, variance=1.0, noise=1e-6):
    
    def rbf_kernel(t1, t2, lengthscale=1.0, variance=1.0):
        sqdist = np.subtract.outer(t1, t2) ** 2
        return variance * np.exp(-0.5 * sqdist / lengthscale ** 2)
    
    num_joints = jnt_path.shape[1]
    time_steps = np.linspace(0, 1, len(jnt_path))
    K = rbf_kernel(time_steps, time_steps, lengthscale, variance)
    K_inv = np.linalg.inv(K + noise * np.eye(len(jnt_path)))
    q_smooth = []
    for j in range(num_joints):
        y = jnt_path[:, j]
        alpha = K_inv @ y
        mean = K @ alpha
        q_smooth.append(mean)
    return np.stack(q_smooth, axis=1)



if __name__ == "__main__":
    num_joints = robot.n_dof
    num_points = 100

    # === 生成圆轨迹 ===
    paper_surface_z = paper_pos[2] + paper_size[2]/2
    circle_center = np.array([0.35, 0.0, paper_surface_z + 0.01])
    mgm.gen_sphere(radius=0.005, pos=circle_center, rgb=[0,1,0], alpha=1).attach_to(base)
    # visualize_rotation_cone(base, apex=circle_center,
    #                         a_axis=np.array([0,0,-1]),   # 垂直于xy平面
    #                         alpha_max_rad=np.deg2rad(30),
    #                         height=0.15)
    # base.run()
    raw_jnt_path, pos_list = generate_circle_path(radius=0.23,
                                                  robot=robot,
                                                  table=table,
                                                  paper=paper,
                                                  num_points=num_points,
                                                  center=circle_center)
    for pos in pos_list:
        sphere = mgm.gen_sphere(radius=0.005, pos=pos, rgb=[1,0,0], alpha=1)
        sphere.attach_to(base)
    
    import helper_functions as helper
    import utils
    jnt_path = gpmp_smooth_trajectory(np.array(raw_jnt_path), lengthscale=0.2, variance=1.0, noise=1e-6)
    utils.plot_joint_and_workspace(robot, raw_jnt_path, jnt_path, labels=["Raw Path", "Smoothed Path"])
    helper.visualize_anime_path(base, robot, jnt_path)
