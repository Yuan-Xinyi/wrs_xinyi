'''
single plot
'''
# import numpy as np
# from scipy.stats import qmc
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.neighbors import NearestNeighbors

# from wrs import wd, rm, mcm
# import wrs.robot_sim.robots.cobotta.cobotta as cbt

# # 初始化仿真环境
# base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
# mcm.mgm.gen_frame().attach_to(base)

# # 初始化机器人
# robot = cbt.Cobotta(pos=rm.vec(0.1, .3, .5), enable_cc=True)
# joint_ranges = robot.jnt_ranges  # 每个关节的上下限
# n_samples = 10000  # LHS采样数量
# seed = 42

# # LHS采样函数
# def lhs_joint_space_sampling(joint_ranges, n_samples, seed=None):
#     n_dof = joint_ranges.shape[0]
#     sampler = qmc.LatinHypercube(d=n_dof, seed=seed)
#     unit_samples = sampler.random(n=n_samples)
#     samples = qmc.scale(unit_samples, joint_ranges[:, 0], joint_ranges[:, 1])
#     return samples

# # 执行采样和 FK
# joint_samples = lhs_joint_space_sampling(joint_ranges, n_samples, seed)
# # joint_samples = np.array([robot.rand_conf() for _ in range(n_samples)])
# tcp_positions = np.array([robot.fk(jnt_values=joints)[0] for joints in joint_samples])

# ### ---------------- 可视化部分 ---------------- ###

# # 图 1: 3D 散点图（TCP 分布）
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(tcp_positions[:, 0], tcp_positions[:, 1], tcp_positions[:, 2], s=10, alpha=0.6)
# ax.set_xlabel('X [m]')
# ax.set_ylabel('Y [m]')
# ax.set_zlabel('Z [m]')
# ax.set_title('TCP Position Distribution (LHS Samples)')
# plt.tight_layout()
# plt.show()

# # 打印 TCP 分布范围
# xlim = [tcp_positions[:, 0].min(), tcp_positions[:, 0].max()]
# ylim = [tcp_positions[:, 1].min(), tcp_positions[:, 1].max()]
# zlim = [tcp_positions[:, 2].min(), tcp_positions[:, 2].max()]
# print(f"X range: {xlim}")
# print(f"Y range: {ylim}")
# print(f"Z range: {zlim}")

# # 图 2: 3D 密度热图（根据局部点密度估算）
# nbrs = NearestNeighbors(n_neighbors=6).fit(tcp_positions)
# distances, _ = nbrs.kneighbors(tcp_positions)
# density_score = 1 / (np.mean(distances[:, 1:], axis=1) + 1e-6)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# p = ax.scatter(tcp_positions[:, 0], tcp_positions[:, 1], tcp_positions[:, 2],
#                c=density_score, cmap='viridis', s=10)
# ax.set_xlabel('X [m]')
# ax.set_ylabel('Y [m]')
# ax.set_zlabel('Z [m]')
# ax.set_title('TCP Density Heatmap (LHS Samples)')
# fig.colorbar(p, ax=ax, label='Inverse Mean Distance (Density)')
# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt

# # 图 1: XY 热度图（俯视图）
# plt.figure(figsize=(6, 5))
# plt.hexbin(tcp_positions[:, 0], tcp_positions[:, 1], gridsize=50, cmap='Blues', bins='log')
# plt.xlabel('X [m]')
# plt.ylabel('Y [m]')
# plt.title('XY Projection (Density Heatmap)')
# plt.colorbar(label='log(Count)')
# plt.axis('equal')
# plt.tight_layout()
# plt.show()

# # 图 2: XZ 热度图（右视图）
# plt.figure(figsize=(6, 5))
# plt.hexbin(tcp_positions[:, 0], tcp_positions[:, 2], gridsize=50, cmap='Greens', bins='log')
# plt.xlabel('X [m]')
# plt.ylabel('Z [m]')
# plt.title('XZ Projection (Density Heatmap)')
# plt.colorbar(label='log(Count)')
# plt.axis('equal')
# plt.tight_layout()
# plt.show()

# # 图 3: YZ 热度图（正视图）
# plt.figure(figsize=(6, 5))
# plt.hexbin(tcp_positions[:, 1], tcp_positions[:, 2], gridsize=50, cmap='Reds', bins='log')
# plt.xlabel('Y [m]')
# plt.ylabel('Z [m]')
# plt.title('YZ Projection (Density Heatmap)')
# plt.colorbar(label='log(Count)')
# plt.axis('equal')
# plt.tight_layout()
# plt.show()


'''
comparison with random sampling
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.neighbors import NearestNeighbors

from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.rs007l.rs007l as rs007l
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.manipulators.ur3e.ur3e as ur3e
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi
import wrs.robot_sim.robots.cobotta_pro1300.cobotta_pro1300 as cbtpro1300
import wrs.robot_sim.robots.cobotta_pro900.cobotta_pro900_spine as cbtpro900

# 初始化仿真环境
base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

# robot = cbt.Cobotta(pos=rm.vec(0.1, .3, .5), enable_cc=True)
# robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)
# robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), enable_cc=True)
robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
joint_ranges = robot.jnt_ranges
n_samples = 10000
seed = 42

def lhs_joint_space_sampling(joint_ranges, n_samples, seed=None):
    n_dof = joint_ranges.shape[0]
    sampler = qmc.LatinHypercube(d=n_dof, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    samples = qmc.scale(unit_samples, joint_ranges[:, 0], joint_ranges[:, 1])
    return samples

# ---- 采样 ----
joint_samples_lhs = lhs_joint_space_sampling(joint_ranges, n_samples, seed)
joint_samples_rand = np.array([robot.rand_conf() for _ in range(n_samples)])

# ---- FK 得到 TCP ----
tcp_lhs = np.array([robot.fk(jnt_values=joints)[0] for joints in joint_samples_lhs])
tcp_rand = np.array([robot.fk(jnt_values=joints)[0] for joints in joint_samples_rand])

## ---------------- 3D 散点对比 ---------------- ###
# fig = plt.figure(figsize=(12, 5))
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')

# ax1.scatter(tcp_lhs[:, 0], tcp_lhs[:, 1], tcp_lhs[:, 2], s=5, alpha=0.5)
# ax1.set_title('TCP (LHS Sampling)')
# ax1.set_xlabel('X [m]')
# ax1.set_ylabel('Y [m]')
# ax1.set_zlabel('Z [m]')

# ax2.scatter(tcp_rand[:, 0], tcp_rand[:, 1], tcp_rand[:, 2], s=5, alpha=0.5, color='orange')
# ax2.set_title('TCP (Random Sampling)')
# ax2.set_xlabel('X [m]')
# ax2.set_ylabel('Y [m]')
# ax2.set_zlabel('Z [m]')

# plt.tight_layout()
# plt.show()


# ---------------- 热度投影 散点对比 ---------------- 
# proj_titles = ['XY', 'XZ', 'YZ']
# coords = [(0, 1), (0, 2), (1, 2)]
# colormap = 'plasma'
# axes_labels = [('X [m]', 'Y [m]'), ('X [m]', 'Z [m]'), ('Y [m]', 'Z [m]')]
# padding_ratio = 0.05  # 扩展 5% 边距

# for i, (idx1, idx2) in enumerate(coords):
#     # 联合坐标轴范围 + 边缘扩展
#     all_x = np.concatenate([tcp_lhs[:, idx1], tcp_rand[:, idx1]])
#     all_y = np.concatenate([tcp_lhs[:, idx2], tcp_rand[:, idx2]])
#     x_range = all_x.max() - all_x.min()
#     y_range = all_y.max() - all_y.min()
#     xlim = [all_x.min() - padding_ratio * x_range, all_x.max() + padding_ratio * x_range]
#     ylim = [all_y.min() - padding_ratio * y_range, all_y.max() + padding_ratio * y_range]

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#     # LHS 图
#     hb_lhs = axes[0].hexbin(tcp_lhs[:, idx1], tcp_lhs[:, idx2],
#                             gridsize=50, cmap=colormap, bins='log')
#     axes[0].set_title(f'{proj_titles[i]} Projection (LHS)')
#     axes[0].set_xlabel(axes_labels[i][0])
#     axes[0].set_ylabel(axes_labels[i][1])
#     axes[0].set_xlim(xlim)
#     axes[0].set_ylim(ylim)
#     axes[0].set_aspect('equal')
#     cbar_lhs = fig.colorbar(hb_lhs, ax=axes[0], orientation='vertical')
#     cbar_lhs.set_label('log(Count)', fontsize=10)

#     # Random 图
#     hb_rand = axes[1].hexbin(tcp_rand[:, idx1], tcp_rand[:, idx2],
#                              gridsize=50, cmap=colormap, bins='log')
#     axes[1].set_title(f'{proj_titles[i]} Projection (Random)')
#     axes[1].set_xlabel(axes_labels[i][0])
#     axes[1].set_ylabel(axes_labels[i][1])
#     axes[1].set_xlim(xlim)
#     axes[1].set_ylim(ylim)
#     axes[1].set_aspect('equal')
#     cbar_rand = fig.colorbar(hb_rand, ax=axes[1], orientation='vertical')
#     cbar_rand.set_label('log(Count)', fontsize=10)

#     plt.tight_layout()
#     plt.show()

import matplotlib.pyplot as plt
import numpy as np

proj_titles = ['XY', 'XZ', 'YZ']
coords = [(0, 1), (0, 2), (1, 2)]
axes_labels = [('X [m]', 'Y [m]'), ('X [m]', 'Z [m]'), ('Y [m]', 'Z [m]')]
colormap = 'plasma'
padding_ratio = 0.05  # 5% 边缘扩展

for i, (idx1, idx2) in enumerate(coords):
    all_x = tcp_rand[:, idx1]
    all_y = tcp_rand[:, idx2]
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    xlim = [all_x.min() - padding_ratio * x_range, all_x.max() + padding_ratio * x_range]
    ylim = [all_y.min() - padding_ratio * y_range, all_y.max() + padding_ratio * y_range]

    fig, ax = plt.subplots(figsize=(10, 8))

    hb = ax.hexbin(all_x, all_y, gridsize=50, cmap=colormap, bins='log')
    ax.set_title(f'{proj_titles[i]} Projection (Random)', fontsize=12)
    ax.set_xlabel(axes_labels[i][0], fontsize=11)
    ax.set_ylabel(axes_labels[i][1], fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')

    cbar = fig.colorbar(hb, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label('log(Count)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(f"0000_test_programs/nn_ik/res_figs/0621_save/tcp_rand_{proj_titles[i].lower()}_projection.png", dpi=600, bbox_inches='tight')
    plt.close()




# ---------------- 熵计算 ----------------
from scipy.stats import entropy

def compute_occupancy_entropy(points, bins=30):
    """
    在3D空间中对TCP位置进行体素分箱 然后计算点的空间分布的熵。
    """
    hist, _ = np.histogramdd(points, bins=bins)
    flat_hist = hist.flatten()
    prob = flat_hist / np.sum(flat_hist)
    prob = prob[prob > 0]  # 只保留非零项以避免log(0)
    return entropy(prob, base=2)  # 熵的单位是 bit

lhs_entropy = compute_occupancy_entropy(tcp_lhs, bins=30)
rand_entropy = compute_occupancy_entropy(tcp_rand, bins=30)

def max_entropy(bins):
    return np.log2(bins**3)

print(f"Theoretical Max Entropy (bins=30): {max_entropy(30):.4f} bits")

print(f"Entropy (LHS): {lhs_entropy:.4f} bits")
print(f"Entropy (Random): {rand_entropy:.4f} bits")

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def plot_3d_density(tcp_positions, title='3D TCP Density Scatter', cmap='viridis'):
    nbrs = NearestNeighbors(n_neighbors=6).fit(tcp_positions)
    distances, _ = nbrs.kneighbors(tcp_positions)
    density_score = 1 / (np.mean(distances[:, 1:], axis=1) + 1e-6)  # 排除自身距离

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(tcp_positions[:, 0], tcp_positions[:, 1], tcp_positions[:, 2],
                    c=density_score, cmap=cmap, s=10, alpha=0.6)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label='Inverse Mean Distance (Density)', shrink=0.8, pad=0.02)
    
    plt.tight_layout()
    plt.savefig(f"0000_test_programs/nn_ik/res_figs/0621_save/{title.replace(' ', '_').lower()}.png", dpi=600, bbox_inches='tight')
    plt.show()

# 示例调用
# plot_3d_density(tcp_lhs, title='TCP Density Scatter (LHS Sampling)')
plot_3d_density(tcp_rand, title='TCP Density Scatter (Random Sampling)', cmap='viridis')

