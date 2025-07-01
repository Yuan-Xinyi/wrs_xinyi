import numpy as np
import matplotlib.pyplot as plt
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

# === 可视化 3D 密度 ===
def plot_3d_density(tcp_positions, title='3D TCP Density Scatter', cmap='viridis'):
    nbrs = NearestNeighbors(n_neighbors=6).fit(tcp_positions)
    distances, _ = nbrs.kneighbors(tcp_positions)
    density_score = 1 / (np.mean(distances[:, 1:], axis=1) + 1e-6)  # 排除自身

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
    plt.savefig(f"0000_test_programs/nn_ik/res_figs/0621_save/{title.replace(' ', '_').lower()}.png",
                dpi=600, bbox_inches='tight')
    plt.show()

# ---------- 熵计算 ----------
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
def compute_entropy_and_relative_entropy(points, bins=30):
    hist, _ = np.histogramdd(points, bins=bins)
    flat_hist = hist.flatten()
    prob = flat_hist / np.sum(flat_hist)
    prob = prob[prob > 0]  # 避免 log(0)
    
    H = entropy(prob, base=2)  # 实际熵 H(P)
    H_max = np.log2(bins ** 3)  # 理论最大熵 log2(空间总格子数)
    D_kl = H_max - H            # 相对熵（与均匀分布的 KL 散度）
    return H, H_max, D_kl

def print_entropy_stats(points, label, bins_list):
    print(f"=== Entropy Analysis for {label} ===")
    print(f"{'Bins':>6} | {'H(P)':>8} | {'H_max':>8} | {'D_KL':>8} | {'H/H_max':>8}")
    print("-" * 50)
    for bins in bins_list:
        H, H_max, D_kl = compute_entropy_and_relative_entropy(points, bins)
        ratio = H / H_max * 100
        print(f"{bins:6d} | {H:8.2f} | {H_max:8.2f} | {D_kl:8.2f} | {ratio:8.2f}%")
    print("")

# 初始化
base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)
# robot = cbt.Cobotta(pos=rm.vec(0.1, .3, .5), enable_cc=True)
# robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)
robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), enable_cc=True)
# robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)

# 参数设置
n_trials = 50000
'''c1'''
# voxel_size = 0.06
'''c0'''
voxel_size = 0.02  # 每个 voxel 的大小
max_per_voxel = 1  # 每个voxel最多保留几个样本

np.random.seed(42)
voxel_map = dict()
tcp_points = []

for _ in range(n_trials):
    q = robot.rand_conf()
    tcp = robot.fk(jnt_values=q)[0]  # [0]是位置
    voxel_key = tuple(np.floor(tcp / voxel_size).astype(int))

    # 每个 voxel 最多保留 max_per_voxel 个样本
    if voxel_key not in voxel_map:
        voxel_map[voxel_key] = [tcp]
        tcp_points.append(tcp)
    elif len(voxel_map[voxel_key]) < max_per_voxel:
        voxel_map[voxel_key].append(tcp)
        tcp_points.append(tcp)

tcp_points = np.array(tcp_points)
print(f"Sampled TCP points (filtered): {len(tcp_points)}")

bins_list = [20, 30, 50, 100]
print_entropy_stats(tcp_points, label="Random Sampling", bins_list=bins_list)

# 可视化
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tcp_points[:, 0], tcp_points[:, 1], tcp_points[:, 2], s=3, alpha=0.5)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('TCP Distribution (Joint Sampling + Voxel Filtering)')
plt.tight_layout()
plt.show()

plot_3d_density(tcp_points, title="TCP Density Scatter (Random Sampling)")