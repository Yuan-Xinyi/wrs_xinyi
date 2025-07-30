import numpy as np
import samply
from wrs.basis.robot_math import vec
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.robots.cobotta_pro1300.cobotta_pro1300 as cbtpro1300
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Robot 配置列表
# robot_constructors = {
#     'cobotta': lambda: cbt.Cobotta(pos=vec(0.1, .3, .5), enable_cc=True),
#     'cobotta_pro1300': lambda: cbtpro1300.CobottaPro1300WithRobotiq140(pos=vec(0.1, .3, .5), enable_cc=True),
#     'ur3': lambda: ur3.UR3(pos=vec(0.1, .3, .5), enable_cc=True),
#     'yumi': lambda: yumi.YumiSglArm(pos=vec(0.1, .3, .5), enable_cc=True),
# }
robot_constructors = {
    'ur3': lambda: ur3.UR3(pos=vec(0.1, .3, .5), enable_cc=True)
}

for name, ctor in robot_constructors.items():
    if name == 'yumi':
        n_samples = 1814400*2
    else:
        n_samples = 2592000

    robot = ctor()
    jnt_ranges = robot.jnt_ranges  # list of (low, high)
    D = len(jnt_ranges)
    print(f"[{name}] joint dims =", D)
    # 生成 CVT 样本
    points = samply.hypercube.cvt(n_samples, D)
    print(f"[{name}] Generated cvt samples:", points.shape)
    low = np.array([rng[0] for rng in jnt_ranges])
    high = np.array([rng[1] for rng in jnt_ranges])
    scaled = low + (high - low) * points  # 映射到真实角度范围
    print(f"[{name}] Scaled samples shape:", scaled.shape)
    fname = f'cvt_joint_samples_{name}_largest.npy'
    np.save(fname, scaled)
    print(f"[{name}] Saved joint samples to `{fname}`\n")

    # 假设 scaled 是任意一款机器人的最终样本
    nbrs = NearestNeighbors(n_neighbors=2).fit(scaled)
    distances, _ = nbrs.kneighbors(scaled)
    nn = distances[:,1]
    print("Mean nn distance:", nn.mean(), "CV:", nn.std()/nn.mean())
    # plt.hist(nn, bins=50)
    # plt.title("Nearest neighbor for "+name)
    # # plt.show()
    # plt.savefig(f'cvt_nn_hist_{name}_sacle9.png')