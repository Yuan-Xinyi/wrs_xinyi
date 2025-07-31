# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.neighbors import NearestNeighbors
# # from mpl_toolkits.mplot3d import Axes3D
# # import samply

# # # 设置全局字体为 Times New Roman，字号为 16
# # plt.rcParams['font.family'] = 'Times New Roman'
# # plt.rcParams['font.size'] = 16

# # # 参数设置
# # n_samples = 10000

# # # === 1. CVT 采样 ===
# # cvt_samples = samply.hypercube.cvt(n_samples, 3)

# # # === 2. 轴向独立均匀采样 ===
# # grid_x = np.random.uniform(0, 1, n_samples)
# # grid_y = np.random.uniform(0, 1, n_samples)
# # grid_z = np.random.uniform(0, 1, n_samples)
# # grid_samples = np.stack([grid_x, grid_y, grid_z], axis=1)

# # # === 画图：3D 分布对比 ===
# # fig = plt.figure(figsize=(12, 5))

# # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
# # ax1.scatter(cvt_samples[:, 0], cvt_samples[:, 1], cvt_samples[:, 2], s=3, c='blue')
# # ax1.set_title('CVT Sampling')
# # ax1.set_xlim(0, 1)
# # ax1.set_ylim(0, 1)
# # ax1.set_zlim(0, 1)

# # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# # ax2.scatter(grid_samples[:, 0], grid_samples[:, 1], grid_samples[:, 2], s=3, c='green')
# # ax2.set_title('Axis-wise Sampling')
# # ax2.set_xlim(0, 1)
# # ax2.set_ylim(0, 1)
# # ax2.set_zlim(0, 1)

# # plt.suptitle("3D Sampling Distribution Comparison")
# # plt.tight_layout()
# # plt.show()

# # # === 最近邻函数 ===
# # def get_nn_distances(samples):
# #     nbrs = NearestNeighbors(n_neighbors=2).fit(samples)
# #     dists, _ = nbrs.kneighbors(samples)
# #     return dists[:, 1]  # 排除自身距离

# # # === 最近邻距离直方图 ===
# # cvt_nn = get_nn_distances(cvt_samples)
# # grid_nn = get_nn_distances(grid_samples)

# # plt.figure(figsize=(6, 4))
# # plt.hist(cvt_nn, bins=30, alpha=0.7, label='CVT', color='blue')
# # plt.hist(grid_nn, bins=30, alpha=0.7, label='Axis-wise', color='green')
# # plt.title("Nearest Neighbor Distance Distribution")
# # plt.xlabel("Distance")
# # plt.ylabel("Count")
# # plt.legend()
# # plt.tight_layout()
# # plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import samply

# # --------- 全局字体 ---------
# plt.rcParams["font.family"] = "serif"   # 若想用 Times 且系统已安装可写 "Times New Roman"
# plt.rcParams["font.size"]   = 16

# # --------- 参数 ---------
# n_samples = 100   # 初始点数
# n_iters   = 30    # 动画帧数（每帧一次 CVT 迭代）

# pts = np.random.rand(n_samples, 2)      # 初始随机点

# # --------- 画布初始化 ---------
# fig, ax = plt.subplots(figsize=(6, 6))
# sc     = ax.scatter(pts[:, 0], pts[:, 1], s=12, c="steelblue")
# title  = ax.set_title("CVT Sampling Iteration: 0")
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)

# # --------- 把任何返回值都安全转成 (n,2) ndarray ---------
# def to_array(p):
#     p = np.asarray(p, dtype=float)   # 保证 float
#     p = p.reshape(-1, 2)             # 不论原来什么维度都强制成 (n, 2)
#     return p

# # --------- 更新函数 ---------
# def update(frame):
#     global pts
#     try:
#         # 每帧迭代一步；部分 samply 版本返回 list[list[float]]
#         pts = to_array(samply.hypercube.cvt(pts))

#         sc.set_offsets(pts)                          # 更新散点
#         title.set_text(f"CVT Sampling Iteration: {frame+1}")
#     except Exception as e:
#         print(f"[错误] 第 {frame+1} 帧出错：{e}")
#         ani.event_source.stop()                      # 停止动画
#     return sc,                                       # 注意逗号→返回 tuple(Artist,)

# # --------- 创建动画 ---------
# ani = FuncAnimation(fig, update, frames=n_iters, interval=300)

# plt.tight_layout()
# plt.show()

# # 如需保存动图，可在最后加：
# # ani.save("cvt_process.gif", fps=5, writer="pillow")


"""
quick_cvt_demo.py
一次性展示 CVT 在迭代 0 / 1 / 2 / 5 / 30 步时的点云分布
"""

import numpy as np
import matplotlib.pyplot as plt
import samply

# -------- 参数 --------
n_samples      = 100
iter_show_list = [0, 1, 2, 5, 30, 2000000]    # 想看更多/更少步，直接改这里
n_cols         = len(iter_show_list)

# -------- 生成各步点集 --------
# 初始随机
initial_pts = np.random.rand(n_samples, 2)
pts_by_iter = [initial_pts]

# 之后直接调用 samply 的迭代结果
for k in iter_show_list[1:]:
    pts_k = samply.hypercube.cvt(count=n_samples,
                                 dimensionality=2,
                                 max_iterations=k)
    pts_by_iter.append(np.asarray(pts_k, float).reshape(-1, 2))

# -------- 绘图 --------
fig, axes = plt.subplots(1, n_cols, figsize=(3*n_cols, 3))
for ax, pts, k in zip(axes, pts_by_iter, iter_show_list):
    ax.scatter(pts[:, 0], pts[:, 1], s=12, c="steelblue")
    ax.set_title(f"Iter {k}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")
    ax.axis("off")

plt.suptitle("CVT Convergence: from Random to Uniform")
plt.tight_layout()
plt.show()










