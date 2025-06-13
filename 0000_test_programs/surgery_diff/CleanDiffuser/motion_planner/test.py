import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 已知起点和终点
start = np.array([-0.56887997, -0.06892034, 0.321434])
end = np.array([-0.39887989, -0.06892035, 0.32143399])

# 中点
midpoint = (start + end) / 2

# 计算边向量和边长
vec = end - start
length = np.linalg.norm(vec)

# 找一个垂直于 vec 的单位向量用于确定第三点方向
# 这里我们可以用简单的法向计算（比如叉乘 vec 和 z 轴）
z_axis = np.array([0, 0, 1])
normal_vec = np.cross(vec, z_axis)
if np.linalg.norm(normal_vec) < 1e-6:
    # 如果刚好平行于 z 轴，改用 y 轴
    y_axis = np.array([0, 1, 0])
    normal_vec = np.cross(vec, y_axis)

normal_vec = normal_vec / np.linalg.norm(normal_vec)

# 构造等边三角形，第三个点在边的法向方向，距离为 sqrt(3)/2 * length
height = np.sqrt(3) / 2 * length
third = midpoint + normal_vec * height

# 准备点用于绘图
points = np.array([start, end, third, start])  # 闭合三角形

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(points[:, 0], points[:, 1], points[:, 2], marker='o')
ax.set_title("3D Triangle with One Given Edge")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
