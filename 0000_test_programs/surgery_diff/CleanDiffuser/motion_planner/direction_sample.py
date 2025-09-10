import numpy as np
import matplotlib.pyplot as plt

def fibonacci_sphere(samples=20):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # 黄金角

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y 从 1 到 -1
        radius = np.sqrt(1 - y * y)           # 圆半径

        theta = phi * i                       # 旋转角度
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

# 生成 20 个方向
directions = fibonacci_sphere(6)

# 可视化
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(directions[:,0], directions[:,1], directions[:,2], c='b', s=20)

# 画一个球体参考
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 25)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(x, y, z, color="lightgray", linewidth=0.5, alpha=0.3)

ax.set_title("Fibonacci Sphere Sampling")
ax.set_box_aspect([1,1,1])  # 保持比例一致
plt.show()
