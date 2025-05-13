import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline

# 生成示例轨迹：带噪声的正弦波
t = np.linspace(0, 4 * np.pi, 1000)
x = np.sin(t) + 0.05 * np.random.randn(1000)

# 参数化变量 s
s = np.linspace(0, 1, len(x))

# 选择控制点数量
degree = 3
num_ctrl_pts = 20
knots = np.linspace(0, 1, num_ctrl_pts - degree + 1)
knots = np.concatenate(([0] * degree, knots, [1] * degree))

# 使用 make_lsq_spline 从轨迹中生成控制点
spline = make_lsq_spline(s, x, knots, degree)
control_points = spline.c  # 提取控制点

# 通过控制点重建轨迹
s_fine = np.linspace(0, 1, 1000)
reconstructed_trajectory = spline(s_fine)

# 可视化：原始轨迹 vs 重建轨迹
plt.figure(figsize=(12, 6))
plt.plot(s, x, 'gray', linestyle="--", label="Original Trajectory", alpha=0.5)
plt.plot(s_fine, reconstructed_trajectory, 'b-', label="Reconstructed Trajectory")
plt.plot(knots, control_points, 'ro', label="Control Points", markersize=6)
plt.title("Trajectory Reconstruction using Control Points (B-Spline)")
plt.xlabel("s (parameter)")
plt.ylabel("x (value)")
plt.legend()
plt.grid(True)
plt.show()
