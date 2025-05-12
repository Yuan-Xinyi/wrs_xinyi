import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# 生成带噪声的示例数据
x = np.linspace(0, 4, 50)
y = np.sin(x) + 0.3 * np.random.randn(50)

# 1. 完全插值 (s=0)
tck_interp, _ = splprep([x, y], s=0)
x_interp, y_interp = splev(np.linspace(0, 1, 500), tck_interp)

# 2. 适度平滑 (s=5)
tck_smooth, _ = splprep([x, y], s=5)
x_smooth, y_smooth = splev(np.linspace(0, 1, 500), tck_smooth)

# 3. 高度平滑 (s=20)
tck_high_smooth, _ = splprep([x, y], s=20)
x_high_smooth, y_high_smooth = splev(np.linspace(0, 1, 500), tck_high_smooth)

# 绘制对比
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'ro', label="Original Data (Noisy)", markersize=5)
plt.plot(x_interp, y_interp, 'g--', label="Interpolating Spline (s=0)")
plt.plot(x_smooth, y_smooth, 'b-', label="Smooth Spline (s=5)")
plt.plot(x_high_smooth, y_high_smooth, 'purple', linestyle="--", label="Highly Smooth Spline (s=20)")

plt.title("Comparison: Smoothing Effect of `s` in splprep")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
