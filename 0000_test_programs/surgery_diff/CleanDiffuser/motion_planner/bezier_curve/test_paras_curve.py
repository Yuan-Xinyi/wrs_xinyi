import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# -----------------------------
# Quadratic Bézier curve
# -----------------------------
def quadratic_bezier(P0, P1, P2, num=200):
    t = np.linspace(0, 1, num)
    curve = (1 - t)[:, None]**2 * P0 + 2 * (1 - t)[:, None] * t[:, None] * P1 + (t**2)[:, None] * P2
    return curve

# 控制点
P0 = np.array([0.0, 0.0])
P1 = np.array([0.5, 1.5])
P2 = np.array([1.5, 0.0])

# 计算Bézier
bezier = quadratic_bezier(P0, P1, P2)

# -----------------------------
# Quadratic B-spline curve
# -----------------------------
# 控制点
ctrl = np.array([[0.0, 0.0],
                 [0.5, 1.5],
                 [1.5, 0.0]])

k = 2  # 二次B样条
n = len(ctrl)
# Open uniform knot vector
t = np.concatenate(([0]*k, np.linspace(0,1,n-k+1), [1]*k))
spline = BSpline(t, ctrl, k)
ts = np.linspace(0, 1, 200)
bspline = spline(ts)

# -----------------------------
# 可视化
# -----------------------------
plt.figure(figsize=(6,4))
# Bézier
plt.plot(bezier[:,0], bezier[:,1], label="Quadratic Bézier")
plt.plot([P0[0], P1[0], P2[0]], [P0[1], P1[1], P2[1]], 'o--', alpha=0.5)

# B-spline
plt.plot(bspline[:,0], bspline[:,1], label="Quadratic B-spline")
plt.plot(ctrl[:,0], ctrl[:,1], 'o--', alpha=0.5)

plt.legend()
plt.axis("equal")
plt.title("Quadratic Bézier vs Quadratic B-spline")
plt.show()
