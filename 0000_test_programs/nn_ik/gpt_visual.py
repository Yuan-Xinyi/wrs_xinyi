import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate torus (T^2)
theta1 = np.linspace(0, 2 * np.pi, 100)
theta2 = np.linspace(0, 2 * np.pi, 100)
theta1, theta2 = np.meshgrid(theta1, theta2)

R = 2  # Major radius
r = 1  # Minor radius

X = (R + r * np.cos(theta2)) * np.cos(theta1)
Y = (R + r * np.cos(theta2)) * np.sin(theta1)
Z = r * np.sin(theta2)

# Generate bounded Euclidean space (R^2 example)
x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-1.5, 1.5, 100)
x, y = np.meshgrid(x, y)
z = np.zeros_like(x)  # Flat plane

# Visualization
fig = plt.figure(figsize=(16, 8))

# Torus (T^2)
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, rstride=5, cstride=5, color='lightblue', edgecolor='grey', alpha=0.7)
ax1.set_title('Torus (T^2): Joint Space Without Limits', fontsize=12)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Euclidean Space (R^2 example)
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x, y, z, color='lightcoral', edgecolor='grey', alpha=0.7)
ax2.set_title('Euclidean Space (R^2): Joint Space With Limits', fontsize=12)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.tight_layout()
plt.show()
