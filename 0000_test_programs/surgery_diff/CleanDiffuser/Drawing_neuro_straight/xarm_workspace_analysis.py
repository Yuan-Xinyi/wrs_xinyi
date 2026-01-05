'''wrs reliance'''
from tqdm import tqdm
from wrs import wd, rm, mcm
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
import wrs.modeling.geometric_model as mgm

'''global variables'''
import time
import pickle
import numpy as np
import torch
torch.autograd.set_detect_anomaly(False)
import matplotlib.pyplot as plt

xarm = xarm6_sim.XArmLite6Miller(enable_cc=True)
pos_list = []
rotmat_list = []

for i in tqdm(range(int(1e6))):
    jnt = xarm.rand_conf()
    pos, rotmat = xarm.fk(jnt)
    pos_list.append(pos)
    rotmat_list.append(rotmat)
pos_arr = np.array(pos_list)


# # Plotting the workspace
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pos_arr[:,0], pos_arr[:,1], pos_arr[:,2], s=1, alpha=0.5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('XArm Lite6 FK Workspace')
# plt.show()

z_value = 0 
epsilon = 1e-3
mask = np.abs(pos_arr[:,2] - z_value) < epsilon
xy_points = pos_arr[mask][:, :2]

plt.figure()
plt.scatter(xy_points[:,0], xy_points[:,1], s=2, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Workspace Cross-section at z={z_value}')
if len(xy_points) > 3:
    from scipy.spatial import ConvexHull
    hull = ConvexHull(xy_points)
    plt.fill(xy_points[hull.vertices,0], xy_points[hull.vertices,1], alpha=0.3, label=f'Area={hull.area:.4f}')
    plt.legend()
    print(f"Area: {hull.area:.4f}")

    contour = xy_points[hull.vertices]
    np.save(f'xarm_contour_z{z_value:.3f}.npy', contour)
    import pickle
    with open(f'xarm_contour_z{z_value:.3f}.pkl', 'wb') as f:
        pickle.dump(contour, f)
    print(f"the contour has been saved as xarm_contour_z{z_value:.3f}.npy and .pkl")
else:
    print("Too few points in cross-section to calculate area")
plt.show()