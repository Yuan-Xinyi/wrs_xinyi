import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline

import zarr
import numpy as np
import matplotlib.pyplot as plt

import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm

ruckig_root = zarr.open('/home/lqin/zarr_datasets/franka_ruckig_100hz.zarr', mode='r')

'''new the dataset'''
dataset_name = '/home/lqin/zarr_datasets/franka_ruckig_100hz_polynomial.zarr'
store = zarr.DirectoryStore(dataset_name)
root = zarr.group(store=store)
print('dataset created in:', dataset_name)
meta_group = root.create_group("meta")
data_group = root.create_group("data")
episode_ends_ds = meta_group.create_dataset("episode_ends", shape=(0,), chunks=(1,), dtype=np.float32, append=True)
poly_coef_ds = data_group.create_dataset("control_points", shape=(0, 9), chunks=(1, 9), dtype=np.float32, append=True)

'''polynomial parameter'''
degree = 8
dt = 0.01

episode_ends_counter = 0
for traj_id in range(len(ruckig_root['meta']['episode_ends'][:])):
# for traj_id in range(10):
    '''extract traj pos'''
    print('**' * 100)
    print(f'traj id: {traj_id}')
    traj_start = int(np.sum(ruckig_root['meta']['episode_ends'][:traj_id]))
    traj_end = int(np.sum(ruckig_root['meta']['episode_ends'][:traj_id + 1]))
    jnt_pos_list = ruckig_root['data']['jnt_pos'][traj_start:traj_end]
    print(f'start conf: {jnt_pos_list[0]}, end conf: {jnt_pos_list[-1]}')

    '''construct b-spline'''
    T = len(jnt_pos_list)
    t = np.linspace(0, (T - 1) * dt, T)
    num_joints = jnt_pos_list.shape[1]  # recconstruct multiple joints
    s = np.linspace(0, 1, T)
    
    '''approximate b-spline'''
    ctrl_values_list = []

    for j in range(num_joints):
        x = jnt_pos_list[:, j]
        ctrl_values = np.interp(ctrl_points, s, x)
        ctrl_values[0] = x[0]  # make sure start aligned
        ctrl_values[-1] = x[-1]  # make sure end aligned
        ctrl_values_list.append(ctrl_values)

    # control points to 2D array (each column is a joint)
    ctrl_values_matrix = np.vstack(ctrl_values_list).T
    spline = make_lsq_spline(s, jnt_pos_list, knots, degree)

    '''print parameters and save into zarr'''
    print(f'c shape: {spline.c.shape}')
    control_points_ds.append(np.array(spline.c))
    episode_ends_counter += len(spline.c)
    episode_ends_ds.append(np.array([episode_ends_counter], dtype=np.int32))