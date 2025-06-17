import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline
from scipy.optimize import minimize
import zarr
from tqdm import tqdm

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
poly_coef_ds = data_group.create_dataset("poly_coef", shape=(0, 8), chunks=(1, 8), dtype=np.float32, append=True)
start_conf_ds = data_group.create_dataset("start_conf", shape=(0, 7), chunks=(1, 7), dtype=np.float32, append=True)
goal_conf_ds = data_group.create_dataset("goal_conf", shape=(0, 7), chunks=(1, 7), dtype=np.float32, append=True)

'''polynomial parameter'''
degree = 7
dt = 0.01

def constrained_polynomial_fit(s, y, degree):
    """
    基于 s 参数化 0-1 的多项式拟合，带约束：位置、速度、加速度和抖动在起始和末尾为 0。
    """
    p_init = np.polyfit(s, y, degree)
    
    def objective(p):
        poly = np.poly1d(p)
        residuals = np.sum((poly(s) - y) ** 2)
        return residuals

    # 约束：起始和终止的速度、加速度、抖动和位置
    def position_constraint(p):
        poly = np.poly1d(p)
        return [poly(0) - y[0], poly(1) - y[-1]]

    def velocity_constraint(p):
        poly = np.poly1d(p)
        return [np.polyder(poly, 1)(0), np.polyder(poly, 1)(1)]

    def acceleration_constraint(p):
        poly = np.poly1d(p)
        return [np.polyder(poly, 2)(0), np.polyder(poly, 2)(1)]

    def jerk_constraint(p):
        poly = np.poly1d(p)
        return [np.polyder(poly, 3)(0), np.polyder(poly, 3)(1)]

    constraints = [
        {"type": "eq", "fun": position_constraint},
        {"type": "eq", "fun": velocity_constraint},
        {"type": "eq", "fun": acceleration_constraint},
        {"type": "eq", "fun": jerk_constraint}
    ]

    result = minimize(objective, p_init, constraints=constraints, method='SLSQP')
    return np.poly1d(result.x)

episode_ends_counter = 0
for traj_id in tqdm(range(len(ruckig_root['meta']['episode_ends'][:]))):
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

    for j in range(num_joints):
        episode_ends_counter += 1
        y = jnt_pos_list[:, j]
        poly_s = constrained_polynomial_fit(s, y, degree)
        poly_coef_ds.append((np.poly1d(poly_s).coefficients).reshape(1, 8))
        start_conf_ds.append(np.array([jnt_pos_list[0]]).reshape(1, 7))
        goal_conf_ds.append(np.array([jnt_pos_list[-1]]).reshape(1, 7))
    
    episode_ends_ds.append(np.array([episode_ends_counter], dtype=np.int32))