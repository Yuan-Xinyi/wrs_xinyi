import zarr
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from ruckig import InputParameter, OutputParameter, Result, Ruckig

import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
robot_s = franka.FrankaResearch3(enable_cc=True)
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)

def plot_details(robot_s, jnt_pos_list, jnt_vel_list, jnt_acc_list):
    sampling_interval = 0.001
    time_points = np.arange(0, len(jnt_pos_list) * sampling_interval, sampling_interval)[:len(jnt_pos_list)]

    plt.figure(figsize=(10, 3 * robot_s.n_dof))

    for i in range(robot_s.n_dof):
        plt.subplot(robot_s.n_dof, 1, i + 1)
        pos = [p[i] for p in jnt_pos_list]
        vel = [v[i] for v in jnt_vel_list]
        acc = [a[i] for a in jnt_acc_list]

        plt.plot(time_points, pos, label='Position')
        plt.plot(time_points, vel, label='Velocity')
        plt.plot(time_points, acc, label='Acceleration')

        plt.ylabel(f'DoF {i}')
        plt.legend()
        plt.grid(True)

    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.show()

'''test the joint configuration'''
jnt = robot_s.rand_conf()
print('random joint configuration:', repr(jnt))
robot_s.goto_given_conf(jnt_values=jnt)
robot_s.gen_meshmodel(alpha=0.2).attach_to(base)

for i in np.linspace(-0.5, 1.5, 10):
    tmp = copy(jnt)
    tmp[6] += i
    robot_s.goto_given_conf(jnt_values=tmp)
    robot_s.gen_meshmodel(alpha=0.2, rgb=[1, 0, 0]).attach_to(base)
base.run()

# root = zarr.open('/home/lqin/zarr_datasets/franka_kinodyn_obstacles_3.zarr', mode='r')
# root = zarr.open('/home/lqin/zarr_datasets/franka_ruckig_100hz.zarr', mode='r')
root = zarr.open('/home/lqin/zarr_datasets/straight_jntpath_finegrained_paras.zarr', mode='r')
# total_length = root['meta']['episode_ends'][-1]
# goal_conf = np.zeros((int(total_length), 7), dtype=np.float32)

# current_start = 0
# for end_idx in root['meta']['episode_ends']:
#     end_idx = int(end_idx) 
#     goal_conf[current_start:end_idx+1, :] = root['data']['jnt_pos'][end_idx]
#     current_start = end_idx + 1
# root['data'].create_dataset('goal_conf', data=goal_conf, chunks=True)

# jnt = np.load('jnt.npy')
# plt.figure(figsize=(10, 3 * robot_s.n_dof))
# for i in range(robot_s.n_dof):
#     plt.subplot(robot_s.n_dof, 1, i + 1)
#     plt.plot(jnt[0,:,i], label='Position')
#     print(f"joint {i}")
#     plt.ylabel(f'DoF {i}')
#     plt.legend()
#     plt.grid(True)
# plt.xlabel('Time [s]')
# plt.tight_layout()
# plt.show()

# for i in range(jnt.shape[1]-1):
#     # robot_s.goto_given_conf(jnt_values=jnt[0,i,:])
#     # robot_s.gen_meshmodel(alpha=0.2).attach_to(base)
#     s_pos, _ = robot_s.fk(jnt_values=jnt[0,i,:])
#     e_pos, _ = robot_s.fk(jnt_values=jnt[0,i+1,:])
#     mgm.gen_stick(spos=s_pos, epos=e_pos,radius=.0005, rgb=[0,0,0]).attach_to(base)
# base.run()

print(repr(np.min(root['data']['poly_coef'], axis=0)))
print(repr(np.max(root['data']['poly_coef'], axis=0)))
exit(0)

traj_id =1
traj_start = int(np.sum(root['meta']['episode_ends'][:traj_id]))
traj_end = int(np.sum(root['meta']['episode_ends'][:traj_id + 1]))
jnt_pos_list = root['data']['jnt_pos'][traj_start:traj_end]
jnt_vel_list = root['data']['jnt_vel'][traj_start:traj_end]
jnt_acc_list = root['data']['jnt_acc'][traj_start:traj_end]
goal_conf = root['data']['goal_conf'][traj_start:traj_end]
print('start:', repr(jnt_pos_list[0]))
# print('waypoint:', repr(jnt_pos_list[1000]))
print('goal:', repr(jnt_pos_list[-1]))

robot_s.goto_given_conf(jnt_values=jnt_pos_list[-1])
robot_s.gen_meshmodel(alpha=0.2, rgb=[0,1,0]).attach_to(base)
robot_s.goto_given_conf(jnt_values=jnt_pos_list[0])
robot_s.gen_meshmodel(alpha=0.2, rgb=[0,0,1]).attach_to(base)
for id in range(0, len(jnt_pos_list)-1):
    # robot_s.gen_meshmodel(alpha=0.2).attach_to(base)
    s_pos, _ = robot_s.fk(jnt_values=jnt_pos_list[id])
    e_pos, _ = robot_s.fk(jnt_values=jnt_pos_list[id+1])
    mgm.gen_stick(spos=s_pos, epos=e_pos, rgb=[0,0,0]).attach_to(base)
# plot_details(robot_s, jnt_pos_list, jnt_vel_list, jnt_acc_list)

# import numpy as np
# with open('jnt_info.npz', 'rb') as f:
#     data = np.load(f)
#     jnt_pos = data['jnt_pos']

# for id in range(0, len(jnt_pos_list)-1):
#     # robot_s.gen_meshmodel(alpha=0.2).attach_to(base)
#     s_pos, _ = robot_s.fk(jnt_values=jnt_pos_list[id])
#     e_pos, _ = robot_s.fk(jnt_values=jnt_pos_list[id+1])
#     mgm.gen_stick(spos=s_pos, epos=e_pos, rgb=[1,0,0]).attach_to(base)


base.run()
print('done')