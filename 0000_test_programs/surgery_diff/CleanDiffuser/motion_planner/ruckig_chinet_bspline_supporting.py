import numpy as np
import random
import copy
from tqdm import tqdm
import time

data = []
success_count = 0
attempt = 0

import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from wrs import wd, rm, mcm
robot_s = franka.FrankaResearch3(enable_cc=True)

# while success_count < 5000:
#     attempt += 1
#     gth_jnt_seed = robot_s.rand_conf()
#     pos, rot = robot_s.fk(jnt_values=gth_jnt_seed)
#     pos_start = copy.deepcopy(pos)
#     jnt_list = [gth_jnt_seed]

#     axis = random.choice(['x', 'y', 'z'])
#     axis_map = {'x': 0, 'y': 1, 'z': 2}
#     axis_idx = axis_map[axis]

#     for _ in range(200):
#         pos[axis_idx] += 0.01
#         jnt = robot_s.ik(tgt_pos=pos, tgt_rotmat=rot, seed_jnt_values=jnt_list[-1])
#         if jnt is not None:
#             jnt_list.append(jnt)
#         else:
#             break

#     if len(jnt_list) > 1:
#         pos_goal = robot_s.fk(jnt_values=jnt_list[-1])[0]
#         traj_len = (len(jnt_list) - 1) * 0.01  # 每一步0.01m
#         data.append({
#             "start_pos": pos_start,
#             "goal_pos": pos_goal,
#             "start_jnt": gth_jnt_seed,
#             "traj_len": traj_len
#         })
#         success_count += 1

#     if success_count % 100 == 0:
#         print(f"Success samples: {success_count} / 1000 (Total attempts: {attempt})")

# # 保存数据为 npy 文件
# np.save("ik_traj_dataset.npy", data)
# print(f"Finished. Collected {len(data)} successful samples.")

'''Load the dataset'''
data = np.load("ik_traj_dataset.npy", allow_pickle=True)
print(f"Loaded dataset with {len(data)} samples.")

start_positions = [item["start_pos"] for item in data]
goal_positions = [item["goal_pos"] for item in data]
start_joints = [item["start_jnt"] for item in data]
traj_lengths = [item["traj_len"] for item in data]
print(f'gth average length: {np.mean(traj_lengths):.4f} m')

'''random initial joint conf test'''
init_pos_diff = []
traj_len_list = []

success_count = 0
longer_traj_count = 0

ts = time.time()
for id in tqdm(range(len(data))):
# for id in tqdm(range(10)):
    seed = robot_s.rand_conf()
    # seed = np.zeros(robot_s.n_dof)
    pos_guess, rot = robot_s.fk(jnt_values=seed)

    gth_pos = data[id]['start_pos']
    real_jnt = robot_s.ik(tgt_pos=gth_pos, tgt_rotmat=rot)
    
    if real_jnt is not None:
        success_count += 1
        jnt_list = [real_jnt]

        '''record'''
        init_pos_diff.append(np.linalg.norm(pos_guess - data[id]['start_pos']))

        goal_pos = data[id]['goal_pos']
        disp = np.abs(goal_pos - gth_pos)
        axis_idx = np.argmax(disp)
        # print(f"ID: {id}, Start Position: {gth_pos}, Goal Position: {goal_pos}, Axis: {axis_idx}")

        for _ in range(200):
            gth_pos[axis_idx] += 0.01
            jnt = robot_s.ik(tgt_pos=gth_pos, tgt_rotmat=rot, seed_jnt_values=jnt_list[-1])
            if jnt is not None:
                jnt_list.append(jnt)
            else:
                break

        if len(jnt_list) > 1:
            pos_goal = robot_s.fk(jnt_values=jnt_list[-1])[0]
            traj_len = (len(jnt_list) - 1) * 0.01  # 每一步0.01m
            if traj_len > data[id]['traj_len']:
                longer_traj_count += 1
            traj_len_list.append(traj_len)
            # print(f"Start Position: {gth_pos}, Goal Position: {pos_goal}, Trajectory Length: {traj_len}")
time_cost = time.time() - ts

print(f"Time cost: {time_cost:.2f} seconds")
print(f'average time per sample: {time_cost/success_count:.4f} seconds')
print(f"Total successful samples: {success_count/len(data)*100:.2f}%")
print(f"Average initial position difference: {np.sum(init_pos_diff)/len(data):.4f}")
print(f"Average trajectory length: {np.sum(traj_len_list)/len(data):.4f}")
print(f"Longer trajectory count: {longer_traj_count/len(data)*100:.2f}%")