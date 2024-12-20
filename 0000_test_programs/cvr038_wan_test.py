from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import numpy as np
import wrs
import pickle

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
# mcm.mgm.gen_frame().attach_to(base)

robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
nupdate = 1000

'''original version'''
# while True:
#     success_rate = 0
#     time_list = []
#     success_list = []
#     for i in tqdm(range(nupdate)):
#         jnt_values = robot.rand_conf()
#         tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
#         tic = time.time()
#         result = robot.ik(tgt_pos, tgt_rotmat,best_sol_num = 1)
#         toc = time.time()
#         time_list.append(toc-tic)
#         if result is not None:
#             success_rate += 1
#             success_list.append(0.0001)
#         else:
#             success_list.append(-0.0001)
            

#     print(success_rate/nupdate)
#     plt.plot(range(nupdate), time_list)
#     plt.plot(range(nupdate), success_list)
#     for x in range(nupdate):
#                 plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
#     plt.legend(['time', 'success'])
#     plt.show()


'''best sol num iteration'''
# Lists to store results

# success_rates = []
# best_sol_range = range(1, 201)

# for best_sol_num in tqdm(best_sol_range):
#     success_rate = 0
#     for i in range(nupdate):
#         jnt_values = robot.rand_conf()
#         tgt_pos, tgt_rotmat = robot.fk(jnt_values=jnt_values)
#         result = robot.ik(tgt_pos, tgt_rotmat, best_sol_num=best_sol_num)
#         if result is not None:
#             success_rate += 1
    
#     # Calculate success rate for this best_sol_num
#     success_rates.append(success_rate / nupdate)

# with open("ddik_sort_minmax_rcond.txt", "w", encoding="utf-8") as f:
#     for item in success_rates:
#         f.write(str(item) + "\n")


# # Plot success rate as a function of best_sol_num
# plt.figure(figsize=(10, 6))
# plt.plot(best_sol_range, success_rates, label='Success Rate', color='blue', linewidth=2)
# plt.xlabel('Best Solution Number (best_sol_num)')
# plt.ylabel('Success Rate')
# plt.title('Success Rate vs. Best Solution Number')
# plt.grid(True)
# plt.legend()
# plt.show()

# success_rates_nosorted = np.loadtxt("ddik_nosort.txt", unpack=True)
# success_rates_sorted = np.loadtxt("ddik_sort.txt", unpack=True)
# success_rates_sorted_minmax = np.loadtxt("ddik_sort_minmax.txt", unpack=True)

# plt.figure(figsize=(10, 6))

# plt.plot(best_sol_range, success_rates_sorted, label='Sorted Data (l2 norm)', color='blue', linewidth=1.5)
# plt.plot(best_sol_range, success_rates_sorted_minmax, label='Sorted Data (MinMax)', color='green', linewidth=1.5)
# plt.plot(best_sol_range, success_rates_nosorted, label='Unsorted Data', color='red', linewidth=1.5)

# plt.xlabel('Best Solution Number (best_sol_num)')
# plt.ylabel('Success Rate')
# plt.grid(True)
# plt.legend()
# plt.show()

success_rate = 0
time_list = []
for i in range(nupdate):
    robot.get_jnt_values()
    robot.jnt_ranges
    jnt_values = robot.rand_conf()
    print(jnt_values)
    # jnt_values = np.array([-1.2935607032013121, 1.6634372002455124, 1.986319875862302, 0.8050965371297347, 1.4874555191511198, -0.6103529336325728])
    tgt_pos, tgt_rotmat = robot.fk(jnt_values=jnt_values)

    tic = time.time()
    result = robot.ik(tgt_pos, tgt_rotmat, best_sol_num = 5)
    toc = time.time()
    time_list.append(toc-tic)
    if result is not None:
        success_rate += 1

print(success_rate/nupdate)
plt.plot(range(nupdate), time_list)
plt.show()


'''visualize the best solutions'''
# # mcm.mgm.gen_frame().attach_to(base)
# arm = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
# with open('wrs/robot_sim/_data_files/cobotta_arm_jnt_data.pkl', 'rb') as f_jnt:
#     kdt_jnt_data = pickle.load(f_jnt)

# arm_mesh = arm.gen_meshmodel(alpha=.3)
# arm_mesh.attach_to(base)
# # base.run()
# # tmp_arm_stick = arm.gen_stickmodel(toggle_flange_frame=True)
# # tmp_arm_stick.attach_to(base)

# #  wrs.robot_sim._kinematics.ik_dd.DDIKSolver
# # jnt_values = np.array([-1.2935607032013121, 1.6634372002455124, 1.986319875862302, 0.8050965371297347, 1.4874555191511198, -0.6103529336325728])
# # nn_index = np.array([35957, 35837, 35117, 36697, 30197, 35237, 36577, 29477, 36338, 30698])

# jnt_values = np.array([-0.25096607, -0.94363903,  1.96662395, -1.59687914,  0.10637626,  0.34194314])
# nn_index = np.array([ 1222,  4523,  6862,  4403,   932,  4938,  4697,  3823,  3703, 10578])
# tgt_pos, tgt_rotmat = arm.fk(jnt_values=jnt_values)
# mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

# for idx in nn_index:
#     jnt_seed = kdt_jnt_data[idx]
#     arm.goto_given_conf(jnt_values=jnt_seed)
#     if idx == nn_index[1]:
#         arm_mesh = arm.gen_meshmodel(alpha=.2,rgb=[1,0,0])
#         arm_mesh.attach_to(base)
#     tmp_arm_stick = arm.gen_stickmodel(toggle_flange_frame=True)
#     tmp_arm_stick.attach_to(base)

# arm.goto_given_conf(jnt_values=jnt_values)
# arm_mesh = arm.gen_meshmodel(alpha=.2,rgb=[0,0,1])
# arm_mesh.attach_to(base)

# seed_pos, seed_rotmat = robot.fk(jnt_values=kdt_jnt_data[nn_index[0]])
# print('first seed pos:', seed_pos, '\n first seed seed_rotmat:\n', seed_rotmat)
# mcm.mgm.gen_frame(pos=seed_pos, rotmat=seed_rotmat).attach_to(base)

# seed_pos, seed_rotmat = robot.fk(jnt_values=kdt_jnt_data[nn_index[1]])
# print('second seed pos:', seed_pos, '\n first seed seed_rotmat:\n', seed_rotmat)
# mcm.mgm.gen_frame(pos=seed_pos, rotmat=seed_rotmat).attach_to(base)

# base.run()

