from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import numpy as np

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
nupdate = 1

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
for i in range(nupdate):
    robot.get_jnt_values()
    robot.jnt_ranges
    jnt_values = robot.rand_conf()
    tgt_pos, tgt_rotmat = robot.fk(jnt_values=jnt_values)
    result = robot.ik(tgt_pos, tgt_rotmat, best_sol_num=200)
    if result is not None:
        success_rate += 1

print(success_rate/nupdate)
