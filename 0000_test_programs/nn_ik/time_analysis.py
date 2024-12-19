import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
dataset = np.load('0000_test_programs/nn_ik/datasets/ik_time_label.npz')
labels, res_bool, target, jnt_result = dataset['label'], dataset['res_bool'], dataset['target'], dataset['jnt_result']
cut_limit = 800
tgt_pos, tgt_rot = target[:cut_limit, :3], target[:cut_limit, 3:]

# # tgt_pos = tgt_rot
# # Create a 3D scatter plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Plot each point based on its label
# for i in range(len(tgt_pos)):
#     if labels[i] == '1':
#         ax.scatter(tgt_pos[i, 0], tgt_pos[i, 1], tgt_pos[i, 2], color='blue', alpha=0.8)
#         # continue
#     elif labels[i] == '2':
#         ax.scatter(tgt_pos[i, 0], tgt_pos[i, 1], tgt_pos[i, 2], color='red', alpha=0.8)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

label_2_indices = np.where(labels == '2')[0]  # 获取所有 label = 2 的索引
res_bool_label_2 = res_bool[label_2_indices]  # 提取对应的 res_bool

# 计算 res_bool=False 的概率
prob_false_when_label_2 = np.sum(res_bool_label_2 == False) / len(res_bool_label_2) if len(res_bool_label_2) > 0 else 0

print(f"Probability of res_bool=False when label=2: {prob_false_when_label_2:.2f}")
