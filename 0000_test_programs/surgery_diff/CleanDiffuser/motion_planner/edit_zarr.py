#===========================================================================================
'''去除 Zarr 数据集中 'episode_ends' 的重复值'''
# import zarr

# # 打开 Zarr 数据集
# ruckig_root = zarr.open('/home/lqin/zarr_datasets/0616_curvelineIK.zarr', mode='r+')
# # ruckig_root = zarr.open('/home/lqin/zarr_datasets/straight_line_joint_path.zarr', mode='r+')

# # 获取原始 'episode_ends' 数据
# episode_ends = ruckig_root['meta']['episode_ends'][:]
# print('shape of episode_ends:', episode_ends.shape)

# # 去重并保留顺序
# unique_episode_ends = list(dict.fromkeys(episode_ends))

# # 删除原 'episode_ends' 数据（可以直接覆盖或删除）
# del ruckig_root['meta']['episode_ends']  # 删除原数据

# # 创建新的 'episode_ends' 数据集，并将新的值写入
# ruckig_root['meta']['episode_ends'] = unique_episode_ends

# # 检查修改结果
# print('shape of new episode_ends:', ruckig_root['meta']['episode_ends'][:].shape)


#===========================================================================================
'''去除 Zarr 数据集中 'episode_ends' 的连续重复值，只保留每段连续序列的第一个值'''
import numpy as np
import zarr

# 打开 Zarr 数据集
ruckig_root = zarr.open('/home/lqin/zarr_datasets/fr3_mixed_trajs_clean.zarr', mode='r+')

# 读出原始 episode_ends
episode_ends = ruckig_root['meta']['episode_ends'][:]
print("原始 shape:", episode_ends.shape)

# 处理逻辑：保留每段连续序列的第一个
def keep_first_of_consecutive(arr):
    arr = np.array(arr)
    if len(arr) == 0:
        return arr
    mask = np.ones(len(arr), dtype=bool)
    mask[1:] &= (arr[1:] - arr[:-1] != 1)  # 当前值和前一个差 1 → 去掉当前
    return arr[mask]

new_episode_ends = keep_first_of_consecutive(episode_ends)

# 删除旧数据
del ruckig_root['meta']['episode_ends']

# 写入新数据（保持 dtype 一致）
ruckig_root['meta']['episode_ends'] = new_episode_ends.astype(episode_ends.dtype)

print("新 shape:", ruckig_root['meta']['episode_ends'][:].shape)

