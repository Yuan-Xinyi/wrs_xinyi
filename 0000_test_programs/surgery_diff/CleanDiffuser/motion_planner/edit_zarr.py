import zarr

# 打开 Zarr 数据集
ruckig_root = zarr.open('/home/lqin/zarr_datasets/0616_curvelineIK.zarr', mode='r+')
# ruckig_root = zarr.open('/home/lqin/zarr_datasets/straight_line_joint_path.zarr', mode='r+')

# 获取原始 'episode_ends' 数据
episode_ends = ruckig_root['meta']['episode_ends'][:]
print('shape of episode_ends:', episode_ends.shape)

# 去重并保留顺序
unique_episode_ends = list(dict.fromkeys(episode_ends))

# 删除原 'episode_ends' 数据（可以直接覆盖或删除）
del ruckig_root['meta']['episode_ends']  # 删除原数据

# 创建新的 'episode_ends' 数据集，并将新的值写入
ruckig_root['meta']['episode_ends'] = unique_episode_ends

# 检查修改结果
print('shape of new episode_ends:', ruckig_root['meta']['episode_ends'][:].shape)
