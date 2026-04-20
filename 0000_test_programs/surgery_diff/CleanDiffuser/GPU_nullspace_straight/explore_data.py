#!/usr/bin/env python3
"""
探索 franka_research_3_gpu_trajectories_sub10_long.hdf5 数据集的分布
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
DATA_PATH = Path(__file__).parent / 'datasets' / 'franka_research_3_gpu_trajectories_sub10_long.hdf5'

def explore_dataset():
    """探索数据集的结构和分布"""
    
    if not DATA_PATH.exists():
        print(f"❌ 数据文件不存在: {DATA_PATH}")
        return
    
    print(f"📂 打开数据文件: {DATA_PATH}")
    print("=" * 80)
    
    with h5py.File(DATA_PATH, 'r') as f:
        # 1. 数据集结构
        print("\n📋 数据集结构:")
        print(f"  顶级键: {list(f.keys())}")
        
        if 'trajectories' not in f:
            print("❌ 找不到 'trajectories' 组")
            return
        
        traj_group = f['trajectories']
        traj_names = sorted(traj_group.keys())
        
        print(f"\n  轨迹数量: {len(traj_names)}")
        print(f"  轨迹名称 (前5个): {traj_names[:5]}")
        
        # 2. 第一条轨迹的详细信息
        first_traj = traj_group[traj_names[0]]
        print(f"\n📊 第一条轨迹的结构 ({traj_names[0]}):")
        
        for key in first_traj.keys():
            data = first_traj[key]
            print(f"  {key:<20} shape={str(data.shape):<20} dtype={data.dtype}")
        
        print(f"\n  属性 (attrs):")
        for key, val in first_traj.attrs.items():
            if isinstance(val, np.ndarray):
                print(f"    {key:<20} = {val} (shape: {val.shape})")
            else:
                print(f"    {key:<20} = {val}")
        
        # 3. 统计所有轨迹的信息
        print("\n" + "=" * 80)
        print("📈 数据集统计信息:")
        
        total_points = 0
        total_points_filtered = 0
        q_data = []
        pos_data = []
        remaining_length_data = []
        traj_lengths = []
        
        for traj_idx, traj_name in enumerate(traj_names):
            traj = traj_group[traj_name]
            
            num_points = traj.attrs['num_points']
            traj_lengths.append(num_points)
            total_points += num_points
            
            # 收集原始数据
            q_raw = np.asarray(traj['q'][:], dtype=np.float32)
            pos_raw = np.asarray(traj['tcp_pos'][:], dtype=np.float32)
            rl_raw = np.asarray(traj['remaining_length'][:], dtype=np.float32)
            
            # 过滤: only keep remaining_length > 0.75
            mask = rl_raw > 0.75
            total_points_filtered += mask.sum()
            
            q_data.append(q_raw[mask])
            pos_data.append(pos_raw[mask])
            remaining_length_data.append(rl_raw[mask])
        
            if (traj_idx + 1) % 100 == 0 or traj_idx + 1 == len(traj_names):
                print(f"  处理进度: {traj_idx + 1}/{len(traj_names)}")
        
        print(f"\n总轨迹数: {len(traj_names)}")
        print(f"总点数: {total_points}")
        print(f"过滤后 (remaining_length > 0.75) 的点数: {total_points_filtered}")
        print(f"过滤保留比例: {100*total_points_filtered/total_points:.1f}%")
        
        # 轨迹长度统计
        traj_lengths = np.array(traj_lengths)
        print(f"\n轨迹长度统计:")
        print(f"  平均: {traj_lengths.mean():.1f}")
        print(f"  最小: {traj_lengths.min()}")
        print(f"  最大: {traj_lengths.max()}")
        print(f"  中位数: {np.median(traj_lengths):.1f}")
        
        # 4. 关节角统计
        q_data = np.vstack(q_data)
        print(f"\n关节角 (q) 统计 [过滤后 remaining_length > 0.75] (shape: {q_data.shape}):")
        print(f"  均值: {q_data.mean(axis=0)}")
        print(f"  标准差: {q_data.std(axis=0)}")
        print(f"  最小值: {q_data.min(axis=0)}")
        print(f"  最大值: {q_data.max(axis=0)}")
        
        # 5. 位置统计
        pos_data = np.vstack(pos_data)
        print(f"\nTCP位置 (pos) 统计 [过滤后 remaining_length > 0.75] (shape: {pos_data.shape}):")
        print(f"  均值: {pos_data.mean(axis=0)}")
        print(f"  标准差: {pos_data.std(axis=0)}")
        print(f"  最小值: {pos_data.min(axis=0)}")
        print(f"  最大值: {pos_data.max(axis=0)}")
        
        # 6. 剩余长度统计
        remaining_length_data = np.hstack(remaining_length_data)
        print(f"\n剩余长度 (remaining_length) 统计 [已过滤 > 0.75]:")
        print(f"  形状: {remaining_length_data.shape}")
        print(f"  均值: {remaining_length_data.mean():.4f}")
        print(f"  标准差: {remaining_length_data.std():.4f}")
        print(f"  最小值: {remaining_length_data.min():.4f}")
        print(f"  最大值: {remaining_length_data.max():.4f}")
        print(f"  中位数: {np.median(remaining_length_data):.4f}")
        
        # 7. 绘制可视化
        print("\n📊 生成可视化图表...")
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'Franka Trajectories Data Distribution\n(Filtered: remaining_length > 0.75, N={len(remaining_length_data)})', 
                     fontsize=16, fontweight='bold')
        
        # 轨迹长度分布
        axes[0, 0].hist(traj_lengths, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Trajectory Length Distribution (All Data)')
        axes[0, 0].set_xlabel('Points per Trajectory')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 关节角分布 (第一个关节) - 过滤后
        axes[0, 1].hist(q_data[:, 0], bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_title('Joint Angles Distribution (q[0])\n(Filtered > 0.75)')
        axes[0, 1].set_xlabel('Angle (rad)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # TCP位置分布 (X方向) - 过滤后
        axes[0, 2].hist(pos_data[:, 0], bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[0, 2].set_title('TCP Position Distribution (X)\n(Filtered > 0.75)')
        axes[0, 2].set_xlabel('Position (m)')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 剩余长度分布 (已过滤)
        axes[1, 0].hist(remaining_length_data, bins=50, edgecolor='black', alpha=0.7, color='red')
        axes[1, 0].set_title(f'Remaining Length Distribution\n(Filtered > 0.75, N={len(remaining_length_data)})')
        axes[1, 0].set_xlabel('Remaining Length')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # TCP位置3D scatter (前1000个点)
        sample_idx = np.random.choice(len(pos_data), min(1000, len(pos_data)), replace=False)
        scatter = axes[1, 1].scatter(pos_data[sample_idx, 0], pos_data[sample_idx, 1], 
                          alpha=0.3, s=10, c=remaining_length_data[sample_idx], cmap='viridis')
        axes[1, 1].set_title('TCP Position (X-Y plane, sample)\n(Filtered > 0.75)')
        axes[1, 1].set_xlabel('X (m)')
        axes[1, 1].set_ylabel('Y (m)')
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Remaining Length')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 关节角均值和标准差 - 过滤后
        q_mean = q_data.mean(axis=0)
        q_std = q_data.std(axis=0)
        axes[1, 2].bar(range(len(q_mean)), q_mean, alpha=0.7, label='Mean')
        axes[1, 2].bar(range(len(q_std)), q_std, alpha=0.5, label='Std')
        axes[1, 2].set_title('Joint Angles Statistics\n(Filtered > 0.75)')
        axes[1, 2].set_xlabel('Joint Index')
        axes[1, 2].set_ylabel('Value (rad)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        output_path = Path(__file__).parent / 'data_distribution_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存: {output_path}")
        
        # 显示第一个轨迹的样本
        print("\n" + "=" * 80)
        print("📌 第一条轨迹的样本数据:")
        first_traj_data = traj_group[traj_names[0]]
        q_sample = first_traj_data['q'][:5]
        pos_sample = first_traj_data['tcp_pos'][:5]
        rl_sample = first_traj_data['remaining_length'][:5]
        
        print(f"\nq (前5个时间步):")
        for i, q in enumerate(q_sample):
            print(f"  t={i}: {q}")
        
        print(f"\ntcp_pos (前5个时间步):")
        for i, pos in enumerate(pos_sample):
            print(f"  t={i}: {pos}")
        
        print(f"\nremaining_length (前5个时间步):")
        for i, rl in enumerate(rl_sample):
            print(f"  t={i}: {rl:.4f}")
        
        print("\n" + "=" * 80)
        print("✓ 数据探索完成！")


if __name__ == '__main__':
    explore_dataset()
