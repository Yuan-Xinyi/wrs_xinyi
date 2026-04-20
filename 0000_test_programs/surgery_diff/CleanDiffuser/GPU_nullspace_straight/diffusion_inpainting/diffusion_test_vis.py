#!/usr/bin/env python3
"""
测试和可视化训练好的Inpainting扩散模型
支持：
- 加载训练好的模型
- 从数据集中抽取测试样本
- 生成条件下的关节角预测
- 可视化对比结果
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from diffusion import (
    FeatureLayout,
    normalize_condition,
    denormalize_q,
    sample_q_from_condition,
    create_model,
    infer_layout_from_h5,
)


def load_bundle(bundle_path: Path, device: torch.device):
    """加载训练好的模型和参数"""
    print(f"📂 加载模型包: {bundle_path}")
    
    payload = torch.load(bundle_path, map_location=device, weights_only=False)
    
    stats = payload['stats']
    x_min = payload['x_min']
    x_max = payload['x_max']
    args_dict = payload['args']
    metadata = payload['metadata']
    
    print(f"✓ 模型信息:")
    print(f"  q_dim: {stats['q_dim']}")
    print(f"  token_dim: {stats['token_dim']}")
    print(f"  数据集: {metadata['h5_path']}")
    print(f"  训练样本数: {metadata['train_entries']}")
    
    return stats, x_min, x_max, args_dict, metadata


def sample_from_dataset(h5_path: Path, layout: FeatureLayout, n_samples: int = 5, seed: int = 42):
    """从数据集中随机抽取测试样本"""
    np.random.seed(seed)
    
    print(f"\n📂 从数据集加载样本: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        traj_names = sorted(f['trajectories'].keys())
        
        # 随机选择轨迹和时间步
        test_samples = []
        for _ in range(n_samples):
            traj_idx = np.random.randint(0, len(traj_names))
            traj_name = traj_names[traj_idx]
            traj = f['trajectories'][traj_name]
            
            # 过滤: only remaining_length > 0.75
            rl = np.asarray(traj['remaining_length'][:], dtype=np.float32)
            valid_indices = np.where(rl > 0.75)[0]
            
            if len(valid_indices) == 0:
                continue
            
            time_idx = np.random.choice(valid_indices)
            
            q_true = np.asarray(traj['q'][time_idx], dtype=np.float32)
            pos = np.asarray(traj['tcp_pos'][time_idx], dtype=np.float32)
            direction = np.asarray(traj.attrs['direction'], dtype=np.float32)
            target_normal = np.asarray(traj.attrs['target_normal'], dtype=np.float32)
            rl_true = rl[time_idx]
            
            test_samples.append({
                'q_true': q_true,
                'pos': pos,
                'direction': direction,
                'normal': target_normal,
                'remaining_length': rl_true,
                'traj_name': traj_name,
                'time_idx': time_idx,
            })
    
    print(f"✓ 加载了 {len(test_samples)} 个测试样本")
    return test_samples


def generate_predictions(model, stats: dict, test_samples: list, device: torch.device, 
                         q_dim: int, n_prediction_samples: int = 8, sample_steps: int = 32):
    """对测试样本进行预测"""
    print(f"\n🎯 生成预测 (每个样本生成 {n_prediction_samples} 个预测)...")
    
    predictions = []
    
    for idx, sample in enumerate(test_samples):
        condition = np.concatenate([
            sample['pos'],
            sample['direction'],
            sample['normal']
        ]).astype(np.float32)
        
        q_pred = sample_q_from_condition(
            model=model,
            stats=stats,
            condition=condition,
            device=device,
            q_dim=q_dim,
            n_samples=n_prediction_samples,
            sample_steps=sample_steps,
            temperature=1.0,
        )
        
        predictions.append({
            **sample,
            'q_pred': q_pred,  # shape: (n_prediction_samples, q_dim)
        })
        
        print(f"  [{idx+1}/{len(test_samples)}] ✓")
    
    return predictions


def visualize_results(predictions: list, output_path: Path, q_dim: int = 7):
    """可视化预测结果"""
    print(f"\n📊 生成可视化...")
    
    n_samples = len(predictions)
    
    # 创建大图表
    fig = plt.figure(figsize=(16, 4 * n_samples))
    
    for sample_idx, pred in enumerate(predictions):
        q_true = pred['q_true']
        q_preds = pred['q_pred']  # shape: (n_prediction_samples, q_dim)
        
        # 为每个关节创建一个subplot
        for joint_idx in range(q_dim):
            ax = plt.subplot(n_samples, q_dim, sample_idx * q_dim + joint_idx + 1)
            
            # 绘制真值
            ax.axhline(y=q_true[joint_idx], color='red', linewidth=2, label='Ground Truth', linestyle='--')
            
            # 绘制所有预测
            for pred_idx, q_pred_sample in enumerate(q_preds):
                alpha = 0.3 if pred_idx > 0 else 0.7
                color = 'blue' if pred_idx == 0 else 'cyan'
                ax.scatter([joint_idx], [q_pred_sample[joint_idx]], 
                          alpha=alpha, s=100, color=color)
            
            # 计算预测的均值和方差
            q_mean = q_preds.mean(axis=0)
            q_std = q_preds.std(axis=0)
            
            ax.errorbar([joint_idx], [q_mean[joint_idx]], 
                       yerr=[q_std[joint_idx]], fmt='o', 
                       color='green', ecolor='green', capsize=5, 
                       linewidth=2, markersize=8, label='Mean±Std')
            
            ax.set_xlim(-0.5, q_dim - 0.5)
            ax.set_ylabel(f'q{joint_idx+1} (rad)')
            ax.set_xlabel('Joint')
            ax.grid(True, alpha=0.3)
            
            if sample_idx == 0 and joint_idx == 0:
                ax.legend(loc='upper right', fontsize=8)
            
            # 标题
            if joint_idx == 0:
                ax.set_title(f'Sample {sample_idx+1}: {pred["traj_name"]} (t={pred["time_idx"]})')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_path}")
    plt.close()


def generate_statistics_table(predictions: list, q_dim: int = 7):
    """生成统计表"""
    print(f"\n" + "=" * 100)
    print("[ 预测结果统计 ]")
    print("=" * 100)
    
    for sample_idx, pred in enumerate(predictions):
        print(f"\n📍 样本 {sample_idx+1}: {pred['traj_name']} (t={pred['time_idx']})")
        print(f"   Remaining Length: {pred['remaining_length']:.4f}")
        print(f"   Position: {pred['pos']}")
        print(f"   Direction: {pred['direction']}")
        print(f"   Target Normal: {pred['normal']}")
        
        q_true = pred['q_true']
        q_preds = pred['q_pred']  # shape: (n_prediction_samples, q_dim)
        
        q_mean = q_preds.mean(axis=0)
        q_std = q_preds.std(axis=0)
        
        print(f"\n   关节角对比 (Ground Truth vs Predicted Mean±Std):")
        print(f"   {'Joint':<10} {'True':<12} {'Pred Mean':<12} {'Std':<12} {'Error':<12}")
        print(f"   {'-'*58}")
        
        errors = []
        for j in range(q_dim):
            error = abs(q_true[j] - q_mean[j])
            errors.append(error)
            print(f"   q{j+1:<9} {q_true[j]:>11.4f} {q_mean[j]:>11.4f} {q_std[j]:>11.4f} {error:>11.4f}")
        
        print(f"   平均误差: {np.mean(errors):.4f}")
        print(f"   最大误差: {np.max(errors):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Test and visualize trained inpainting diffusion model')
    parser.add_argument('--bundle', type=Path, 
                       default=Path(__file__).parent.parent / 'runs' / 'dit_kinematic_inpainting_runs' / 'ddpm32_dit_inpaint_q_only_long' / 'bundle_latest.pt',
                       help='Path to model bundle')
    parser.add_argument('--h5-path', type=Path, 
                       default=Path(__file__).parent.parent / 'datasets' / 'franka_research_3_gpu_trajectories_sub10_long.hdf5',
                       help='Path to test dataset HDF5')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n-test-samples', type=int, default=3, help='Number of test samples')
    parser.add_argument('--n-predictions', type=int, default=8, help='Number of predictions per sample')
    parser.add_argument('--sample-steps', type=int, default=32, help='Number of diffusion steps for sampling')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=None, help='Output visualization path')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = Path(__file__).parent / 'test_visualization.png'
    
    device = torch.device(args.device)
    
    # 检查文件
    if not args.bundle.exists():
        print(f"❌ 模型文件不存在: {args.bundle}")
        return
    
    if not args.h5_path.exists():
        print(f"❌ 数据集文件不存在: {args.h5_path}")
        return
    
    print("=" * 100)
    print("[ 训练模型测试和可视化 ]")
    print("=" * 100)
    
    # 加载模型
    stats, x_min, x_max, args_dict, metadata = load_bundle(args.bundle, device)
    
    # 推导layout
    layout = infer_layout_from_h5(args.h5_path)
    q_dim = layout.q_dim
    
    # 创建模型
    model = create_model(
        device=device,
        x_min=x_min,
        x_max=x_max,
        diffusion_steps=args_dict['diffusion_steps'],
        q_dim=q_dim
    )
    
    # 加载模型权重
    model_path = args.bundle.parent / 'model_latest.pt'
    if model_path.exists():
        print(f"\n📂 加载模型权重: {model_path}")
        model.load(str(model_path))
    
    model.eval()
    
    # 从数据集加载测试样本
    test_samples = sample_from_dataset(args.h5_path, layout, n_samples=args.n_test_samples, seed=args.seed)
    
    if len(test_samples) == 0:
        print("❌ 没有可用的测试样本")
        return
    
    # 生成预测
    predictions = generate_predictions(
        model, stats, test_samples, device, q_dim,
        n_prediction_samples=args.n_predictions,
        sample_steps=args.sample_steps
    )
    
    # 生成统计表
    generate_statistics_table(predictions, q_dim)
    
    # 可视化结果
    visualize_results(predictions, args.output, q_dim)
    
    print("\n" + "=" * 100)
    print("✓ 测试完成！")
    print("=" * 100)


if __name__ == '__main__':
    main()
