#!/usr/bin/env python3
"""
快速验证3D可视化脚本的核心逻辑（不显示GUI）
"""

import sys
from pathlib import Path
import numpy as np
import torch

# 添加wrs路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from diffusion import (
    FeatureLayout,
    sample_q_from_condition,
    create_model,
    infer_layout_from_h5,
)

try:
    from wrs.robot_sim.robots.franka_research_3 import FrankaResearch3
    print("✓ FrankaResearch3导入成功")
except ImportError as e:
    print(f"❌ FrankaResearch3导入失败: {e}")
    sys.exit(1)

# 加载模型
bundle_path = Path(__file__).parent.parent / 'runs' / 'dit_kinematic_inpainting_runs' / 'ddpm32_dit_inpaint_q_only_long' / 'bundle_latest.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"📂 加载模型: {bundle_path}")
payload = torch.load(bundle_path, map_location=device, weights_only=False)
stats = payload['stats']
x_min = payload['x_min']
x_max = payload['x_max']
args_dict = payload['args']

print(f"✓ 模型加载成功 (q_dim={stats['q_dim']})")

# 创建和加载模型
model = create_model(
    device=device,
    x_min=x_min,
    x_max=x_max,
    diffusion_steps=args_dict['diffusion_steps'],
    q_dim=stats['q_dim']
)
model_path = bundle_path.parent / 'model_latest.pt'
if model_path.exists():
    model.load(str(model_path))
    print(f"✓ 模型权重加载成功")
model.eval()

# 生成一个随机条件和预测
q_dim = stats['q_dim']
condition = np.random.randn(9).astype(np.float32)

print(f"\n🎯 生成预测...")
q_pred = sample_q_from_condition(
    model=model,
    stats=stats,
    condition=condition,
    device=device,
    q_dim=q_dim,
    n_samples=4,
    sample_steps=16,
    temperature=1.0,
)
print(f"✓ 预测成功，形状: {q_pred.shape}")

# 测试FrankaResearch3
print(f"\n🤖 测试Franka机器人...")
robot = FrankaResearch3()
print(f"✓ Franka机器人创建成功")

# 尝试meshmodel生成
try:
    meshmodel = robot.gen_meshmodel(rgb=[0, 0.8, 0], alpha=0.7)
    print(f"✓ Meshmodel生成成功")
except Exception as e:
    print(f"⚠️  Meshmodel生成出错: {e}")

print(f"\n✅ 核心功能验证完成!")
print(f"💡 提示: 运行 'python diffusion_eval_3d_vis.py' 来启动3D可视化")
