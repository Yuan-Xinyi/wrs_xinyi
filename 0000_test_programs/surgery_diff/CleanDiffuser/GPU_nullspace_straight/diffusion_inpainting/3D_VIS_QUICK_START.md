# 3D可视化脚本 - 快速指南

## 📋 文件清单

| 文件 | 说明 | 用途 |
|------|------|------|
| **diffusion_eval_3d_vis.py** | 主可视化脚本 | 3D机器人可视化 + 误差分析 |
| **README_3D_VIS.md** | 详细文档 | 完整使用说明 |
| **run_3d_vis.sh** | 快速启动脚本 | Shell启动器 |
| **test_3d_vis_core.py** | 核心功能测试 | 验证集成状态 |

## 🚀 快速开始

### 最简单的方式

```bash
cd /path/to/diffusion_inpainting

# 方法1: 直接运行Python脚本
python diffusion_eval_3d_vis.py

# 方法2: 使用Shell脚本
bash run_3d_vis.sh
```

### 指定参数

```bash
# 使用最优模型，生成5个样本，每个16个预测
python diffusion_eval_3d_vis.py \
  --bundle runs/dit_kinematic_inpainting_runs/ddpm32_dit_inpaint_q_only_long/bundle_best.pt \
  --n-test-samples 5 \
  --n-predictions 16

# 使用Shell包装器
bash run_3d_vis.sh 5 16 32
```

## ✨ 新功能说明

### 为什么显示所有预测而不仅仅是平均值？

**旧方式**（仅显示平均预测）：
```
- 只看到2个机器人：绿色(真实) + 红色(平均预测)
- 丢失了预测的多样性信息
- 无法看到预测的不确定性范围
```

**新方式**（显示所有预测）：
```
- 看到N+1个机器人：绿色(真实) + N个彩色(所有预测)
- 直观看到预测的分布和多样性
- 置信度高低一目了然：
  * 预测都聚在一起 → 模型很确定 ✓
  * 预测分散开来 → 需要调查不确定性原因
```

### FK批量计算（fk_batch）

虽然当前使用的是逐个FK计算，但结构支持未来升级为批量计算：
- 准备所有配置：`all_configs = np.vstack([q_true, q_preds])`
- 批量传输给GPU更高效
- 减少重复的初始化开销

## ✨ 主要功能

### 🎨 可视化特性

```
┌─────────────────────────────────────┐
│      3D Franka机器人可视化窗口      │
├─────────────────────────────────────┤
│                                     │
│    🟢 绿色机器人 (α=0.9)            │
│    (真实关节配置)                   │
│         ╱╲                          │
│        ╱  ╲    🔴 红色机器人        │
│            ╲  (预测样本1)           │
│             🟠 橙色机器人            │
│             (预测样本2)             │
│              🟡 黄色机器人           │
│              (预测样本N)            │
│                                     │
│    所有颜色机器人并排显示            │
│    透明度递减表示预测顺序            │
│    灰色坐标系作为参考               │
│                                     │
└─────────────────────────────────────┘

控制: 左击旋转 | 滚轮缩放 | 右击平移

💡 解读方式:
   - 如果所有预测叠在一起 → 预测很确定 ✓
   - 如果预测分散开来 → 不确定性高，需要检查
   - 与绿色的距离 → 系统偏差大小
```

### 📊 误差分析

```
关节角对比 (Ground Truth vs Predicted Mean±Std):
┌─────────┬─────────┬─────────┬────────┬────────┐
│ Joint   │ True    │ Mean    │ Std    │ Error  │
├─────────┼─────────┼─────────┼────────┼────────┤
│ q1      │ -1.3228 │ -1.2884 │ 0.0399 │ 0.0344 │
│ q2      │ -0.7982 │ -0.8248 │ 0.0112 │ 0.0266 │
│ q3      │ -0.1450 │ -0.1910 │ 0.0332 │ 0.0461 │
│ ...     │  ...    │  ...    │  ...   │  ...   │
└─────────┴─────────┴─────────┴────────┴────────┘

📊 统计:
  平均误差: 0.0270 rad (1.55°)  ← 关键指标
  最大误差: 0.0461 rad (2.64°)
  预测不确定性: 0.0253 rad      ← 置信度
```

## 📈 性能指标

### 主要指标说明

| 指标 | 范围 | 评价 |
|------|------|------|
| 平均误差 | < 0.05 rad | ⭐⭐⭐⭐⭐ 优秀 |
| | 0.05~0.1 rad | ⭐⭐⭐⭐ 良好 |
| | > 0.1 rad | ⭐⭐⭐ 需改进 |
| 预测Std | < 0.02 rad | ⭐⭐⭐⭐⭐ 高置信 |
| | 0.02~0.05 rad | ⭐⭐⭐⭐ 合理 |
| | > 0.05 rad | ⭐⭐⭐ 低置信 |

### 解读3D可视化

**预测聚集度（一致性）**：
```
高聚集 ✓                    低聚集 ⚠️
所有红→黄机器人            预测分散开来
堆在一起                     分布很广

→ 意味着：               → 意味着：
模型很确定                 模型不确定
预测稳定                   需要调查
重复实验结果一致           可能需要更多训练数据
```

**与真实的偏差（系统偏差）**：
```
预测聚集在真实值附近 ✓   预测聚集但远离真实值 ⚠️
（绿色和红色堆一起）      （红色聚集但离绿色远）

→ 意味着：                  → 意味着：
模型既准确又确定           模型确定但有系统偏差
理想情况                    需要调整学习率或数据
                          可能需要bias correction
```

## 🔧 常见任务

### 任务1：快速验证模型

```bash
# 用默认参数测试 (2个样本, 快速)
python diffusion_eval_3d_vis.py
```

**预期输出**：控制台打印误差统计和3D窗口显示

### 任务2：全面评估

```bash
# 测试更多样本，更多预测
python diffusion_eval_3d_vis.py \
  --n-test-samples 10 \
  --n-predictions 16 \
  --sample-steps 64
```

### 任务3：对比两个模型

```bash
# 模型A
echo "=== 模型 A ===" 
python diffusion_eval_3d_vis.py \
  --bundle runs/.../modelA/bundle_best.pt \
  --n-test-samples 3

# 模型B
echo "=== 模型 B ==="
python diffusion_eval_3d_vis.py \
  --bundle runs/.../modelB/bundle_best.pt \
  --n-test-samples 3
```

### 任务4：生成报告数据

```bash
# 输出结果到文件
python diffusion_eval_3d_vis.py \
  --n-test-samples 20 \
  --n-predictions 8 \
  2>&1 | tee evaluation_report.txt
```

## 🐛 调试技巧

### 检查模型是否加载成功

```bash
python -c "
import torch
from pathlib import Path

bundle = Path('runs/.../bundle_latest.pt')
if bundle.exists():
    data = torch.load(bundle, weights_only=False)
    print(f'✓ 模型找到')
    print(f'  q_dim: {data[\"stats\"][\"q_dim\"]}')
else:
    print('❌ 模型不存在')
"
```

### 测试核心功能

```bash
python test_3d_vis_core.py
```

### 检查WRS集成

```bash
python -c "
import sys
sys.path.insert(0, '/home/lqin/wrs_xinyi')
from wrs.robot_sim.robots.franka_research_3 import FrankaResearch3
print('✓ WRS and Franka available')
" 2>&1
```

## 💡 参数调优建议

### 对于快速测试
```bash
python diffusion_eval_3d_vis.py \
  --n-test-samples 2 \
  --n-predictions 4 \
  --sample-steps 16
```
**耗时**：~30秒 | **质量**：基础

### 对于标准评估
```bash
python diffusion_eval_3d_vis.py \
  --n-test-samples 5 \
  --n-predictions 8 \
  --sample-steps 32
```
**耗时**：~2分钟 | **质量**：推荐

### 对于深度分析
```bash
python diffusion_eval_3d_vis.py \
  --n-test-samples 10 \
  --n-predictions 16 \
  --sample-steps 64
  ```
**耗时**：~10分钟 | **质量**：高质量

## 📚 相关文档

- [详细使用指南](README_3D_VIS.md) - 所有参数和功能的完整说明
- [测试脚本说明](diffusion_test_vis.py) - 2D可视化脚本
- [训练指南](../README.md) - 模型训练方法

## ✅ 检查清单

在运行3D可视化前，确保：

- [ ] 已完成模型训练
- [ ] 模型文件存在: `runs/dit_kinematic_inpainting_runs/ddpm32_dit_inpaint_q_only_long/bundle_latest.pt`
- [ ] 数据集存在: `datasets/franka_research_3_gpu_trajectories_sub10_long.hdf5`
- [ ] WRS和Panda3D已安装 (`pip install panda3d`)
- [ ] PyTorch可用 (验证: `python -c "import torch; print(torch.cuda.is_available())"`)

## 🎯 典型工作流

```
1. 训练模型
   └─→ python diffusion_train.py --epochs 50

2. 快速验证
   └─→ python diffusion_eval_3d_vis.py

3. 全面评估
   └─→ python diffusion_eval_3d_vis.py \
        --n-test-samples 10 \
        --n-predictions 16

4. 对比模型
   └─→ 使用不同的 --bundle 参数多次运行

5. 生成报告
   └─→ 复制console输出到文档中
```

## 🔗 快速链接

```bash
# 快速启动可视化
cd /home/lqin/wrs_xinyi/0000_test_programs/surgery_diff/CleanDiffuser/GPU_nullspace_straight/diffusion_inpainting
python diffusion_eval_3d_vis.py
```

---
**更新时间**: 2026-04-20  
**版本**: 1.0  
**状态**: ✓ 核心功能验证完毕，可以使用
