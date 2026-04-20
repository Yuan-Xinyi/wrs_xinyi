# 训练模型测试和可视化脚本

## 脚本说明

`diffusion_test_vis.py` 是用来测试和可视化训练好的条件关节角Inpainting扩散模型的脚本。

## 功能特性

✅ **加载已训练的模型**
- 自动从 `bundle_latest.pt` 或 `bundle_best.pt` 加载模型参数
- 恢复训练时保存的所有统计信息（均值、标准差等）

✅ **从数据集中随机抽取测试样本**
- 自动过滤 `remaining_length > 0.75` 的高质量数据
- 支持自定义采样数量

✅ **生成多个随机预测**
- 对每个测试条件生成多个预测样本
- 计算预测的均值和不确定性（标准差）

✅ **详细的统计表格**
- 逐个样本对比真值 vs 预测均值
- 计算关节角误差
- 显示平均误差和最大误差

✅ **可视化对比**
- 真值 vs 预测分布可视化
- 展示不确定性范围
- 支持保存为高质量PNG图表

## 使用方法

### 基础用法

```bash
python diffusion_test_vis.py
```

### 自定义参数

```bash
# 指定模型路径
python diffusion_test_vis.py \
  --bundle /path/to/bundle_best.pt

# 增加测试样本数
python diffusion_test_vis.py \
  --n-test-samples 5

# 每个样本生成更多预测
python diffusion_test_vis.py \
  --n-predictions 16

# 增加采样步数（通常会提高质量但耗时更长）
python diffusion_test_vis.py \
  --sample-steps 64

# 指定输出路径
python diffusion_test_vis.py \
  --output /path/to/output.png
```

### 完整示例

```bash
python diffusion_test_vis.py \
  --bundle runs/ddpm32_dit_inpaint_q_only_long/bundle_best.pt \
  --h5-path datasets/franka_research_3_gpu_trajectories_sub10_long.hdf5 \
  --n-test-samples 4 \
  --n-predictions 8 \
  --sample-steps 32 \
  --device cuda \
  --seed 42 \
  --output test_results.png
```

## 输入参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--bundle` | Path | `runs/.../bundle_latest.pt` | 已训练模型的bundle文件 |
| `--h5-path` | Path | `datasets/.../long.hdf5` | 测试数据集HDF5文件 |
| `--device` | str | `cuda` (if available) | 计算设备 (cuda/cpu) |
| `--n-test-samples` | int | 3 | 从数据集中随机抽取多少个测试样本 |
| `--n-predictions` | int | 8 | 对每个样本生成多少个预测 |
| `--sample-steps` | int | 32 | DDPM采样的扩散步数 |
| `--seed` | int | 42 | 随机种子 |
| `--output` | Path | `test_visualization.png` | 输出可视化图表路径 |

## 输出信息

### 1. 控制台输出

```
=========================================================================================
[ 训练模型测试和可视化 ]
=========================================================================================

📂 加载模型包: ...
✓ 模型信息:
  q_dim: 7
  token_dim: 16
  数据集: ...
  训练样本数: ...

📂 从数据集加载样本: ...
✓ 加载了 3 个测试样本

🎯 生成预测 (每个样本生成 8 个预测)...
  [1/3] ✓
  [2/3] ✓
  [3/3] ✓

============================================================================================
[ 预测结果统计 ]
============================================================================================

📍 样本 1: traj_XXXXX (t=42)
   Remaining Length: 0.9245
   Position: [0.123 0.456 0.789]
   Direction: [0.1 0.2 0.3]
   Target Normal: [0.5 0.6 0.7]

   关节角对比 (Ground Truth vs Predicted Mean±Std):
   Joint      True         Pred Mean    Std          Error
   ----------------------------------------------------------
   q1         -1.4698       -1.4523       0.0145      0.0175
   q2         -0.8011       -0.7856       0.0123      0.0155
   ...
   平均误差: 0.0142
   最大误差: 0.0298

📊 生成可视化...
✓ 图表已保存: test_visualization.png

============================================================================================
✓ 测试完成！
============================================================================================
```

### 2. 可视化输出

生成的PNG图表展示：
- **每行代表一个测试样本**
- **每列代表一个关节**
- **红色虚线**: 真实的关节角值
- **蓝色点**: 单个预测样本
- **绿色点+误差棒**: 预测的均值±标准差

## 典型工作流

1. **训练模型**
   ```bash
   python diffusion_train.py \
     --h5-path datasets/franka_research_3_gpu_trajectories_sub10_long.hdf5 \
     --epochs 100
   ```

2. **生成模型束**（训练期间自动保存到 `runs/*/bundle_latest.pt`）

3. **测试模型**
   ```bash
   python diffusion_test_vis.py \
     --bundle runs/ddpm32_dit_inpaint_q_only_long/bundle_best.pt \
     --n-test-samples 5
   ```

4. **查看结果**
   - 检查控制台输出的统计表格
   - 分析生成的 `test_visualization.png`

## 常见问题

### Q: 脚本提示找不到模型文件？
A: 确保已经完成训练，检查 `runs/` 目录下是否有对应的模型文件夹

### Q: 预测结果与真值差异很大？
A: 
- 检查模型是否已收敛（查看训练日志）
- 增加采样步数 `--sample-steps`
- 尝试不同的随机种子

### Q: 运行很慢？
A: 
- 减少 `--n-test-samples` 或 `--n-predictions`
- 减少 `--sample-steps`
- 确认在GPU上运行 `--device cuda`

### Q: 内存不足？
A: 
- 减少每个样本的预测数量 `--n-predictions`
- 使用CPU: `--device cpu`（虽然会更慢）

## 输出文件

| 文件 | 说明 |
|------|------|
| `test_visualization.png` | 主要可视化结果，展示预测 vs 真值对比 |

## 扩展建议

可以进一步修改脚本来：
- 添加轨迹可视化（整条轨迹的生成）
- 计算更复杂的评估指标（例如轨迹平滑度）
- 生成GIF动画显示采样过程
- 与其他基线方法对比
