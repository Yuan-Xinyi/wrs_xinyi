# 3D可视化评估脚本 - Franka机器人可视化

## 概述

`diffusion_eval_3d_vis.py` 脚本使用WRS框架的Franka机器人进行3D可视化，直观对比预测结果和真实结果。

## 功能特性

✅ **多预测样本可视化**
- 🟢 绿色（α=0.9）：真实关节配置 (Ground Truth)
- 🔴→🟡 红→黄渐变：**所有预测样本**（不仅仅是平均值）
- 透明度递减表示预测顺序
- 同时显示所有预测的分布

✅ **详细误差分析**
- 逐关节对比真值vs预测均值
- 计算关节角误差和不确定性
- 显示预测的标准差（不确定性范围）
- 统计平均误差和最大误差

✅ **批量FK计算**
- 为所有配置（真实+所有预测）高效计算正向运动学
- 支持多机器人实例并行可视化
- 错误处理：FK失败时自动使用默认配置

✅ **数据集过滤**
- 自动过滤 remaining_length > 0.75 的高质量数据
- 支持自定义采样

✅ **可交互的3D场景**
- 可旋转、缩放、平移视图
- 同时显示多个机器人配置对比
- 坐标系可视化参考

## 系统要求

- Python 3.8+
- WRS (世界机器人学模拟系统)
- PyTorch
- Panda3D (for visualization)

## 使用方法

### 基础用法

```bash
cd /path/to/diffusion_inpainting
python diffusion_eval_3d_vis.py
```

### 自定义参数

```bash
python diffusion_eval_3d_vis.py \
  --bundle runs/dit_kinematic_inpainting_runs/ddpm32_dit_inpaint_q_only_long/bundle_best.pt \
  --h5-path datasets/franka_research_3_gpu_trajectories_sub10_long.hdf5 \
  --n-test-samples 3 \
  --n-predictions 8 \
  --sample-steps 32 \
  --device cuda \
  --seed 42
```

## 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--bundle` | Path | `bundle_latest.pt` | 已训练模型的bundle文件路径 |
| `--h5-path` | Path | 长数据集路径 | 测试数据集HDF5文件 |
| `--device` | str | `cuda/cpu` | 计算设备选择 |
| `--n-test-samples` | int | 2 | 从数据集中随机抽取多少个测试样本 |
| `--n-predictions` | int | 8 | 对每个样本生成多少个随机预测 |
| `--sample-steps` | int | 32 | DDPM采样的扩散步数 |
| `--seed` | int | 42 | 随机种子 (重现结果用) |

## 输出信息

### 1. 控制台输出示例

```
====================================================================================================
[ Franka 机器人 - 3D可视化评估 ]
====================================================================================================

📂 加载模型包: ...
✓ 模型信息:
  q_dim: 7
  token_dim: 16

📂 加载模型权重: ...

📂 从数据集加载样本: ...
✓ 加载了 2 个测试样本

🎯 生成预测 (每个样本生成 8 个预测)...
  [1/2] ✓
  [2/2] ✓

📊 生成3D可视化...

【样本 1】 traj_053780 (t=28)

  Remaining Length: 0.8908
  Position: [-0.01525938 -0.41420713  0.4596424 ]
  Direction: [-0.9466461   0.16470166 -0.27701014]
  Target Normal: [ 0.15186583 -0.53015035 -0.8341927 ]

  关节角对比 (Ground Truth vs Predicted Mean±Std):
  Joint      True         Pred Mean    Std          Error       
  ----------------------------------------------------------
  q1             -1.3228     -1.2884      0.0399      0.0344
  q2             -0.7982     -0.8248      0.0112      0.0266
  q3             -0.1450     -0.1910      0.0332      0.0461
  q4             -2.6400     -2.6542      0.0192      0.0142
  q5             -0.7474     -0.7152      0.0395      0.0321
  q6              2.3338      2.3542      0.0361      0.0204
  q7              1.9460      1.9611      0.0123      0.0152

  📊 误差统计:
    平均关节角误差: 0.0270 rad (1.55°)
    最大关节角误差: 0.0461 rad (2.64°)
    平均预测不确定性: 0.0253 rad

  💡 图例:
    🟢 绿色: 真实关节配置 (Ground Truth)
    🔴 红色: 预测关节配置 (Prediction Mean)

  ✓ 完成样本 1
```

### 2. 3D可视化窗口

**窗口标题**：显示样本信息和remaining_length值

**场景内容**：
- 灰色坐标系作为参考
- 🟢 绿色（不透明α=0.9）机器人：真实关节配置
- 🔴→🟠→🟡 红→橙→黄（逐渐透明）机器人：**所有预测样本**
  - 绿色是ground truth
  - 所有其他颜色的机器人是各个预测样本
  - 透明度递减表示预测顺序（第1个预测最明显，最后一个预测最透明）
  - 所有机器人的分布直观显示了预测的多样性和稳定性

**交互操作**：
- 鼠标左键拖动：旋转视图
- 鼠标滚轮：缩放
- 鼠标右键拖动：平移

**如何理解可视化**：
- 如果所有预测的机器人都聚集在一起（颜色叠在一起）→ 预测很确定
- 如果预测的机器人分散开 → 预测的不确定性大
- 与绿色机器人的距离 → 预测的系统偏差

## 工作流程

### 第1步：准备模型

确保已经完成训练：
```bash
python diffusion_inpainting/diffusion_train.py \
  --epochs 50  # 最少训练轮次
```

输出文件位置：
```
runs/dit_kinematic_inpainting_runs/ddpm32_dit_inpaint_q_only_long/
├── bundle_latest.pt      # 最新checkpoint
├── bundle_best.pt        # 最优模型  ✓ 推荐使用
├── model_latest.pt
└── ...
```

### 第2步：运行可视化

```bash
python diffusion_eval_3d_vis.py --n-test-samples 3
```

### 第3步：查看结果

1. **控制台统计**：你会看到详细的误差数据和不确定性信息
2. **3D窗口**：逐个展示样本的可视化结果
3. **交互分析**：在3D窗口中旋转、缩放来观察差异

## 评估指标解释

### 关节角误差 (rad 和 °)

- **平均误差 < 0.05 rad (3°)**：优秀
- **平均误差在 0.05~0.1 rad**：良好
- **平均误差 > 0.1 rad**：需要改进

### 预测不确定性 (Std)

- **低方差 (< 0.02 rad)**：模型很有信心
- **中等方差 (0.02~0.05 rad)**：合理的预测范围
- **高方差 (> 0.05 rad)**：模型不确定性高，需要检查数据

### 最大误差

用于检测异常情况：
- 观察哪个关节出现最大误差是否合理
- 检查是否接近关节限制或奇异点

## 常见问题

### Q: 窗口显示不出来？

A: 如果在服务器或无GUI环境运行，这是正常的。此时脚本会输出详细的控制台统计信息。

### Q: 预测误差很大？

A: 
- 检查模型是否收敛（查看训练日志的loss曲线）
- 尝试用`bundle_best.pt`而不是`bundle_latest.pt`
- 增加训练轮次
- 检查数据集过滤是否正确

### Q: 某个关节的误差特别大？

A:
- 可能是关节处于奇异点或接近关节限制
- 检查该关键位置的数据是否足够
- 考虑在loss函数中增加该关节的权重

### Q: 怎么对比不同模型？

A: 运行脚本时指定不同的bundle：
```bash
# 模型A
python diffusion_eval_3d_vis.py --bundle runs/.../modelA/bundle_best.pt

# 模型B
python diffusion_eval_3d_vis.py --bundle runs/.../modelB/bundle_best.pt
```

## 扩展建议

可以进一步改进脚本来：
- 保存多视图截图进行报告
- 生成本体轨迹动画 (展示完整运动过程)
- 计算轨迹平滑度指标
- 与其他基线方法对比
- 添加TCP位置误差分析（一旦FK问题解决）

## 故障排除

### ImportError: No module named 'wrs'

确保从项目根目录运行，或添加到Python路径：
```bash
cd /home/lqin/wrs_xinyi
python 0000_test_programs/surgery_diff/CleanDiffuser/GPU_nullspace_straight/diffusion_inpainting/diffusion_eval_3d_vis.py
```

### 模型加载失败

检查模型路径是否正确，并确保有写入权限：
```bash
ls -la runs/dit_kinematic_inpainting_runs/ddpm32_dit_inpaint_q_only_long/
```

### GPU内存不足

减少参数：
- `--n-test-samples` 
- `--n-predictions`
- `--sample-steps`

或使用CPU：
```bash
python diffusion_eval_3d_vis.py --device cpu
```
