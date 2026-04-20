#!/bin/bash
# 快速启动3D可视化脚本

cd "$(dirname "$0")"

echo "======================================================================================================"
echo "[ Franka 机器人 3D 可视化评估 ]"
echo "======================================================================================================"
echo ""
echo "💡 用法:"
echo "   bash run_3d_vis.sh                    # 使用默认参数"
echo "   bash run_3d_vis.sh 3 8 32             # n_samples=3, n_predictions=8, sample_steps=32"
echo "   bash run_3d_vis.sh --bundle best.pt   # 使用最优模型"
echo ""
echo "======================================================================================================"
echo ""

# 默认参数
N_SAMPLES=${1:-2}
N_PREDICTIONS=${2:-8}
SAMPLE_STEPS=${3:-32}

# 处理标志参数
EXTRA_ARGS=""
if [[ "$*" == *"--bundle"* ]]; then
    # 传递所有额外参数
    EXTRA_ARGS="$@"
fi

echo "🚀 启动3D可视化..."
echo "   测试样本数: $N_SAMPLES"
echo "   每样本预测数: $N_PREDICTIONS"
echo "   采样步数: $SAMPLE_STEPS"
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi

# 运行脚本
python diffusion_eval_3d_vis.py \
    --n-test-samples "$N_SAMPLES" \
    --n-predictions "$N_PREDICTIONS" \
    --sample-steps "$SAMPLE_STEPS" \
    $EXTRA_ARGS

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================================================"
    echo "✓ 3D可视化完成！"
    echo "======================================================================================================"
else
    echo ""
    echo "❌ 3D可视化失败"
    exit 1
fi
