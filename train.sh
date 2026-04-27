#!/bin/bash
# ============================================================
# Qwen2-VL-2B 茶叶病虫害微调训练 - 一键训练脚本
# 适用于 RTX 3090 (24GB) 单卡环境
# ============================================================
set -e

echo "========================================="
echo "Qwen2-VL-2B 茶叶病虫害微调训练"
echo "========================================="
echo ""

# 检测 Python 环境
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ 未找到 Python，请先安装 Python 3.9+"
    exit 1
fi

echo "使用 Python: $($PYTHON --version)"
echo ""

# 检测 GPU
$PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')" || {
    echo "❌ 未检测到 CUDA，此脚本需要 NVIDIA GPU"
    exit 1
}

# 步骤1: 转换数据格式
echo ""
echo "========================================="
echo "[1/3] 转换数据格式..."
echo "========================================="
$PYTHON 2_convert_data.py

# 步骤2: 下载模型（如果尚未下载）
if [ ! -d "Qwen" ]; then
    echo ""
    echo "========================================="
    echo "[2/3] 下载模型（首次运行约需 10 分钟）..."
    echo "========================================="
    $PYTHON 1_download_model.py
else
    echo ""
    echo "[跳过] 模型已下载 (./Qwen/)"
fi

# 步骤3: 开始训练
echo ""
echo "========================================="
echo "[3/3] 开始训练..."
echo "========================================="
echo "提示: 按 Ctrl+C 可中断训练"
echo "      训练日志保存在 ./output/tea_disease_vl/"
echo "      checkpoint 每 50 步保存一次"
echo ""

# 可自定义参数
#   TEA_NUM_EPOCHS=5           训练轮数（默认3）
#   TEA_GRAD_ACCUM_STEPS=16    梯度累积（默认8）
#   TEA_MAX_STEPS=300          最多训练步数
#   TEA_TRAIN_LIMIT=100        仅用前100条数据（调试）
#   TEA_DEBUG_FORWARD=1        单batch测试后退出

$PYTHON 3_train.py

echo ""
echo "========================================="
echo "训练完成！"
echo "========================================="
echo ""
echo "📊 结果位置: ./output/tea_disease_vl/"
echo ""
echo "🔍 评估模型:"
echo "  python 4_eval.py"
echo ""
echo "💬 交互式推理:"
echo "  python 5_inference.py"
echo ""
echo "📦 合并导出:"
echo "  python 6_export_model.py"
echo ""
echo "========================================="
