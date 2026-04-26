#!/bin/bash
# Qwen2-VL-2B 茶叶病虫害微调训练脚本
# RTX 3090 (24GB) 优化配置

echo "========================================="
echo "Qwen2-VL-2B 茶叶病虫害微调训练"
echo "========================================="

# 每次训练前重新转换，避免旧的路径缓存影响训练
echo "[1/3] 转换数据格式..."
python 2_convert_data.py

# 检查是否有模型
if [ ! -d "Qwen" ]; then
    echo "[2/3] 下载模型..."
    python 1_download_model.py
else
    echo "[跳过] 模型已下载"
fi

# 开始训练
echo "[3/3] 开始训练..."
python 3_train.py

echo "========================================="
echo "训练完成！"
echo "模型保存在: output/tea_disease_vl/"
echo "========================================="
