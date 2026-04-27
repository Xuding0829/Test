#!/bin/bash
# ============================================================
# Qwen2-VL-2B 茶叶病虫害微调 - 一键打包脚本
# 在本地 Mac 上运行，打包后上传到 GPU 服务器
# ============================================================
set -e

echo "========================================="
echo "Qwen2-VL 训练包打包工具"
echo "========================================="
echo ""

# ===== 配置 =====
PACK_NAME="qwen2vl_teapest_train_$(date +%Y%m%d_%H%M)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 原始数据路径（根据实际情况修改）
DATA_SOURCE_DIR="${SCRIPT_DIR}/../multimodal_finetune/tea_data"
DEST_DIR="${SCRIPT_DIR}/../${PACK_NAME}"

echo "📦 打包名称: ${PACK_NAME}"
echo "📂 数据源: ${DATA_SOURCE_DIR}"
echo "📍 输出目录: ${DEST_DIR}"
echo ""

# ===== 1. 创建目录结构 =====
echo "[1/5] 创建目录结构..."
mkdir -p "${DEST_DIR}/tea_data"
mkdir -p "${DEST_DIR}/output"
echo "  ✅ 目录创建完成"

# ===== 2. 复制脚本 =====
echo "[2/5] 复制训练脚本..."
cp "${SCRIPT_DIR}/1_download_model.py" "${DEST_DIR}/"
cp "${SCRIPT_DIR}/2_convert_data.py" "${DEST_DIR}/"
cp "${SCRIPT_DIR}/3_train.py" "${DEST_DIR}/"
cp "${SCRIPT_DIR}/4_eval.py" "${DEST_DIR}/"
cp "${SCRIPT_DIR}/5_inference.py" "${DEST_DIR}/"
cp "${SCRIPT_DIR}/6_export_model.py" "${DEST_DIR}/"
cp "${SCRIPT_DIR}/requirements.txt" "${DEST_DIR}/"
cp "${SCRIPT_DIR}/train.sh" "${DEST_DIR}/"
cp "${SCRIPT_DIR}/README.md" "${DEST_DIR}/"
echo "  ✅ 脚本复制完成"

# ===== 3. 复制数据（标注 + 图片） =====
echo "[3/5] 复制训练数据..."

if [ -d "${DATA_SOURCE_DIR}" ]; then
    # 复制 JSONL 标注文件
    if [ -f "${DATA_SOURCE_DIR}/sft_vl_train_shuffle.jsonl" ]; then
        cp "${DATA_SOURCE_DIR}/sft_vl_train_shuffle.jsonl" "${DEST_DIR}/tea_data/"
        echo "  ✅ 训练标注 (sft_vl_train_shuffle.jsonl)"
    fi
    if [ -f "${DATA_SOURCE_DIR}/sft_vl_valid_shuffle.jsonl" ]; then
        cp "${DATA_SOURCE_DIR}/sft_vl_valid_shuffle.jsonl" "${DEST_DIR}/tea_data/"
        echo "  ✅ 验证标注 (sft_vl_valid_shuffle.jsonl)"
    fi

    # 复制图片目录
    for category in Anthracnose "algal leaf" "bird eye spot" "brown blight" "gray light" healthy "red leaf spot" "white spot"; do
        if [ -d "${DATA_SOURCE_DIR}/${category}" ]; then
            cp -r "${DATA_SOURCE_DIR}/${category}" "${DEST_DIR}/tea_data/"
            count=$(ls -1 "${DEST_DIR}/tea_data/${category}"/*.jpg 2>/dev/null | wc -l | tr -d ' ')
            echo "  ✅ ${category} (${count} 张)"
        fi
    done
else
    echo "  ⚠️ 数据目录不存在: ${DATA_SOURCE_DIR}"
    echo "  请手动将数据放入: ${DEST_DIR}/tea_data/"
fi

# ===== 4. 清理 =====
echo "[4/5] 清理临时文件..."
find "${DEST_DIR}" -name ".DS_Store" -delete 2>/dev/null || true
echo "  ✅ 清理完成"

# ===== 5. 打包 =====
echo "[5/5] 打包为 tar.gz..."
cd "${SCRIPT_DIR}/.."
tar -czf "${PACK_NAME}.tar.gz" "${PACK_NAME}"
PACK_SIZE=$(du -h "${PACK_NAME}.tar.gz" | cut -f1)
echo "  ✅ 打包完成: ${PACK_NAME}.tar.gz (${PACK_SIZE})"

echo ""
echo "========================================="
echo "打包完成！"
echo "========================================="
echo ""
echo "📁 文件位置: $(pwd)/${PACK_NAME}.tar.gz"
echo "📦 大小: ${PACK_SIZE}"
echo ""
echo "📤 上传到服务器:"
echo "  scp ${PACK_NAME}.tar.gz user@server:/path/to/destination/"
echo ""
echo "🖥️  在服务器上解压:"
echo "  tar -xzf ${PACK_NAME}.tar.gz"
echo "  cd ${PACK_NAME}"
echo "  pip install -r requirements.txt"
echo "  bash train.sh"
echo "========================================="
