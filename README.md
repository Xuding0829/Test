# Qwen2-VL-2B 茶叶病虫害多模态微调训练包

## 📦 包内容

```
qwen2vl_train_pack/
├── README.md                    # 本文档
├── requirements.txt             # Python 依赖
├── pack.sh                      # 本地打包脚本（Mac上运行）
├── train.sh                     # 服务器一键训练脚本
│
├── 1_download_model.py          # 步骤1：下载 Qwen2-VL-2B 模型
├── 2_convert_data.py            # 步骤2：数据格式转换
├── 3_train.py                   # 步骤3：LoRA 微调训练
├── 4_eval.py                    # 步骤4：评估模型
├── 5_inference.py               # 步骤5：交互式推理演示
├── 6_export_model.py            # 步骤6：合并导出完整模型（可选）
│
└── tea_data/                    # 茶叶病虫害数据集（打包时从原始目录复制）
    ├── sft_vl_train_shuffle.jsonl    # 训练标注（797条）
    ├── sft_vl_valid_shuffle.jsonl    # 验证标注（90条）
    ├── Anthracnose/                   # 炭疽病（99张）
    ├── algal leaf/                    # 藻斑病（113张）
    ├── bird eye spot/                 # 鸟眼斑病（95张）
    ├── brown blight/                  # 茶褐斑病（135张）
    ├── gray light/                    # 茶灰斑病（100张）
    ├── healthy/                       # 健康叶片（108张）
    ├── red leaf spot/                 # 红叶斑病（155张）
    └── white spot/                    # 白斑病（80张）
```

---

## 🚀 快速开始

### 在本地 Mac 上打包

```bash
cd qwen2vl_train_pack
chmod +x pack.sh
./pack.sh
```

这会生成一个 `qwen2vl_tepest_train_YYYYMMDD_HHMM.tar.gz` 文件。

### 上传到 GPU 服务器

```bash
scp qwen2vl_teapest_train_*.tar.gz user@your-server:/home/user/
```

### 在服务器上训练

```bash
# 解压
tar -xzf qwen2vl_teapest_train_*.tar.gz
cd qwen2vl_teapest_train_*/

# 安装依赖
pip install -r requirements.txt

# 一键训练
bash train.sh
```

或者分步执行：

```bash
python 2_convert_data.py    # 转换数据格式
python 1_download_model.py  # 下载模型（首次运行）
python 3_train.py           # 开始训练
```

---

## 📊 硬件要求

| 显卡 | 显存 | 配置 | 等效 Batch |
|------|------|------|-----------|
| **RTX 3090** | 24GB | bs=1, accum=8 | 8 |
| RTX 4090 | 24GB | bs=1, accum=8 | 8 |
| A100 | 40GB+ | bs=2, accum=8 | 16 |
| V100 | 16GB | bs=1, accum=16 | 16 |

> 默认配置针对 **RTX 3090 (24GB)** 优化。

---

## ⚙️ 自定义训练参数

通过环境变量覆盖（不需要改代码）：

```bash
# 训练更多轮数
TEA_NUM_EPOCHS=5 python 3_train.py

# 更大的梯度累积（更稳定）
TEA_GRAD_ACCUM_STEPS=16 python 3_train.py

# 只训练 100 步（快速测试）
TEA_MAX_STEPS=100 python 3_train.py

# 只用前 50 条数据调试
TEA_TRAIN_LIMIT=50 python 3_train.py

# 运行单 batch 前向/反向测试（验证流程通不通）
TEA_DEBUG_FORWARD=1 python 3_train.py
```

---

## 📈 训练流程说明

### 数据格式

**原始格式** (ERNIEKit SFT-VL JSONL)：
```json
{
  "image_info": [{"matched_text_index": 0, "image_url": "tea sickness dataset/gray light/xxx.jpg"}],
  "text_info": [
    {"text": "这张茶叶图片显示了什么病害？", "tag": "mask"},
    {"text": "这张茶叶图片显示的是茶灰斑病。...", "tag": "no_mask"}
  ]
}
```

**转换后** (Qwen2-VL 对话 JSON)：
```json
{
  "id": "tea_001",
  "conversations": [
    {"from": "user", "value": "<|vision_start|>tea_data/gray light/xxx.jpg<|vision_end|>这张茶叶图片显示了什么病害？"},
    {"from": "assistant", "value": "这张茶叶图片显示的是茶灰斑病。..."}
  ]
}
```

### 训练配置

- **模型**: Qwen2-VL-2B-Instruct（约 3.9GB）
- **微调方式**: LoRA (r=4, alpha=8)
- **图片分辨率**: 224×224
- **混合精度**: FP16
- **梯度检查点**: 开启（省显存）
- **LoRA 目标模块**: q/k/v/o_proj, gate/up/down_proj

---

## 🔍 训练后操作

### 评估模型
```bash
python 4_eval.py
```
在验证集上评估准确率（自动选最新 checkpoint）。

### 交互式推理
```bash
python 5_inference.py
```
输入图片路径，实时问答。

### 合并导出（用于系统集成）
```bash
python 6_export_model.py
```
将 LoRA 权重合并到基础模型，生成 `./output/tea_disease_merged/`。

导出后，将 `pipeline.py` 中的路径指向合并后的模型即可集成到 TeaVisor 系统。

---

## ⚠️ 注意事项

1. **路径不要有中文** — 所有路径必须是英文（服务器上解压后自动满足）
2. **确保图片完整** — 共 885 张图片，8 个类别
3. **显存不足时** — 在 `3_train.py` 中将 `IMAGE_SIZE` 从 224 减到 128
4. **训练中断恢复** — Trainer 会自动从最新 checkpoint 恢复
5. **首次下载模型** — 约 3.9GB，国内用 ModelScope 速度较快

---

## ❓ 常见问题

| 问题 | 解决方案 |
|------|---------|
| `No module named 'transformers'` | `pip install -r requirements.txt` |
| `CUDA out of memory` | 减小 `IMAGE_SIZE` 或增大 `GRAD_ACCUM_STEPS` |
| `找不到模型` | 先运行 `python 1_download_model.py` |
| `图片路径不存在` | 检查 `tea_data/` 下图片目录是否完整 |
| 训练 loss 不下降 | 增加 `NUM_EPOCHS` 或增大 `lora_r` |

---

## 🔗 与 TeaVisor 系统集成

训练完成后，将微调模型集成到系统的步骤：

1. **合并 LoRA**：运行 `python 6_export_model.py`
2. **复制模型**：将 `output/tea_disease_merged/` 复制到项目的 `multimodal_finetune/output/tea_disease_vl/` 目录
3. **启动系统**：运行 `python system/app.py`
4. 系统的 `pipeline.py` 会自动发现并加载微调后的模型

如果不想合并，也可以直接让 `pipeline.py` 加载 LoRA checkpoint（自动选择最新的）。
