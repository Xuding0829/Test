# Qwen2-VL-2B 茶叶病虫害多模态微调包

## 📦 包内容

```
multimodal_finetune/
├── README.md                      # 本文档
├── requirements.txt               # Python 依赖
├── train.sh                       # 一键训练脚本
│
├── 1_download_model.py            # 步骤1：下载模型
├── 2_convert_data.py              # 步骤2：转换数据格式
├── 3_train.py                     # 步骤3：开始训练
├── 4_eval.py                      # 步骤4：评估模型
├── 5_inference.py                 # 步骤5：推理演示
│
└── tea_data/                      # 茶叶病虫害数据集
    ├── sft_vl_train_shuffle.jsonl # 训练数据 (796条)
    ├── sft_vl_valid_shuffle.jsonl # 验证数据 (89条)
    └── tea sickness dataset/      # 图片文件夹 (885张)
        ├── Anthracnose/           # 炭疽病
        ├── algal leaf/            # 藻斑病
        ├── bird eye spot/         # 鸟眼斑病
        ├── brown blight/          # 茶赤叶斑病
        ├── gray light/            # 茶灰斑病
        ├── healthy/               # 健康叶片
        ├── red leaf spot/         # 红叶斑病
        └── white spot/            # 白斑病
```

## 🚀 快速开始

### 1. 上传到服务器

将整个 `multimodal_finetune` 文件夹上传到学校服务器。

### 2. 安装依赖

```bash
cd multimodal_finetune
pip install -r requirements.txt
```

### 3. 下载模型

```bash
python 1_download_model.py
```

### 4. 转换数据格式

```bash
python 2_convert_data.py
```

### 5. 开始训练

```bash
bash train.sh
```

或者直接运行：
```bash
python 3_train.py
```

---

## 📊 硬件配置要求

| 显卡 | 显存 | batch_size | gradient_accumulation |
|------|------|------------|----------------------|
| RTX 3090 | 24GB | 2 | 8 (总batch=16) |
| RTX 4090 | 24GB | 2 | 8 (总batch=16) |
| A100 | 40GB+ | 4 | 8 (总batch=32) |
| V100 | 16GB | 1 | 16 (总batch=16) |

---

## ⚠️ 注意事项

1. **路径不要有中文** - 所有路径必须是英文
2. **确保图片和标注数量一致** - 图片损坏会导致加载失败
3. **显存不足时** - 减小 `per_device_train_batch_size`
4. **训练中断** - 使用 `trainer.train(resume_from_checkpoint=True)`

---

## 📈 训练输出

训练完成后，结果保存在 `output/tea_disease_vl/` 目录下：

```
output/tea_disease_vl/
├── checkpoint-100/       # 第100步的checkpoint
├── checkpoint-200/       # 第200步的checkpoint
├── checkpoint-300/       # 第300步的checkpoint
├── runs/                  # TensorBoard 日志
└── trainer_state.json     # 训练状态
```

---

## 🔧 自定义配置

如需修改训练参数，编辑 `3_train.py`：

```python
# 训练参数
per_device_train_batch_size = 2      # 根据显存调整
gradient_accumulation_steps = 8     # 总batch = 2*8=16
num_train_epochs = 3                 # 训练轮数
learning_rate = 1e-4                 # 学习率
lora_r = 32                          # LoRA 秩

# 模型路径（如果模型不在默认位置）
MODEL_PATH = "./Qwen/Qwen2-VL-2B-Instruct"
```

---

## ❓ 常见问题

### Q: 报错 "No module named 'transformers'"
A: 运行 `pip install -r requirements.txt`

### Q: 报错 "CUDA out of memory"
A: 减小 batch_size 到 1，或使用更小的图像分辨率

### Q: 报错 "找不到模型"
A: 确保已运行 `1_download_model.py` 下载模型

### Q: 训练中断后如何继续？
A: 修改 `3_train.py`，添加 `resume_from_checkpoint=True`

### Q: 如何只训练几步测试？
A: 修改 `num_train_epochs = 0.1`（训练10%个epoch）

---

## 📝 数据格式说明

### 原始数据格式 (jsonl)
```json
{
    "image_info": [{"matched_text_index": 0, "image_url": "..."}],
    "text_info": [
        {"text": "问题", "tag": "mask"},
        {"text": "回答", "tag": "no_mask"}
    ]
}
```

### 转换后格式 (json)
```json
{
    "id": "tea_001",
    "conversations": [
        {"from": "user", "value": "<|vision_start|>图片路径<|vision_end|>问题"},
        {"from": "assistant", "value": "回答"}
    ]
}
```

---

## 📧 联系方式

如有问题，请联系作者。