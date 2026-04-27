"""
步骤3：Qwen2-VL-2B 茶叶病虫害多模态微调训练
针对 RTX 3090 (24GB) 显存优化配置

支持环境变量覆盖训练参数:
  TEA_DEBUG_BATCH=1          打印 collator 调试信息
  TEA_DEBUG_FORWARD=1        运行单 batch 前向/反向测试后退出
  TEA_TRAIN_LIMIT=N          只使用前 N 条训练数据（调试用）
  TEA_VAL_LIMIT=N            只使用前 N 条验证数据
  TEA_MAX_STEPS=N            最多训练 N 步（覆盖 epoch）
  TEA_GRAD_ACCUM_STEPS=N     梯度累积步数（默认 8）
  TEA_NUM_EPOCHS=3           训练轮数（默认 3）
"""
import os
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# ===== 配置 =====
MODEL_PATH = "./Qwen/Qwen2-VL-2B-Instruct"
TRAIN_DATA = "tea_data_processed/tea_train.json"
VAL_DATA = "tea_data_processed/tea_val.json"
OUTPUT_DIR = "./output/tea_disease_vl"
IMAGE_SIZE = 224          # 图片分辨率，越小越省显存
MAX_ANSWER_CHARS = 80     # 回答最大字符数（截断过长的回答）

# 环境变量覆盖
DEBUG_BATCH = os.getenv("TEA_DEBUG_BATCH", "0") == "1"
RUN_DEBUG_FORWARD = os.getenv("TEA_DEBUG_FORWARD", "0") == "1"
TRAIN_LIMIT = int(os.getenv("TEA_TRAIN_LIMIT", "0"))
VAL_LIMIT = int(os.getenv("TEA_VAL_LIMIT", "0"))
MAX_STEPS = int(os.getenv("TEA_MAX_STEPS", "-1"))
GRAD_ACCUM_STEPS = int(os.getenv("TEA_GRAD_ACCUM_STEPS", "8"))
NUM_EPOCHS = float(os.getenv("TEA_NUM_EPOCHS", "3"))

processor = None
tokenizer = None


class MultiModalDataCollator:
    """在 batch 阶段统一构造 Qwen2-VL 所需的多模态输入。
    
    核心逻辑：
    1. 对每个 sample 构造 user+assistant 消息
    2. 用 processor 分别处理 full_text 和 prompt_text
    3. 用 prompt 长度将 labels 中 prompt 部分设为 -100（不计算 loss）
    """

    def __init__(self, processor):
        self.processor = processor

    @staticmethod
    def build_user_message(example):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": example["image_path"],
                        "resized_height": IMAGE_SIZE,
                        "resized_width": IMAGE_SIZE,
                    },
                    {"type": "text", "text": example["question"]},
                ],
            }
        ]

    def __call__(self, features):
        batch_start = time.time()
        if DEBUG_BATCH:
            print(f"[collator] start, batch_size={len(features)}", flush=True)

        full_texts = []
        prompt_texts = []
        images = []

        for idx, example in enumerate(features):
            sample_start = time.time()
            user_messages = self.build_user_message(example)
            full_messages = user_messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["answer"]}],
                }
            ]

            # 完整对话 (用于 labels)
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            # 仅 prompt (用于确定 loss 掩码范围)
            prompt_texts.append(
                self.processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            # 加载图片
            with Image.open(example["image_path"]) as img:
                images.append(img.convert("RGB"))

            if DEBUG_BATCH:
                print(
                    f"[collator] sample {idx} prepared in {time.time() - sample_start:.2f}s",
                    flush=True,
                )

        # 处理完整对话
        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        # 处理仅 prompt
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        # 构造 labels: prompt 部分不计算 loss
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100  # padding 不计算

        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for idx, prompt_len in enumerate(prompt_lengths):
            labels[idx, :prompt_len] = -100  # prompt 不计算

        batch["labels"] = labels

        if DEBUG_BATCH:
            for key, value in batch.items():
                if hasattr(value, "shape"):
                    print(f"[collator] {key} shape={tuple(value.shape)}", flush=True)
            print(f"[collator] done in {time.time() - batch_start:.2f}s", flush=True)

        return batch


def process_func(example):
    """从转换后的 JSON 中提取训练需要的字段。"""
    conversation = example["conversations"]
    input_value = conversation[0]["value"]
    output_value = conversation[1]["value"]

    # 检查 vision 标记
    if "<|vision_start|>" not in input_value or "<|vision_end|>" not in input_value:
        return {
            "image_path": "",
            "question": "",
            "answer": "",
            "valid": False,
        }

    # 解析图片路径和问题
    image_path = input_value.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    question = input_value.split("<|vision_end|>")[-1].strip()
    answer = output_value.strip()[:MAX_ANSWER_CHARS]

    valid = bool(image_path and question and answer and os.path.exists(image_path))

    return {
        "image_path": image_path,
        "question": question,
        "answer": answer,
        "valid": valid,
    }


def main():
    global processor, tokenizer

    print("=" * 60)
    print("Qwen2-VL-2B 茶叶病虫害多模态微调")
    print("=" * 60)
    print(f"训练参数:")
    print(f"  epochs = {NUM_EPOCHS}")
    print(f"  grad_accum_steps = {GRAD_ACCUM_STEPS}")
    print(f"  image_size = {IMAGE_SIZE}")
    print(f"  max_answer_chars = {MAX_ANSWER_CHARS}")
    if MAX_STEPS > 0:
        print(f"  max_steps = {MAX_STEPS} (覆盖 epochs)")
    if TRAIN_LIMIT > 0:
        print(f"  train_limit = {TRAIN_LIMIT} (调试)")
    print()

    # ===== 前置检查 =====
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型 {MODEL_PATH}")
        print("请先运行: python 1_download_model.py")
        sys.exit(1)

    if not os.path.exists(TRAIN_DATA):
        print(f"❌ 错误: 找不到训练数据 {TRAIN_DATA}")
        print("请先运行: python 2_convert_data.py")
        sys.exit(1)

    if not os.path.exists(VAL_DATA):
        print(f"❌ 错误: 找不到验证数据 {VAL_DATA}")
        print("请先运行: python 2_convert_data.py")
        sys.exit(1)

    # ===== 加载模型 =====
    print("\n[1/5] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, use_fast=False, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        min_pixels=IMAGE_SIZE * IMAGE_SIZE,
        max_pixels=IMAGE_SIZE * IMAGE_SIZE,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    if not torch.cuda.is_available():
        print("❌ 错误: 未检测到 CUDA，此脚本需要 NVIDIA GPU 环境。")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    model = model.to("cuda")
    model.enable_input_require_grads()
    model.config.use_cache = False
    print("✅ 模型加载完成")

    # ===== 加载数据 =====
    print("\n[2/5] 加载数据...")
    train_ds = Dataset.from_json(TRAIN_DATA)
    val_ds = Dataset.from_json(VAL_DATA)
    print(f"原始 - 训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

    # ===== 整理数据 =====
    print("\n[3/5] 整理数据字段...")
    train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
    val_dataset = val_ds.map(process_func, remove_columns=val_ds.column_names)
    train_dataset = train_dataset.filter(lambda x: x["valid"])
    val_dataset = val_dataset.filter(lambda x: x["valid"])

    if TRAIN_LIMIT > 0:
        train_dataset = train_dataset.select(range(min(TRAIN_LIMIT, len(train_dataset))))
    if VAL_LIMIT > 0:
        val_dataset = val_dataset.select(range(min(VAL_LIMIT, len(val_dataset))))

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ 过滤后数据为空，请检查图片路径和数据格式。")
        sys.exit(1)

    print(f"有效 - 训练={len(train_dataset)}, 验证={len(val_dataset)}")

    # ===== 配置 LoRA =====
    print("\n[4/5] 配置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        inference_mode=False,
        r=4,                # LoRA 秩（3090 24GB 用 4 足够）
        lora_alpha=8,       # 缩放系数
        lora_dropout=0.05,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # ===== 训练参数 =====
    total_batch = 1 * GRAD_ACCUM_STEPS
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
        learning_rate=1e-4,
        warmup_steps=5,
        logging_steps=1,
        eval_steps=1000,
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        eval_strategy="no",
        save_strategy="steps",
        report_to="none",
    )

    print(f"\n训练配置:")
    print(f"  每卡 batch_size = 1")
    print(f"  梯度累积 = {GRAD_ACCUM_STEPS}")
    print(f"  等效 batch_size = {total_batch}")
    print(f"  总训练步数 ≈ {len(train_dataset) * NUM_EPOCHS // total_batch}")
    print(f"  学习率 = 1e-4")
    print(f"  fp16 = True")

    # ===== 开始训练 =====
    print("\n[5/5] 开始训练...")
    print("-" * 60)

    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=MultiModalDataCollator(processor=processor),
    )

    # 可选：单 batch 前向/反向测试
    if RUN_DEBUG_FORWARD:
        print("\n[debug] 单 batch 前向/反向测试", flush=True)
        debug_collator = MultiModalDataCollator(processor=processor)
        debug_batch = debug_collator([train_dataset[0]])

        for key, value in debug_batch.items():
            if hasattr(value, "shape"):
                print(f"[debug] {key}: {tuple(value.shape)}", flush=True)

        forward_batch = {
            key: value.to("cuda") if torch.is_tensor(value) else value
            for key, value in debug_batch.items()
        }

        forward_start = time.time()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            forward_outputs = peft_model(**forward_batch)
        print(
            f"[debug] forward done in {time.time() - forward_start:.2f}s, "
            f"loss={forward_outputs.loss.item():.6f}",
            flush=True,
        )

        backward_start = time.time()
        forward_outputs.loss.backward()
        print(
            f"[debug] backward done in {time.time() - backward_start:.2f}s",
            flush=True,
        )
        peft_model.zero_grad(set_to_none=True)

        print("[debug] 测试通过，退出。", flush=True)
        sys.exit(0)

    trainer.train()

    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print(f"模型保存于: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
