"""
步骤3：Qwen2-VL-2B 茶叶病虫害多模态微调训练
RTX 3090 (24GB) 优化配置
"""
import os

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
IMAGE_SIZE = 336
MAX_ANSWER_CHARS = 120

processor = None
tokenizer = None


class MultiModalDataCollator:
    """在 batch 阶段统一构造 Qwen2-VL 所需的多模态输入。"""

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
        full_texts = []
        prompt_texts = []
        images = []

        for example in features:
            user_messages = self.build_user_message(example)
            full_messages = user_messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["answer"]}],
                }
            ]

            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            with Image.open(example["image_path"]) as img:
                images.append(img.convert("RGB"))

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for idx, prompt_len in enumerate(prompt_lengths):
            labels[idx, :prompt_len] = -100

        batch["labels"] = labels
        return batch


def process_func(example):
    """从转换后的 json 中提取训练需要的字段。"""
    conversation = example["conversations"]
    input_value = conversation[0]["value"]
    output_value = conversation[1]["value"]

    if "<|vision_start|>" not in input_value or "<|vision_end|>" not in input_value:
        return {
            "image_path": "",
            "question": "",
            "answer": "",
            "valid": False,
        }

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

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型 {MODEL_PATH}")
        print("请先运行 python 1_download_model.py 下载模型")
        return

    if not os.path.exists(TRAIN_DATA):
        print(f"❌ 错误: 找不到训练数据 {TRAIN_DATA}")
        print("请先运行 python 2_convert_data.py 转换数据")
        return

    if not os.path.exists(VAL_DATA):
        print(f"❌ 错误: 找不到验证数据 {VAL_DATA}")
        print("请先运行 python 2_convert_data.py 转换数据")
        return

    print("\n[1/5] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, use_fast=False, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.enable_input_require_grads()
    model.config.use_cache = False
    print("✅ 模型加载完成")

    print("\n[2/5] 加载数据...")
    train_ds = Dataset.from_json(TRAIN_DATA)
    val_ds = Dataset.from_json(VAL_DATA)
    print(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

    print("\n[3/5] 整理数据字段...")
    train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
    val_dataset = val_ds.map(process_func, remove_columns=val_ds.column_names)
    train_dataset = train_dataset.filter(lambda x: x["valid"])
    val_dataset = val_dataset.filter(lambda x: x["valid"])

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ 过滤后数据为空，请检查图片路径和数据格式。")
        return
    print(f"整理完成: 训练={len(train_dataset)}, 验证={len(val_dataset)}")

    print("\n[4/5] 配置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        learning_rate=1e-4,
        warmup_steps=15,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        eval_strategy="steps",
        save_strategy="steps",
        report_to="none",
    )

    print("\n[5/5] 开始训练...")
    print("-" * 60)

    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=MultiModalDataCollator(processor=processor),
    )

    trainer.train()

    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print(f"模型保存于: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
