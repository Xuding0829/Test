"""
步骤4：评估微调后的模型
使用验证集对微调模型进行简单准确率评估
"""
import os
import re

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, LoraConfig, TaskType
from qwen_vl_utils import process_vision_info
from datasets import Dataset

# ===== 配置 =====
MODEL_PATH = "./Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR = "./output/tea_disease_vl"
LORA_PATH = None  # 可手动指定，例如 "./output/tea_disease_vl/checkpoint-300"
VAL_DATA = "tea_data_processed/tea_val.json"


def resolve_lora_path():
    """优先使用手动指定路径，否则自动选择最新 checkpoint。"""
    if LORA_PATH:
        return LORA_PATH

    # 检查是否有 adapter_config.json (merge 后的最终模型)
    if os.path.exists(os.path.join(OUTPUT_DIR, "adapter_config.json")):
        return OUTPUT_DIR

    if not os.path.isdir(OUTPUT_DIR):
        return None

    # 选择 step 最大的 checkpoint
    checkpoint_dirs = []
    for name in os.listdir(OUTPUT_DIR):
        if not name.startswith("checkpoint-"):
            continue
        full_path = os.path.join(OUTPUT_DIR, name)
        if os.path.isdir(full_path):
            try:
                step = int(name.split("-")[-1])
            except ValueError:
                continue
            checkpoint_dirs.append((step, full_path))

    if not checkpoint_dirs:
        return None
    return max(checkpoint_dirs, key=lambda item: item[0])[1]


def normalize_text(text):
    """去除标点空格，转小写，便于比较"""
    return re.sub(r"\s+", "", re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text or "")).lower()


def is_prediction_match(pred_answer, true_answer):
    """宽松匹配：包含关系或字符重叠度 >= 60%"""
    pred = normalize_text(pred_answer)
    truth = normalize_text(true_answer)
    if not pred or not truth:
        return False
    if pred in truth or truth in pred:
        return True

    pred_chars = set(pred)
    truth_chars = set(truth)
    overlap = len(pred_chars & truth_chars) / max(len(truth_chars), 1)
    return overlap >= 0.6


def predict(model, processor, messages):
    """推理函数"""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def main():
    print("=" * 60)
    print("Qwen2-VL-2B 微调模型评估")
    print("=" * 60)

    lora_path = resolve_lora_path()

    if not lora_path or not os.path.exists(lora_path):
        print(f"\n❌ 错误: 找不到 LoRA 模型")
        print(f"输出目录: {OUTPUT_DIR}")
        print("请先完成训练，或在脚本顶部手动设置 LORA_PATH")
        return

    if not os.path.exists(VAL_DATA):
        print(f"\n❌ 错误: 找不到验证数据 {VAL_DATA}")
        print("请先运行: python 2_convert_data.py")
        return

    # 加载模型
    print(f"\n[1/3] 加载模型...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
    )

    model = PeftModel.from_pretrained(base_model, lora_path, config=config)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print(f"✅ 模型加载完成 (LoRA: {lora_path})")

    # 加载验证数据
    print(f"\n[2/3] 加载验证数据...")
    val_ds = Dataset.from_json(VAL_DATA)
    print(f"验证样本数: {len(val_ds)}")

    # 评估
    print(f"\n[3/3] 开始评估（最多 20 个样本）...")
    print("-" * 60)

    correct = 0
    total = min(len(val_ds), 20)

    for i in range(total):
        sample = val_ds[i]
        conversation = sample["conversations"]
        image_path = conversation[0]["value"].split("<|vision_start|>")[1].split("<|vision_end|>")[0]
        true_answer = conversation[1]["value"]
        question = conversation[0]["value"].split("<|vision_end|>")[-1]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }
        ]

        pred_answer = predict(model, processor, messages)

        # 简单匹配
        is_correct = is_prediction_match(pred_answer, true_answer)
        if is_correct:
            correct += 1

        print(f"\n[{i+1}/{total}]")
        print(f"  问题: {question}")
        print(f"  真实: {true_answer[:50]}...")
        print(f"  预测: {pred_answer[:50]}...")
        print(f"  结果: {'✅ 正确' if is_correct else '❌ 错误'}")

    print("\n" + "=" * 60)
    print(f"准确率: {correct}/{total} = {correct/total*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
