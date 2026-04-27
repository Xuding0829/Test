"""
步骤5：单张图片推理演示
使用微调后的模型对任意茶叶图片进行问答
"""
import os

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, LoraConfig, TaskType
from qwen_vl_utils import process_vision_info

# ===== 配置 =====
MODEL_PATH = "./Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR = "./output/tea_disease_vl"
LORA_PATH = None  # 可手动指定，例如 "./output/tea_disease_vl/checkpoint-300"


def resolve_lora_path():
    """优先使用手动指定路径，否则自动选择最新 checkpoint。"""
    if LORA_PATH:
        return LORA_PATH

    if os.path.exists(os.path.join(OUTPUT_DIR, "adapter_config.json")):
        return OUTPUT_DIR

    if not os.path.isdir(OUTPUT_DIR):
        return None

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


def load_model():
    """加载微调模型"""
    print("加载模型...")
    lora_path = resolve_lora_path()
    if not lora_path:
        raise FileNotFoundError(f"未找到 LoRA 权重，请检查 {OUTPUT_DIR}")

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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

    print("✅ 模型加载完成")
    print(f"使用 LoRA: {lora_path}")
    return model, processor


def answer_question(image_path, question, model, processor):
    """对图片提问"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
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
    print("Qwen2-VL-2B 茶叶病虫害问答演示")
    print("=" * 60)

    if not resolve_lora_path():
        print(f"\n❌ 错误: 找不到 LoRA 模型")
        print(f"输出目录: {OUTPUT_DIR}")
        print("请先完成训练，或在脚本顶部手动设置 LORA_PATH")
        return

    # 加载模型
    model, processor = load_model()

    # 交互式问答
    print("\n" + "=" * 60)
    print("问答模式：输入图片路径和问题，或按 Ctrl+C 退出")
    print("默认问题: 这张茶叶图片显示了什么病害？")
    print("=" * 60)

    while True:
        try:
            image_path = input("\n📷 图片路径: ").strip()
            if not image_path:
                continue
            if not os.path.exists(image_path):
                print(f"❌ 文件不存在: {image_path}")
                continue

            question_input = input("❓ 问题 (回车使用默认): ").strip()
            if not question_input:
                question_input = "这张茶叶图片显示了什么病害？"

            print("🤔 推理中...")
            answer = answer_question(image_path, question_input, model, processor)

            print(f"\n💬 回答: {answer}\n")
            print("-" * 40)

        except KeyboardInterrupt:
            print("\n\n退出程序")
            break


if __name__ == "__main__":
    main()
