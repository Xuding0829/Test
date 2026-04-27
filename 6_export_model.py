"""
步骤6：导出合并模型（可选）
将 LoRA 权重合并到基础模型，生成可独立部署的完整模型
用于集成到 system/pipeline.py 中
"""
import os

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# ===== 配置 =====
MODEL_PATH = "./Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR = "./output/tea_disease_vl"
EXPORT_DIR = "./output/tea_disease_merged"
LORA_PATH = None  # 可手动指定


def resolve_lora_path():
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


def main():
    print("=" * 60)
    print("合并 LoRA 权重到基础模型")
    print("=" * 60)

    lora_path = resolve_lora_path()
    if not lora_path or not os.path.exists(lora_path):
        print(f"❌ 错误: 找不到 LoRA 权重 ({OUTPUT_DIR})")
        print("请先完成训练。")
        return

    print(f"\n基础模型: {MODEL_PATH}")
    print(f"LoRA 权重: {lora_path}")
    print(f"输出目录: {EXPORT_DIR}")

    # 加载基础模型 + LoRA
    print("\n[1/3] 加载模型...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)

    # 合并权重
    print("[2/3] 合并权重...")
    model = model.merge_and_unload()

    # 保存
    print("[3/3] 保存模型...")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    model.save_pretrained(EXPORT_DIR, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor.save_pretrained(EXPORT_DIR)

    print(f"\n✅ 合并完成！")
    print(f"完整模型保存在: {EXPORT_DIR}")
    print(f"\n此模型可以直接用于 system/pipeline.py 的多模态问答模块。")
    print("=" * 60)


if __name__ == "__main__":
    main()
