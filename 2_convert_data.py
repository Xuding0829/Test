"""
步骤2：将茶叶病虫害数据转换为 Qwen2-VL 训练格式
原始格式: ERNIEKit SFT-VL (image_info + text_info)
目标格式: Qwen2-VL 对话格式 (conversations + <|vision_start|>)
"""
import json
import os
import sys

# ===== 配置 =====
TRAIN_INPUT = "tea_data/sft_vl_train_shuffle.jsonl"
VAL_INPUT = "tea_data/sft_vl_valid_shuffle.jsonl"
OUTPUT_DIR = "tea_data_processed"
IMAGE_BASE = "tea_data/"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def resolve_image_path(raw_path):
    """兼容不同数据包中的图片相对路径格式。"""
    candidates = [os.path.normpath(os.path.join(IMAGE_BASE, raw_path))]

    # 兼容原始数据集路径前缀
    prefix = "tea sickness dataset/"
    if raw_path.startswith(prefix):
        trimmed = raw_path[len(prefix):]
        candidates.append(os.path.normpath(os.path.join(IMAGE_BASE, trimmed)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def convert_sample(sample):
    """将 ERNIEKit 格式转换为 Qwen2-VL 对话格式"""
    image_info = sample.get("image_info", [])
    text_info = sample.get("text_info", [])

    # 获取第一张图
    image_path = None
    for img in image_info:
        if img["matched_text_index"] == 0:
            image_path = resolve_image_path(img["image_url"])
            break

    if not image_path:
        return None

    # 提取问题和回答 (mask=no_mask 交替)
    question = ""
    answer = ""
    for text_item in text_info:
        if text_item["tag"] == "mask":
            question = text_item["text"]
        elif text_item["tag"] == "no_mask":
            answer = text_item["text"]

    if not question or not answer:
        return None

    return {
        "id": f"tea_{sample.get('id', 'unknown')}",
        "conversations": [
            {
                "from": "user",
                "value": f"<|vision_start|>{image_path}<|vision_end|>{question}"
            },
            {
                "from": "assistant",
                "value": answer
            }
        ]
    }


def process_file(input_path, output_path):
    """处理单个 JSONL 文件"""
    samples = []
    missing_images = 0
    missing_text = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                print(f"  ⚠️ 第 {line_num} 行 JSON 解析失败，跳过")
                continue

            converted = convert_sample(sample)
            if converted:
                samples.append(converted)
            else:
                # 判断是图片缺失还是文本缺失
                image_info = sample.get("image_info", [])
                text_info = sample.get("text_info", [])
                has_question = any(t["tag"] == "mask" and t["text"] for t in text_info)
                has_answer = any(t["tag"] == "no_mask" and t["text"] for t in text_info)

                if not has_question or not has_answer:
                    missing_text += 1
                else:
                    missing_images += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成: {os.path.basename(input_path)}")
    print(f"  ✅ 有效样本: {len(samples)}")
    if missing_images > 0:
        print(f"  ⚠️ 图片缺失: {missing_images}")
    if missing_text > 0:
        print(f"  ⚠️ 文本缺失: {missing_text}")

    return samples


def main():
    print("=" * 50)
    print("茶叶病虫害数据格式转换")
    print("=" * 50)

    # 检查输入文件
    if not os.path.exists(TRAIN_INPUT):
        print(f"\n❌ 错误: 找不到训练数据 {TRAIN_INPUT}")
        print("请确保数据放在正确位置。")
        print("数据目录结构应为:")
        print("  tea_data/")
        print("    sft_vl_train_shuffle.jsonl")
        print("    sft_vl_valid_shuffle.jsonl")
        print("    Anthracnose/ (炭疽病图片)")
        print("    algal leaf/ (藻斑病图片)")
        print("    ...")
        sys.exit(1)

    if not os.path.exists(VAL_INPUT):
        print(f"\n❌ 错误: 找不到验证数据 {VAL_INPUT}")
        sys.exit(1)

    train_samples = process_file(
        TRAIN_INPUT,
        os.path.join(OUTPUT_DIR, "tea_train.json")
    )
    val_samples = process_file(
        VAL_INPUT,
        os.path.join(OUTPUT_DIR, "tea_val.json")
    )

    print("\n" + "=" * 50)
    print("转换完成！")
    print(f"训练样本: {len(train_samples)}")
    print(f"验证样本: {len(val_samples)}")
    print(f"数据保存在: {OUTPUT_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
