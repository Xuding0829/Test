"""
步骤1：下载 Qwen2-VL-2B-Instruct 模型
使用 ModelScope 下载（国内速度快）
下载完成后模型保存在 ./Qwen/Qwen2-VL-2B-Instruct/
"""
import os
import sys

try:
    from modelscope import snapshot_download
except ImportError:
    print("❌ 未安装 modelscope，请先运行: pip install modelscope")
    sys.exit(1)

print("=" * 50)
print("下载 Qwen2-VL-2B-Instruct 模型")
print("=" * 50)
print("模型约 3.9GB，请耐心等待...\n")

model_dir = snapshot_download(
    "Qwen/Qwen2-VL-2B-Instruct",
    cache_dir="./",
    revision="master"
)

print(f"\n✅ 模型下载完成！")
print(f"模型路径: {model_dir}")

# 验证关键文件
key_files = ["config.json", "model.safetensors.index.json"]
for f in key_files:
    path = os.path.join(model_dir, f)
    if os.path.exists(path):
        print(f"  ✅ {f}")
    else:
        print(f"  ❌ 缺少 {f}")
