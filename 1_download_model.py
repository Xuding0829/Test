"""
步骤1：下载 Qwen2-VL-2B-Instruct 模型
使用 ModelScope 下载，速度较快
"""
from modelscope import snapshot_download

print("开始下载 Qwen2-VL-2B-Instruct 模型...")
print("模型较大，请耐心等待...")

model_dir = snapshot_download(
    "Qwen/Qwen2-VL-2B-Instruct",
    cache_dir="./",
    revision="master"
)

print(f"\n✅ 模型下载完成！")
print(f"模型路径: {model_dir}")