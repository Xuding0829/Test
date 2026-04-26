$ErrorActionPreference = "Stop"

Write-Host "========================================="
Write-Host "Qwen2-VL-2B 茶叶病虫害微调训练"
Write-Host "========================================="

Write-Host "[1/3] 转换数据格式..."
python 2_convert_data.py

if (-not (Test-Path "Qwen")) {
    Write-Host "[2/3] 下载模型..."
    python 1_download_model.py
} else {
    Write-Host "[跳过] 模型已下载"
}

Write-Host "[3/3] 开始训练..."
python 3_train.py

Write-Host "========================================="
Write-Host "训练完成！"
Write-Host "模型保存在: output/tea_disease_vl/"
Write-Host "========================================="
