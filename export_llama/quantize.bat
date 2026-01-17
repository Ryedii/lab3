@echo off
REM 量化流程脚本 - 第二步：模型量化（Windows版本）
REM 该脚本用于完成实验第二步：收集激活值范围并量化模型

setlocal enabledelayedexpansion

REM 默认参数
if "%MODEL_NAME%"=="" set MODEL_NAME=./model/TinyLlama-1.1B-Chat-v1.0
if "%DATASET_PATH%"=="" set DATASET_PATH=./dataset/wikitext-103-v1/train-00000-of-00002.parquet
if "%ACT_OUTPUT%"=="" set ACT_OUTPUT=./act/TinyLlama-chat-v1.0-act.pt
if "%NUM_SAMPLES%"=="" set NUM_SAMPLES=512
if "%SEQ_LEN%"=="" set SEQ_LEN=512

echo =========================================
echo 开始执行量化流程（实验第二步）
echo =========================================
echo 模型路径: %MODEL_NAME%
echo 数据集路径: %DATASET_PATH%
echo 激活值输出路径: %ACT_OUTPUT%
echo 样本数量: %NUM_SAMPLES%
echo 序列长度: %SEQ_LEN%
echo =========================================

REM 检查模型路径是否存在
if not exist "%MODEL_NAME%" (
    echo 错误: 模型路径不存在: %MODEL_NAME%
    echo 请检查模型路径是否正确
    exit /b 1
)

REM 检查数据集路径是否存在
if not exist "%DATASET_PATH%" (
    echo 错误: 数据集文件不存在: %DATASET_PATH%
    echo 请检查数据集路径是否正确
    exit /b 1
)

REM 切换到脚本所在目录
cd /d "%~dp0"

echo.
echo 步骤 1: 生成激活值范围（收集校准数据）
echo ----------------------------------------
python generate_act_scales.py --model-name "%MODEL_NAME%" --output-path "%ACT_OUTPUT%" --dataset-path "%DATASET_PATH%" --num-samples %NUM_SAMPLES% --seq-len %SEQ_LEN%

if %ERRORLEVEL% EQU 0 (
    echo ✓ 激活值范围生成成功: %ACT_OUTPUT%
) else (
    echo ✗ 激活值范围生成失败
    exit /b 1
)

echo.
echo =========================================
echo 量化流程（第二步）完成！
echo =========================================
echo 生成的激活值文件: %ACT_OUTPUT%
echo.
echo 下一步：使用 export_llama.py 导出ONNX模型
echo =========================================

endlocal

