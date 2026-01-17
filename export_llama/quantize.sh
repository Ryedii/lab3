#!/bin/bash

# 量化流程脚本 - 第二步：模型量化
# 该脚本用于完成实验第二步：收集激活值范围并量化模型

set -e  # 遇到错误立即退出

# 默认参数
MODEL_NAME="${MODEL_NAME:-./model/TinyLlama-1.1B-Chat-v1.0}"
DATASET_PATH="${DATASET_PATH:-./dataset/wikitext-103-v1/train-00000-of-00002.parquet}"
ACT_OUTPUT="${ACT_OUTPUT:-./act/TinyLlama-chat-v1.0-act.pt}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
SEQ_LEN="${SEQ_LEN:-512}"

echo "========================================="
echo "开始执行量化流程（实验第二步）"
echo "========================================="
echo "模型路径: $MODEL_NAME"
echo "数据集路径: $DATASET_PATH"
echo "激活值输出路径: $ACT_OUTPUT"
echo "样本数量: $NUM_SAMPLES"
echo "序列长度: $SEQ_LEN"
echo "========================================="

# 检查模型路径是否存在
if [ ! -d "$MODEL_NAME" ]; then
    echo "错误: 模型路径不存在: $MODEL_NAME"
    echo "请检查模型路径是否正确"
    exit 1
fi

# 检查数据集路径是否存在
if [ ! -f "$DATASET_PATH" ]; then
    echo "错误: 数据集文件不存在: $DATASET_PATH"
    echo "请检查数据集路径是否正确"
    exit 1
fi

# 切换到脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "步骤 1: 生成激活值范围（收集校准数据）"
echo "----------------------------------------"
python generate_act_scales.py \
    --model-name "$MODEL_NAME" \
    --output-path "$ACT_OUTPUT" \
    --dataset-path "$DATASET_PATH" \
    --num-samples "$NUM_SAMPLES" \
    --seq-len "$SEQ_LEN"

if [ $? -eq 0 ]; then
    echo "✓ 激活值范围生成成功: $ACT_OUTPUT"
else
    echo "✗ 激活值范围生成失败"
    exit 1
fi

echo ""
echo "========================================="
echo "量化流程（第二步）完成！"
echo "========================================="
echo "生成的激活值文件: $ACT_OUTPUT"
echo ""
echo "下一步：使用 export_llama.py 导出ONNX模型"
echo "========================================="

