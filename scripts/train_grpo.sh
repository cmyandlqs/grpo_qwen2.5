#!/bin/bash
# GRPO 训练 Qwen2.5-0.5B on GSM8K
# 使用方法：编辑下面的配置，然后运行

# ============================================
# 配置区域（根据你的环境修改）
# ============================================

# 模型配置
MODEL_PATH="/mnt/workspace/qwen2.5-0.5B-base"
MODEL_TYPE="qwen2"

# 数据路径
DATA_PATH="/mnt/workspace/dataset_eval/gsm8k_grpo_train.jsonl"

# 输出路径（自动添加时间戳）
OUTPUT_DIR="/mnt/workspace/output/grpo_qwen2.5_0.5b_$(date +%Y%m%d_%H%M%S)"

# ============================================
# GRPO 核心参数
# ============================================

NUM_GENERATIONS=4         # G 值：每个 prompt 生成几个输出
REWARD_FUNCS="accuracy"   # 奖励函数：accuracy（准确率）
TEMPERATURE=1.0          # 采样温度

# ============================================
# 训练参数
# ============================================

BATCH_SIZE=8              # 批大小（根据显存调整：24GB→8，16GB→4）
LEARNING_RATE=1e-5        # 学习率
NUM_EPOCHS=1              # 训练轮数

# LoRA 参数
LORA_RANK=64              # LoRA 秩
LORA_ALPHA=128            # LoRA alpha

# ============================================
# SwanLab 监控（可选）
# ============================================

# 获取 API Key: https://swanlab.cn/
# 留空则不使用 SwanLab
SWANLAB_KEY="INxp6ym1fOllPByTREaiD"

SWANLAB_PROJECT="GRPO-Qwen-GSM8K"
SWANLAB_RUN="qwen-0.5b-$(date +%Y%m%d_%H%M%S)"

# ============================================
# 执行训练
# ============================================

echo "=========================================="
echo "GRPO 训练"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "数据: $DATA_PATH"
echo "输出: $OUTPUT_DIR"
echo "G 值: $NUM_GENERATIONS"
echo "批大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "训练轮数: $NUM_EPOCHS"
if [ -n "$SWANLAB_KEY" ]; then
    echo "监控: SwanLab (https://swanlab.cn)"
else
    echo "监控: 本地日志"
fi
echo "=========================================="
echo ""

# 设置 SwanLab（如果提供了 key）
REPORT_TO="tensorboard"
if [ -n "$SWANLAB_KEY" ]; then
    export SWANLAB_API_KEY="$SWANLAB_KEY"
    REPORT_TO="swanlab"
    echo "✓ SwanLab 已启用"
fi

# ============================================
# 数据集转换（如果需要）
# ============================================

RAW_DATA_PATH="/mnt/workspace/dataset_eval/gsm8k_data"

if [ ! -f "$DATA_PATH" ]; then
    echo "⚠️  GRPO格式数据集不存在，正在转换..."
    echo "   原始数据: $RAW_DATA_PATH"
    echo "   输出文件: $DATA_PATH"

    # 使用转换脚本
    if [ -f "utils/convert_gsm8k_to_grpo.py" ]; then
        python utils/convert_gsm8k_to_grpo.py \
            --input "$RAW_DATA_PATH" \
            --output "$DATA_PATH" \
            --split train
    else
        # 内嵌转换逻辑
        python -c "
import sys
import json
import re
from datasets import load_from_disk

def extract_answer(answer):
    match = re.search(r'####\s*(\d+(?:\.\d+)?)', answer)
    if match:
        return match.group(1)
    numbers = re.findall(r'\d+(?:\.\d+)?', answer)
    return numbers[-1] if numbers else answer.strip()

dataset = load_from_disk('$RAW_DATA_PATH')
if hasattr(dataset, 'keys'):
    dataset = dataset['train']

print(f'转换 {len(dataset)} 个样本...')

with open('$DATA_PATH', 'w', encoding='utf-8') as f:
    for item in dataset:
        solution = extract_answer(item.get('answer', ''))
        data = {
            'messages': [{'role': 'user', 'content': item.get('question', '')}],
            'solution': solution
        }
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f'✓ 保存到: $DATA_PATH')
"
    fi

    if [ $? -eq 0 ]; then
        echo "✓ 数据集转换完成"
    else
        echo "❌ 数据集转换失败"
        exit 1
    fi
else
    echo "✓ GRPO格式数据集已存在: $DATA_PATH"
fi

echo ""

# 获取 ms-swift rlhf.py 路径
RLHF_SCRIPT=$(python -c 'import swift; import os; print(os.path.dirname(swift.__file__) + "/cli/rlhf.py")')

# 执行训练
python -m torch.distributed.run \
    --nproc_per_node 1 \
    "$RLHF_SCRIPT" \
    --rlhf_type grpo \
    --model $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --template qwen2_5 \
    --dataset $DATA_PATH \
    --tuner_type lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --dtype bfloat16 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --num_generations $NUM_GENERATIONS \
    --reward_funcs $REWARD_FUNCS \
    --temperature $TEMPERATURE \
    --top_p 0.9 \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 2 \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --output_dir $OUTPUT_DIR \
    --logging_dir $OUTPUT_DIR/logs \
    --gradient_checkpointing true \
    --report_to $REPORT_TO \
    --project_name $SWANLAB_PROJECT \
    --run_name $SWANLAB_RUN \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "输出: $OUTPUT_DIR"
echo ""

if [ -n "$SWANLAB_KEY" ]; then
    echo "查看监控: https://swanlab.cn"
else
    echo "查看日志: tail -f $OUTPUT_DIR/logs/trainer.log"
fi

echo ""
echo "评估模型:"
echo "  python tests/run_gsm8k_eval.py --model $OUTPUT_DIR/checkpoint-final"
echo "=========================================="
