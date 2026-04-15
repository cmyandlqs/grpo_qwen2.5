#!/bin/bash
# SFT 训练 Qwen2.5-0.5B on GSM8K
# 使用方法：编辑下面的配置，然后运行

# ============================================
# 配置区域
# ============================================

MODEL_PATH="/mnt/workspace/qwen2.5-0.5B-base"
MODEL_TYPE="qwen2.5"
DATA_PATH="/mnt/workspace/dataset_eval/gsm8k_data"
OUTPUT_DIR="/mnt/workspace/output/sft_qwen2.5_0.5b_$(date +%Y%m%d_%H%M%S)"

# ============================================
# 训练参数
# ============================================

BATCH_SIZE=8              # 批大小
LEARNING_RATE=2e-5        # 学习率（SFT 可以用较高的学习率）
NUM_EPOCHS=3              # 训练轮数（SFT 可以多训练几轮）

# LoRA 参数
LORA_RANK=64
LORA_ALPHA=128

# ============================================
# SwanLab 监控（可选）
# ============================================

SWANLAB_KEY=""
SWANLAB_PROJECT="SFT-Qwen-GSM8K"
SWANLAB_RUN="qwen-0.5b-$(date +%Y%m%d_%H%M%S)"

# ============================================
# 执行训练
# ============================================

echo "=========================================="
echo "SFT 训练"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "数据: $DATA_PATH"
echo "批大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "训练轮数: $NUM_EPOCHS"
echo "=========================================="
echo ""

# 设置 SwanLab
REPORT_TO="tensorboard"
if [ -n "$SWANLAB_KEY" ]; then
    export SWANLAB_API_KEY="$SWANLAB_KEY"
    REPORT_TO="swanlab"
    echo "✓ SwanLab 已启用"
fi

# 获取 ms-swift sft.py 路径
SFT_SCRIPT=$(python -c 'import swift; import os; print(os.path.dirname(swift.__file__) + "/cli/sft.py")')

# 执行训练
python -m torch.distributed.run \
    --nproc_per_node 1 \
    "$SFT_SCRIPT" \
    --model_type $MODEL_TYPE \
    --model_id_or_path $MODEL_PATH \
    --dataset $DATA_PATH \
    --sft_type lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_target_modules all \
    --torch_dtype bfloat16 \
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
echo "评估: python tests/run_gsm8k_eval.py --model $OUTPUT_DIR/checkpoint-final"
echo "=========================================="
