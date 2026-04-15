#!/bin/bash
# PPO 训练 Qwen2.5-0.5B on GSM8K
# ⚠️ 注意：需要先训练 Reward Model
# 推荐使用 GRPO 代替 PPO

# ============================================
# 配置区域
# ============================================

MODEL_PATH="/mnt/workspace/qwen2.5-0.5B-base"
REWARD_MODEL_PATH="/mnt/workspace/reward_models/gsm8k_reward"  # 需要预先训练
MODEL_TYPE="qwen2.5"
DATA_PATH="/mnt/workspace/dataset_eval/gsm8k_data"
OUTPUT_DIR="/mnt/workspace/output/ppo_qwen2.5_0.5b_$(date +%Y%m%d_%H%M%S)"

# ============================================
# 训练参数
# ============================================

BATCH_SIZE=4              # PPO 批大小通常较小
LEARNING_RATE=1e-5        # PPO 学习率
NUM_EPOCHS=1              # 训练轮数

# PPO 特有参数
PPO_EPOCHS=4              # 每个 batch 重复训练几次
CLIP_RANGE=0.2            # PPO 裁剪范围
VF_COEF=0.1               # Value loss 系数

# LoRA 参数
LORA_RANK=64
LORA_ALPHA=128

# ============================================
# SwanLab 监控（可选）
# ============================================

SWANLAB_KEY=""
SWANLAB_PROJECT="PPO-Qwen-GSM8K"
SWANLAB_RUN="qwen-0.5b-$(date +%Y%m%d_%H%M%S)"

# ============================================
# 执行训练
# ============================================

echo "=========================================="
echo "PPO 训练"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "Reward Model: $REWARD_MODEL_PATH"
echo "数据: $DATA_PATH"
echo "=========================================="
echo ""
echo "⚠️  注意：PPO 需要 Reward Model"
echo "   推荐使用 GRPO 代替！"
echo ""

# 检查 Reward Model
if [ ! -d "$REWARD_MODEL_PATH" ]; then
    echo "❌ 错误: Reward Model 不存在"
    echo "   请先训练 Reward Model，或使用 GRPO"
    echo "   bash scripts/train_grpo.sh"
    exit 1
fi

# 设置 SwanLab
REPORT_TO="tensorboard"
if [ -n "$SWANLAB_KEY" ]; then
    export SWANLAB_API_KEY="$SWANLAB_KEY"
    REPORT_TO="swanlab"
    echo "✓ SwanLab 已启用"
fi

# 获取 ms-swift ppo.py 路径
PPO_SCRIPT=$(python -c 'import swift; import os; print(os.path.dirname(swift.__file__) + "/cli/ppo.py")')

# 执行训练
python -m torch.distributed.run \
    --nproc_per_node 1 \
    "$PPO_SCRIPT" \
    --rlhf_type ppo \
    --model $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --reward_model $REWARD_MODEL_PATH \
    --dataset $DATA_PATH \
    --train_type lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_target_modules all \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 2 \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --ppo_epochs $PPO_EPOCHS \
    --clip_range $CLIP_RANGE \
    --vf_coef $VF_COEF \
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
echo ""
echo "💡 提示：下次试试 GRPO，更简单高效！"
echo "=========================================="
