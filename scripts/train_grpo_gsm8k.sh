#!/bin/bash
# GRPO 训练 Qwen2.5-0.5B on GSM8K
# 基于 ms-swift 框架

# ============================================
# 配置区域 - 根据你的环境修改
# ============================================

# 模型配置
MODEL_PATH="/mnt/workspace/qwen2.5-0.5B-base"  # 基座模型路径
MODEL_TYPE="qwen2.5"                            # 模型类型

# 数据配置
DATA_PATH="/mnt/workspace/dataset_eval/gsm8k_data"  # GSM8K 数据集路径

# 输出配置
OUTPUT_DIR="/mnt/workspace/output/grpo_qwen2.5_0.5b_gsm8k_$(date +%Y%m%d_%H%M%S)"
LOGGING_DIR="$OUTPUT_DIR/logs"

# ============================================
# GRPO 核心参数
# ============================================

# G 值：每个 prompt 生成多少个输出
# 较大值 = 更好的探索效果，但更慢
NUM_GENERATIONS=8

# 奖励函数
# accuracy: 答案准确性（数学题必需）
# format: 输出格式合规性
# cosine: 与参考答案的相似度
# repetition: 惩罚重复
# soft_overlong: 惩罚过长输出
REWARD_FUNCS="accuracy format"

# 奖励函数权重（如果使用多个）
REWARD_WEIGHTS="0.8 0.2"

# 采样参数
TEMPERATURE=1.0      # 采样温度（0.7-1.0）
TOP_P=0.9           # Top-p 采样（0.9-0.95）
TOP_K=50            # Top-k 采样

# ============================================
# 训练参数
# ============================================

# 训练类型
TRAIN_TYPE="lora"           # 使用 LoRA 训练（推荐）
LORA_RANK=64                # LoRA 秩（越大效果越好但更慢）
LORA_ALPHA=128              # LoRA alpha（通常是 rank 的 2 倍）
LORA_TARGET_MODULES="all"   # LoRA 目标模块

# 批处理
PER_DEVICE_BATCH_SIZE=8     # 每设备批大小（根据显存调整）
GRADIENT_ACCUMULATION_STEPS=2  # 梯度累积步数

# 优化器
LEARNING_RATE=1e-5          # 学习率
WARMUP_RATIO=0.03           # 预热比例
NUM_TRAIN_EPOCHS=1          # 训练轮数
MAX_STEPS=-1                # 最大步数（-1 表示使用 epochs）

# 保存与评估
SAVE_TOTAL_LIMIT=3          # 保存最近几个 checkpoint
SAVE_STEPS=500              # 每 N 步保存一次
EVAL_STEPS=500              # 每 N 步评估一次
LOGGING_STEPS=10            # 每 N 步记录一次

# ============================================
# 模型参数
# ============================================

MAX_LENGTH=2048             # 输入最大长度
MAX_COMPLETION_LENGTH=1024  # 输出最大长度
TORCH_DTYPE="bfloat16"      # 数据类型

# ============================================
# 加速参数
# ============================================

USE_VLLM=true               # 使用 vLLM 加速
VLLM_MODE="colocate"        # vLLM 模式：colocate 或 server
VLLM_GPU_MEMORY_UTILIZATION=0.9  # vLLM 显存利用率

GRADIENT_CHECKPOINTING=true # 梯度检查点（节省显存）

# ============================================
# 日志与监控
# ============================================

REPORT_TO="tensorboard"     # 日志工具：tensorboard, wandb, none
LOG_COMPLETIONS=true        # 记录生成的样本

# ============================================
# 执行训练
# ============================================

echo "=============================================="
echo "GRPO 训练配置"
echo "=============================================="
echo "模型: $MODEL_PATH"
echo "数据: $DATA_PATH"
echo "输出: $OUTPUT_DIR"
echo "G 值: $NUM_GENERATIONS"
echo "奖励函数: $REWARD_FUNCS"
echo "批大小: $PER_DEVICE_BATCH_SIZE × $GRADIENT_ACCUMULATION_STEPS"
echo "学习率: $LEARNING_RATE"
echo "训练轮数: $NUM_TRAIN_EPOCHS"
echo "=============================================="
echo ""

# 获取 ms-swift 的 rlhf.py 路径
RLHF_SCRIPT=$(python -c 'import swift; import os; print(os.path.dirname(swift.__file__) + "/cli/rlhf.py")')

echo "开始训练..."
echo "训练脚本: $RLHF_SCRIPT"
echo ""

# 执行训练
python -m torch.distributed.run \
    --nproc_per_node 1 \
    "$RLHF_SCRIPT" \
    --rlhf_type grpo \
    --model $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --dataset $DATA_PATH \
    --train_type $TRAIN_TYPE \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --torch_dtype $TORCH_DTYPE \
    --use_vllm $USE_VLLM \
    --vllm_mode $VLLM_MODE \
    --vllm_gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --num_generations $NUM_GENERATIONS \
    --reward_funcs $REWARD_FUNCS \
    --reward_weights $REWARD_WEIGHTS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --max_steps $MAX_STEPS \
    --max_length $MAX_LENGTH \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --report_to $REPORT_TO \
    --log_completions $LOG_COMPLETIONS \
    --dataloader_num_workers 4

echo ""
echo "=============================================="
echo "训练完成！"
echo "=============================================="
echo "输出目录: $OUTPUT_DIR"
echo "日志目录: $LOGGING_DIR"
echo ""
echo "查看训练曲线："
echo "  tensorboard --logdir $LOGGING_DIR"
echo ""
echo "评估训练后的模型："
echo "  python tests/run_gsm8k_eval.py --model $OUTPUT_DIR/checkpoint-xxx"
echo "=============================================="
