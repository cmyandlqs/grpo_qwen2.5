#!/bin/bash
# GRPO 参数快速测试
# 测试不同 NUM_GENERATIONS 和 BATCH_SIZE 的显存占用和训练时间

set -e

# ============================================
# 配置区域（根据环境修改）
# ============================================

MODEL_PATH="/mnt/workspace/qwen2.5-0.5B-base"
DATA_PATH="/mnt/workspace/dataset_eval/gsm8k_data"
OUTPUT_DIR="/mnt/workspace/output/grpo_benchmark"

# 测试参数
TEST_NUM_GENERATIONS=(4 8 16)
TEST_BATCH_SIZES=(8 16)

# 快速测试配置
SAMPLE_SIZE=100              # 测试样本数（用于估算完整训练时间）
MAX_STEPS=20                 # 最大训练步数

# ============================================
# 开始测试
# ============================================

echo "=========================================="
echo "GRPO 参数快速测试"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "数据: $DATA_PATH (样本数: $SAMPLE_SIZE)"
echo ""
echo "测试配置:"
echo "  NUM_GENERATIONS: ${TEST_NUM_GENERATIONS[@]}"
echo "  BATCH_SIZE: ${TEST_BATCH_SIZES[@]}"
echo ""
echo "=========================================="
echo ""

# 获取 ms-swift rlhf.py 路径
RLHF_SCRIPT=$(python -c 'import swift; import os; print(os.path.dirname(swift.__file__) + "/cli/rlhf.py")')

# 表头
printf "%-18s | %-12s | %-12s | %-15s | %-15s\n" "配置" "最大显存(GB)" "平均显存(GB)" "测试时间(秒)" "估算完整时间"
echo "-----------------------------------------------------------------------------------------------------"

# 遍历所有配置组合
for num_gen in "${TEST_NUM_GENERATIONS[@]}"; do
    for batch_size in "${TEST_BATCH_SIZES[@]}"; do

        config_name="G${num_gen}_B${batch_size}"
        test_output="$OUTPUT_DIR/$config_name"

        # 创建输出目录
        mkdir -p "$test_output"

        # 记录开始时间
        start_time=$(date +%s)

        # 运行训练（后台）
        python -m torch.distributed.run \
            --nproc_per_node 1 \
            "$RLHF_SCRIPT" \
            --rlhf_type grpo \
            --model "$MODEL_PATH" \
            --model_type qwen2.5 \
            --dataset "$DATA_PATH" \
            --dataset_num_proc 1 \
            --train_type lora \
            --lora_rank 64 \
            --lora_alpha 128 \
            --lora_target_modules all \
            --torch_dtype bfloat16 \
            --use_vllm true \
            --num_generations $num_gen \
            --reward_funcs accuracy \
            --per_device_train_batch_size $batch_size \
            --learning_rate 1e-5 \
            --max_steps $MAX_STEPS \
            --output_dir "$test_output" \
            --logging_steps 5 \
            --save_steps 999999 \
            --report_to none \
            --dataloader_num_workers 0 \
            > "$test_output/train.log" 2>&1 &

        train_pid=$!

        # 监控显存
        max_vram=0
        vram_sum=0
        vram_count=0

        while kill -0 $train_pid 2>/dev/null; do
            vram_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
            vram_gb=$(echo "scale=2; $vram_mb / 1024" | bc)

            if (( $(echo "$vram_gb > $max_vram" | bc -l) )); then
                max_vram=$vram_gb
            fi

            vram_sum=$(echo "$vram_sum + $vram_gb" | bc)
            ((vram_count++))

            sleep 1
        done

        # 等待训练完成
        wait $train_pid

        # 计算时间
        end_time=$(date +%s)
        duration=$((end_time - start_time))

        # 计算平均显存
        avg_vram=$(echo "scale=2; $vram_sum / $vram_count" | bc)

        # 估算完整训练时间（假设完整数据集约 8700 个样本）
        # 计算公式：测试时间 × (8700 / (num_gen × batch_size × steps))
        # 简化估算：基于样本数比例
        total_samples=8700
        tested_samples=$((num_gen * batch_size * MAX_STEPS))
        estimated_full_time=$(echo "scale=0; $duration * $total_samples / $tested_samples / 60" | bc)

        # 输出结果
        printf "%-18s | %-12s | %-12s | %-15s | %-15s\n" \
            "$config_name" \
            "$max_vram" \
            "$avg_vram" \
            "${duration}秒" \
            "${estimated_full_time}分钟"

        # 清理测试输出（保留日志）
        # rm -rf "$test_output"
    done
done

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "说明："
echo "  - 最大显存：训练过程中的峰值显存"
echo "  - 平均显存：训练过程中的平均显存"
echo "  - 测试时间：$MAX_STEPS 步的实际用时"
echo "  - 估算完整时间：基于 $SAMPLE_SIZE 样本估算的完整数据集训练时间"
echo ""
echo "日志保存在: $OUTPUT_DIR/<配置>/train.log"
