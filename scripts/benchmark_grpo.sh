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
# 工具函数
# ============================================

# 浮点数计算（替代 bc）
calc() {
    awk "BEGIN {print $@}"
}

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
RLHF_SCRIPT=$(python -c 'import swift; import os; print(os.path.dirname(swift.__file__) + "/cli/rlhf.py")' 2>/dev/null)

if [ -z "$RLHF_SCRIPT" ] || [ ! -f "$RLHF_SCRIPT" ]; then
    echo "错误: 无法找到 ms-swift rlhf.py"
    echo "请确保 ms-swift 已正确安装"
    exit 1
fi

echo "✓ 找到训练脚本: $RLHF_SCRIPT"
echo ""

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

        echo "[正在测试: $config_name]" >&2

        # 记录开始时间
        start_time=$(date +%s)

        # 运行训练（后台）
        python -m torch.distributed.run \
            --nproc_per_node 1 \
            "$RLHF_SCRIPT" \
            --rlhf_type grpo \
            --model "$MODEL_PATH" \
            --model_type qwen2 \
            --template qwen2_5 \
            --dataset "$DATA_PATH" \
            --dataset_num_proc 1 \
            --tuner_type lora \
            --lora_rank 64 \
            --lora_alpha 128 \
            --use_vllm true \
            --vllm_mode colocate \
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

        # 等待一下确保训练启动
        sleep 3

        # 检查进程是否还在运行
        if ! kill -0 $train_pid 2>/dev/null; then
            echo "✗ 训练启动失败，查看日志: $test_output/train.log" >&2
            printf "%-18s | %-12s | %-12s | %-15s | %-15s\n" "$config_name" "失败" "失败" "失败" "失败"
            continue
        fi

        echo "  [训练已启动，PID: $train_pid]" >&2

        # 监控显存
        max_vram=0
        vram_sum=0
        vram_count=0
        last_output_time=$(date +%s)

        while kill -0 $train_pid 2>/dev/null; do
            # 获取显存
            vram_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1)

            if [ -n "$vram_mb" ]; then
                vram_gb=$(calc "$vram_mb / 1024")

                # 更新最大显存
                max_vram=$(calc "$vram_gb > $max_vram ? $vram_gb : $max_vram")

                # 累加用于计算平均值
                vram_sum=$(calc "$vram_sum + $vram_gb")
                ((vram_count++))

                # 每5秒输出一次进度
                current_time=$(date +%s)
                if [ $((current_time - last_output_time)) -ge 5 ]; then
                    echo "  [显存: ${vram_gb} GB, 已运行: $((current_time - start_time))秒]" >&2
                    last_output_time=$current_time
                fi
            fi

            sleep 1
        done

        # 等待训练完成
        wait $train_pid
        exit_code=$?

        # 计算时间
        end_time=$(date +%s)
        duration=$((end_time - start_time))

        if [ $exit_code -ne 0 ]; then
            echo "  ✗ 训练异常退出 (退出码: $exit_code)" >&2
            echo "  查看日志: cat $test_output/train.log" >&2
            printf "%-18s | %-12s | %-12s | %-15s | %-15s\n" "$config_name" "异常" "异常" "${duration}秒" "异常"
        else
            # 计算平均显存
            if [ $vram_count -gt 0 ]; then
                avg_vram=$(calc "$vram_sum / $vram_count")
            else
                avg_vram=0
            fi

            # 估算完整训练时间（假设完整数据集约 8700 个样本）
            total_samples=8700
            tested_samples=$((num_gen * batch_size * MAX_STEPS))
            estimated_full_time=$(calc "$duration * $total_samples / $tested_samples / 60")

            echo "  ✓ 测试完成" >&2

            # 输出结果
            printf "%-18s | %-12s | %-12s | %-15s | %-15s\n" \
                "$config_name" \
                "$(calc "$max_vram")" \
                "$(calc "$avg_vram")" \
                "${duration}秒" \
                "${estimated_full_time}分钟"
        fi

        echo "" >&2
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
